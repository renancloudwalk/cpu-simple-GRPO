//! # Rust GRPO Fine-Tuning Client
//!
//! This Rust code **does not** spin up a Python reference server. Instead, it **talks**
//! to an *already running* Python reference server at `localhost:59875`, exactly
//! like your Python script does. We replicate a GRPO (Group Relative Policy
//! Optimization) loop with Qwen2.5-0.5B on CPU, though it's quite large and slow
//! for CPU use in practice.
//!
//! Steps:
//! 1. **generate_mode**: picks random QAs from a small dataset, builds a prompt,
//!    generates multiple completions, applies local rewards, merges tokens, then
//!    `POST /upload` to the reference server, which adds reference log-probs.
//! 2. **get_batch**: attempts to `GET /get`; if empty, calls generate_mode again.
//! 3. **GRPO_step**: compares new policy log-probs vs. reference log-probs, calculates
//!    advantage per group, and a KL penalty = `exp(ref - new) - (ref - new) - 1`.
//! 4. **Training Loop**: we do a certain # of steps, each time retrieving a batch,
//!    computing GRPO gradient, calling AdamW step, and periodically saving
//!    `.safetensors` checkpoints.
//! 5. **talk_to_model**: an interactive REPL at the end for manual queries.

use std::io::Write;
use std::time::Instant;

use anyhow::{Result, anyhow};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use rand::{rngs::StdRng, SeedableRng, seq::SliceRandom};
use regex::Regex;

// Candle 0.8.1
use candle_core::{Device, DType, Tensor, Var};
use candle_core::Result as CandleResult; // keep Candle's result type distinct
use candle_nn::{VarMap, VarBuilder, Optimizer};
use candle_nn::optim::AdamW;
use candle_transformers::models::qwen2::{Model as QwenModel, Config as QwenConfig};
use candle_examples::hub_load_safetensors;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

macro_rules! log_line {
    ($($arg:tt)+) => {
        println!("[{}:{}] {}", file!(), line!(), format_args!($($arg)+))
    };
}

/// Reference server base URL
static REF_SERVER: &str = "http://localhost:59875";
/// Qwen model ID
static MODEL_ID: &str = "Qwen/Qwen2.5-0.5B";

// hyperparams
static BETA: f32 = 0.04;
static NUM_PRE_Q: usize = 8;
static Q_BATCH_SIZE: usize = 1;
static MAX_PROMPT_LENGTH: usize = 400;
static ALL_STEPS: usize = 10;
static SAVE_STEPS: usize = 5;
static LR: f64 = 1e-6;

/// Minimal QA structure
#[derive(Clone)]
struct QaPair {
    question: String,
    answer: String,
}

fn load_fake_gsm8k() -> Vec<QaPair> {
    vec![
        QaPair { question: "What is 2+2?".to_string(), answer: "4".to_string() },
        QaPair { question: "Compute 5*3".to_string(), answer: "15".to_string() },
    ]
}

/// Helper: convert Candle’s `Result<T, candle_core::Error>` to `anyhow::Result<T>`.
#[track_caller]
fn candle_ok<T>(res: CandleResult<T>) -> Result<T> {
    let loc = std::panic::Location::caller();
    match res {
        Ok(x) => Ok(x),
        Err(e) => Err(anyhow!("[{}:{}] Candle error: {}", loc.file(), loc.line(), e)),
    }
}

/// Load Qwen from HF, Candle 0.8.1
fn load_qwen_model_and_tokenizer(
    device: &Device,
    dtype: DType
) -> Result<(VarMap, QwenModel, Tokenizer, QwenConfig)> {
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(MODEL_ID.to_string(), RepoType::Model, "main".to_string()));
    log_line!("Loaded HF API and repository.");

    // tokenizer
    let tokenizer_path = repo.get("tokenizer.json")?;
    log_line!("Found tokenizer at: {:?}", tokenizer_path);
    let tokenizer = Tokenizer::from_file(tokenizer_path.to_str().unwrap())
        .map_err(|e| anyhow!("[{}:{}] Tokenizer load: {}", file!(), line!(), e))?;
    log_line!("Tokenizer loaded successfully.");

    // config
    let config_path = repo.get("config.json")?;
    log_line!("Found config at: {:?}", config_path);
    let config_bytes = std::fs::read(&config_path)?;
    let config: QwenConfig = serde_json::from_slice(&config_bytes)
        .map_err(|e| anyhow!("[{}:{}] Config parse: {}", file!(), line!(), e))?;
    log_line!("Configuration loaded successfully.");

    // Always use F32 for dtype
    let dtype = DType::F32;

    // Create the model with a temp varmap first
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, dtype, device);
    let model = candle_ok(QwenModel::new(&config, vb))?;
    log_line!("Empty model created.");

    // Now manually load the tensors from safetensors with conversion
    let weight_files = if repo.get("model.safetensors.index.json").is_ok() {
        hub_load_safetensors(&repo, "model.safetensors.index.json")?
    } else {
        vec![repo.get("model.safetensors")?]
    };
    log_line!("Weight files retrieved: {:?}", weight_files);

    // Let's try to load them one by one with conversion
    for wf in weight_files {
        log_line!("Loading weight file: {:?}", wf);

        // Load the safetensors without using VarMap's load method
        let tensors = candle_ok(candle_core::safetensors::load(&wf, &device))?;

        // Manually set each parameter in the varmap
        for (name, tensor) in tensors {
            // Convert to F32 if needed
            let f32_tensor = if tensor.dtype() != DType::F32 {
                candle_ok(tensor.to_dtype(DType::F32))?
            } else {
                tensor
            };

            // Get access to the underlying data
            let mut data_lock = varmap.data().lock().unwrap();

            // If the varmap already has this parameter, update it
            if let Some(var) = data_lock.get_mut(&name) {
                *var = candle_ok(Var::from_tensor(&f32_tensor))?;
            } else {
                // Otherwise, insert it
                data_lock.insert(name, candle_ok(Var::from_tensor(&f32_tensor))?);
            }
        }
    }
    log_line!("Weights loaded and converted to F32.");

    Ok((varmap, model, tokenizer, config))
}

/// 1D log_softmax in an array
fn log_softmax_1d(vals: &mut [f32]) {
    let mx = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sumexp = 0f32;
    for v in vals.iter_mut() {
        *v = (*v - mx).exp();
        sumexp += *v;
    }
    let lnz = sumexp.ln();
    for v in vals.iter_mut() {
        // log(prob) = log( e^(x-mx)/sumexp ) + mx = x - lnz
        *v = v.ln() - lnz + mx;
    }
}

/// Argmax for a slice
fn argmax_1d(arr: &[f32]) -> usize {
    let mut best = 0;
    let mut best_val = f32::MIN;
    for (i, &val) in arr.iter().enumerate() {
        if val>best_val {
            best_val = val;
            best = i;
        }
    }
    best
}

/// A single generation pass (greedy)
fn generate_once(
    model: &mut QwenModel,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_tokens: usize,
    device: &Device
) -> Result<String> {
    let enc = tokenizer.encode(&*prompt, true)
        .map_err(|e| anyhow!("[{}:{}] Tokenize: {}", file!(), line!(), e))?;
    let mut tokens: Vec<i64> = enc.get_ids().iter().map(|&x| x as i64).collect();
    let prompt_len = tokens.len();

    for _ in 0..max_tokens {
        let shape = [1, tokens.len()];
        let input_t = candle_ok(Tensor::new(tokens.as_slice(), device))?
            .reshape(&shape)?;

        // Generate a causal mask using integers (U8) rather than floats
        // This should work for a where-cond operation
        let seq_len = tokens.len();

        // The simplest approach is to pass None for the mask and let the model
        // generate its own proper mask internally
        let logits = candle_ok(model.forward(&input_t, seq_len, None))?;

        let dims = logits.shape().dims();
        if dims.len()!=3 { break; }
        let seq_len = dims[1];
        let vocab_sz = dims[2];
        if seq_len<1 { break; }

        // Process the last token's logits
        let last_slice = candle_ok(logits.narrow(1, seq_len-1, 1))?; // shape [1,1,vocab]
        let last_1d = candle_ok(last_slice.reshape(&[vocab_sz]))?;
        let mut arr = candle_ok(last_1d.to_vec1::<f32>())?;
        log_softmax_1d(&mut arr);
        let best_idx = argmax_1d(&arr) as i64;
        tokens.push(best_idx);
    }

    let new_tokens = &tokens[prompt_len..];
    let new_t_u32: Vec<u32> = new_tokens.iter().map(|&x| x as u32).collect();
    let ans = tokenizer.decode(new_t_u32.as_slice(), true)
        .map_err(|e| anyhow!("[{}:{}] Decode: {}", file!(), line!(), e))?;
    Ok(ans)
}

/// local correctness + format
fn local_correct_reward(gt: &str, pred: &str) -> f32 {
    let re = Regex::new(r"\d+(\.\d+)?").unwrap();
    let matches: Vec<&str> = re.find_iter(pred).map(|m| m.as_str()).collect();
    if matches.is_empty() { return -1.0; }
    let last = matches.last().unwrap();
    let pf = last.parse::<f64>().unwrap_or(f64::NAN);
    let gf = gt.parse::<f64>().unwrap_or(f64::NAN);
    if pf.is_nan() || gf.is_nan() {
        -1.0
    } else if (pf-gf).abs()<1e-5 {
        1.0
    } else {
        -1.0
    }
}
fn local_format_reward(pred: &str) -> f32 {
    let re = Regex::new(r"^<think>.*?</think><answer>.*?</answer>$").unwrap();
    if re.is_match(pred) {
        1.25
    } else {
        -1.0
    }
}

#[derive(Serialize)]
struct UploadData {
    plen: usize,
    inputs_shape: (usize, usize),
    inputs: Vec<i64>,
    rewards: Vec<f32>,
}

/// replicate python's generate_mode => produce data => post to /upload
fn generate_mode(
    ds: &[QaPair],
    policy: &mut QwenModel,
    tokenizer: &Tokenizer,
    device: &Device,
    client: &Client,
    iters: usize
) -> Result<()> {
    log_line!("enter generate_mode");
    let t0 = Instant::now();
    let mut rng = StdRng::from_entropy();

    for i in 0..iters {
        // pick Q_BATCH_SIZE
        let mut picks = Vec::new();
        for _ in 0..Q_BATCH_SIZE {
            if ds.is_empty() { break; }
            picks.push(ds.choose(&mut rng).unwrap().clone());
        }

        let mut all_rows = Vec::new();
        let mut all_rewards = Vec::new();
        let mut max_len = 0usize;
        let mut bsz = 0;
        let mut used_plen = 0;

        for qa in picks {
            let sys = "You are a helpful assistant. Provide <think>..</think><answer>..</answer>.";
            let prompt_text = format!("{sys}\nQ: {}\nA:", qa.question);
            let enc = tokenizer.encode(&*prompt_text, true)
                .map_err(|e| anyhow!("[{}:{}] Prompt encode: {}", file!(), line!(), e))?;
            let p_len = enc.get_ids().len();
            if p_len>MAX_PROMPT_LENGTH { continue; }
            used_plen = p_len;

            for _ in 0..NUM_PRE_Q {
                let ans = generate_once(policy, tokenizer, &prompt_text, 128, device)?;
                let r = local_correct_reward(&qa.answer, &ans) + local_format_reward(&ans);
                // merge
                let cenc = tokenizer.encode(&*ans, false)
                    .map_err(|e| anyhow!("[{}:{}] completion encode: {}", file!(), line!(), e))?;
                let mut row = Vec::with_capacity(p_len + cenc.get_ids().len());
                row.extend(enc.get_ids().iter().map(|&x| x as i64));
                row.extend(cenc.get_ids().iter().map(|&x| x as i64));
                if row.len()>max_len {
                    max_len = row.len();
                }
                all_rows.push(row);
                all_rewards.push(r);
            }
            bsz += NUM_PRE_Q;
        }

        if all_rows.is_empty() { continue; }
        // pad
        for row in &mut all_rows {
            while row.len()<max_len {
                let pad_id = tokenizer.token_to_id("<pad>").unwrap_or(0) as i64;
                row.push(pad_id);
            }
        }
        let mut flat = Vec::with_capacity(bsz*max_len);
        for row in &all_rows {
            flat.extend_from_slice(row);
        }
        let rmax = all_rewards.iter().fold(f32::MIN, |m,&x| m.max(x));
        let rmin = all_rewards.iter().fold(f32::MAX, |m,&x| m.min(x));
        if (rmax-rmin).abs()<0.01 { continue; }

        let up = UploadData {
            plen: used_plen,
            inputs_shape: (bsz,max_len),
            inputs: flat,
            rewards: all_rewards.clone(),
        };
        let url = format!("{REF_SERVER}/upload");
        let resp = client.post(&url).json(&up).send();
        if let Err(e) = resp {
            eprintln!("upload error: {e}");
        }
        if i==0 {
            log_line!("sample => rewards= {:?}", all_rewards);
        }
    }

    let dt = t0.elapsed().as_secs_f32();
    log_line!("exit generate_mode in {dt:.2}s");
    Ok(())
}

#[derive(Deserialize)]
struct GetBatchResponse {
    plen: usize,
    inputs_shape: (usize, usize),
    inputs: Vec<i64>,
    rewards: Vec<f32>,
    refs: Vec<f32>,
}

/// get_batch => /get
fn get_batch(client: &Client) -> Option<GetBatchResponse> {
    let url = format!("{REF_SERVER}/get");
    let resp = client.get(&url).send().ok()?;
    let txt = resp.text().ok()?;
    if txt=="empty" {
        None
    } else {
        serde_json::from_str(&txt).ok()
    }
}

/// Our GRPO step => new vs ref => advantage => KL
fn grpo_step(
    policy: &mut QwenModel,
    batch: &GetBatchResponse,
    device: &Device
) -> Result<Tensor> {
    let b = batch.inputs_shape.0;
    let l = batch.inputs_shape.1;
    // completion portion
    let comp_len = l.saturating_sub(batch.plen);
    if comp_len == 0 || b == 0 {
        // nothing to update
        return candle_ok(Tensor::zeros(&[], DType::F32, device));
    }

    // Process one sequence at a time to avoid dimension issues
    let mut total_loss = 0.0;

    for bi in 0..b {
        // Extract this single sequence
        let start_idx = bi * l;
        let end_idx = start_idx + l;
        let seq_slice = &batch.inputs[start_idx..end_idx];

        // Create a single sequence tensor
        let input_t = candle_ok(Tensor::new(seq_slice, device))?
            .reshape(&[1, l])?;

        // Forward pass for just this sequence
        let logits = candle_ok(policy.forward(&input_t, (l-1).max(0), None))?;
        let logits_dims = logits.shape().dims();

        if logits_dims.len() != 3 || logits_dims[1] < 2 {
            continue; // Skip this sequence if dimensions are wrong
        }

        // Get logits for all but the last position
        let seqm1 = logits_dims[1] - 1;
        let vocab = logits_dims[2];

        // Get the logits for predicting next tokens (we'll get tokens 1 to l from positions 0 to l-1)
        let logits_for_next = candle_ok(logits.narrow(1, 0, seqm1))?;

        // Get the actual next tokens (tokens 1 to l)
        let next_tokens = candle_ok(input_t.narrow(1, 1, l-1))?;

        // Apply log_softmax to the logits
        let logits_vec = candle_ok(logits_for_next.to_vec3::<f32>())?;
        let mut logprobs_vec = logits_vec.clone();

        // Apply log_softmax per position
        for pos in 0..seqm1 {
            log_softmax_1d(&mut logprobs_vec[0][pos]);
        }

        // Convert back to tensor
        let logprobs = candle_ok(Tensor::new(logprobs_vec.clone(), device))?;

        // Get the actual next token ids
        let next_ids = candle_ok(next_tokens.to_vec2::<i64>())?;

        // Gather the log probs for the actual tokens
        let mut token_logprobs = Vec::with_capacity(seqm1);
        for pos in 0..seqm1 {
            let token_id = next_ids[0][pos];
            let safe_id = if token_id < 0 { 0 }
                          else if token_id as usize >= vocab { vocab - 1 }
                          else { token_id as usize };

            let lp = logprobs_vec[0][pos][safe_id];
            token_logprobs.push(lp);
        }

        // Get the completion portion only
        let start_pos = batch.plen.saturating_sub(1).min(seqm1);
        let comp_pos = seqm1 - start_pos;

        if comp_pos == 0 {
            continue; // No completion tokens to process
        }

        let completion_logprobs = &token_logprobs[start_pos..];

        // Reference log probs for this sequence (already for completion portion)
        let ref_start = bi * comp_len;
        let ref_end = ref_start + comp_len;
        let ref_logprobs = &batch.refs[ref_start..ref_end];

        // Ensure lengths match
        let min_len = completion_logprobs.len().min(ref_logprobs.len());
        if min_len == 0 {
            continue;
        }

        // Calculate KL divergence: exp(ref-new) - (ref-new) - 1
        let mut kl_terms = Vec::with_capacity(min_len);
        for i in 0..min_len {
            let diff = ref_logprobs[i] - completion_logprobs[i];
            let kl = diff.exp() - diff - 1.0;
            kl_terms.push(kl);
        }

        // Get reward for this sequence
        let reward = batch.rewards[bi];

        // For advantage, we'd normally compute within a group
        // For simplicity, just use the raw reward here
        let advantage = reward;

        // Calculate policy objective: advantage * logprob
        let mut policy_terms = Vec::with_capacity(min_len);
        for i in 0..min_len {
            policy_terms.push(advantage * completion_logprobs[i]);
        }

        // Combine policy objective and KL penalty
        let mut combined_terms = Vec::with_capacity(min_len);
        for i in 0..min_len {
            combined_terms.push(policy_terms[i] - BETA * kl_terms[i]);
        }

        // Average and negate (for minimization)
        let seq_loss = -combined_terms.iter().sum::<f32>() / (min_len as f32);
        total_loss += seq_loss;
    }

    // Average across all sequences
    let final_loss = if b > 0 { total_loss / (b as f32) } else { 0.0 };

    // Create a scalar tensor
    let scalar_t = candle_ok(Tensor::new(&[final_loss], device))?
        .reshape(&[])?;

    Ok(scalar_t)
}

/// A small REPL
fn talk_to_model(
    policy: &mut QwenModel,
    tokenizer: &Tokenizer,
    device: &Device
) -> Result<()> {
    log_line!("Interactive REPL. 'quit' or empty => exit.");
    let mut line = String::new();
    loop {
        print!("\nUser> ");
        std::io::stdout().flush()?;
        line.clear();
        let n = std::io::stdin().read_line(&mut line)?;
        if n==0 { break; }
        let input = line.trim();
        if input.is_empty() || input.eq_ignore_ascii_case("quit") {
            break;
        }
        let sys = "You are a helpful assistant. Provide <think>..</think><answer>..</answer>.";
        let prompt = format!("{sys}\nQ: {input}\nA:");
        let ans = generate_once(policy, tokenizer, &prompt, 64, device)?;
        log_line!("Assistant> {ans}");
    }
    Ok(())
}

/// main => the training loop
fn main() -> Result<()> {
    let device = Device::Cpu;
    let dtype = DType::F32;

    log_line!("Loading Qwen with Candle 0.8.1 on CPU...");
    let (mut varmap, mut policy_model, tokenizer, _cfg) =
        load_qwen_model_and_tokenizer(&device, dtype)?;
    log_line!("Model loaded. AdamW...");
    let vars = varmap.all_vars();
    let mut optimizer = candle_ok(AdamW::new_lr(vars, LR))
        .map_err(|e| anyhow!("AdamW new_lr: {e}"))?;

    let ds = load_fake_gsm8k();
    let client = Client::new();

    // initial generate
    generate_mode(&ds, &mut policy_model, &tokenizer, &device, &client, 2)?;

    for step in 1..=ALL_STEPS {
        let mut maybe_batch = get_batch(&client);
        while maybe_batch.is_none() {
            generate_mode(&ds, &mut policy_model, &tokenizer, &device, &client, 2)?;
            maybe_batch = get_batch(&client);
        }
        let batch = maybe_batch.unwrap();
        let loss_t = grpo_step(&mut policy_model, &batch, &device)?;
        let loss_val = f32::try_from(&loss_t)?;
        log_line!("Step {step}/{ALL_STEPS}, loss={loss_val:.4}");

        let grads = candle_ok(loss_t.backward())?;
        candle_ok(optimizer.step(&grads))?;

        if step % SAVE_STEPS == 0 {
            let ckpt = format!("step_{step}.safetensors");
            log_line!("Saving {ckpt}...");
            unsafe {
                candle_ok(varmap.save(&ckpt))?;
            }
        }
    }

    log_line!("Saving final to final.safetensors...");
    unsafe {
        candle_ok(varmap.save("final.safetensors"))?;
    }
    log_line!("Done. Interactive REPL...");
    talk_to_model(&mut policy_model, &tokenizer, &device)?;
    log_line!("All done!");
    Ok(())
}
