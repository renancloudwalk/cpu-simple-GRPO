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
    let mut config_value: serde_json::Value = serde_json::from_slice(&config_bytes)
        .map_err(|e| anyhow!("[{}:{}] Config parse: {}", file!(), line!(), e))?;
    let hidden_size = config_value.get("hidden_size")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| anyhow!("Missing hidden_size in config"))?;
    let num_attention_heads = config_value.get("num_attention_heads")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| anyhow!("Missing num_attention_heads in config"))?;
    let expected_rotary_dim = hidden_size / num_attention_heads;
    config_value.as_object_mut().unwrap().insert("rotary_dim".to_string(), serde_json::Value::Number(serde_json::Number::from(expected_rotary_dim)));
    let config: QwenConfig = serde_json::from_value(config_value)
        .map_err(|e| anyhow!("[{}:{}] Config parse: {}", file!(), line!(), e))?;
    log_line!("Configuration loaded successfully with rotary_dim adjusted.");

    // Always use F32 for dtype
    let dtype = DType::F32;

    // Create an empty varmap; weights will be loaded into this varmap
    let mut varmap = VarMap::new();

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

    // Create the model using the fully loaded varmap
    let vb = VarBuilder::from_varmap(&varmap, dtype, device);
    let model = candle_ok(QwenModel::new(&config, vb))?;
    log_line!("Model constructed with loaded weights.");

    Ok((varmap, model, tokenizer, config))
}

/// We'll do a parallel decode across `num_completions`.
/// Candle handles the batch dimension, so each row is one completion.
fn generate_multi_completions(
    model: &mut QwenModel,
    tokenizer: &Tokenizer,
    prompt: &str,
    num_completions: usize,
    max_tokens: usize,
    device: &Device,
) -> Result<Vec<String>> {
    model.clear_kv_cache();

    // Tokenize the prompt once
    let enc = tokenizer
        .encode(prompt, /* add_special_tokens= */ true)
        .map_err(|e| anyhow!("Tokenizer encode error: {}", e))?;
    let prompt_ids: Vec<i64> = enc.get_ids().iter().map(|&x| x as i64).collect();
    let prompt_len = prompt_ids.len();

    // Create `num_completions` copies => shape [num_completions, prompt_len]
    let mut sequences = Vec::with_capacity(num_completions);
    for _ in 0..num_completions {
        sequences.push(prompt_ids.clone());
    }
    // Flatten
    let batch_input: Vec<i64> = sequences.iter().flatten().copied().collect();
    let shape = [num_completions as i64, prompt_len as i64];
    let input_t = Tensor::new(&batch_input[..], device)?.reshape(&shape)?;

    // One pass to set up the KV cache for the entire prompt
    model.forward(&input_t, 0, None)?;
    let mut offset = prompt_len;

    // Generate up to max_tokens, all completions in parallel
    for _step in 0..max_tokens {
        // Gather last token from each row => shape [num_completions, 1]
        let mut last_tokens = Vec::with_capacity(num_completions);
        for row in &sequences {
            last_tokens.push(*row.last().unwrap_or(&0));
        }

        let shape = [num_completions as i64, 1];
        let input_t = Tensor::new(&last_tokens[..], device)?.reshape(&shape)?;
        let logits = model.forward(&input_t, offset, None)?;
        offset += 1;

        // logits => [num_completions, 1, vocab_size]
        let dims = logits.shape().dims();
        if dims.len() != 3 {
            break; // unexpected
        }
        let vocab_size = dims[2];

        // Flatten to [num_completions, vocab_size]
        let last_slice = logits.narrow(1, dims[1] - 1, 1)?;
        let flatten = last_slice.reshape(&[num_completions as i64, vocab_size as i64])?;
        let arr = flatten.to_vec2::<f32>()?;

        // For each row, pick next token (greedy).
        // If you want sampling, adapt the distribution.
        for (i, mut row_data) in arr.into_iter().enumerate() {
            log_softmax_1d(&mut row_data);
            let next_id = argmax_1d(&row_data) as i64;
            sequences[i].push(next_id);
        }
    }

    // Decode only the newly generated portion
    let mut results = Vec::with_capacity(num_completions);
    for row in sequences {
        let gen_tokens = &row[prompt_len..];
        let gen_u32: Vec<u32> = gen_tokens.iter().map(|&x| x as u32).collect();
        let text = tokenizer
            .decode(&gen_u32, true)
            .map_err(|e| anyhow!("Tokenizer decode error: {}", e))?;
        results.push(text);
    }
    Ok(results)
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

/// True incremental decoding for Qwen2, using the internal K/V cache.
///
/// - `model`: your QwenModel (which includes the internal self-attn K/V).
/// - `tokenizer`: obviously your Qwen tokenizer.
/// - `prompt`: the full text prompt (all initial tokens).
/// - `max_tokens`: how many new tokens to generate (not counting the prompt).
/// - `device`: Candle device.
///
/// Returns the newly generated text, appended after the prompt.
pub fn generate_once(
    model: &mut QwenModel,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_tokens: usize,
    device: &Device,
) -> Result<String> {
    // 1) Clear old K/V.
    model.clear_kv_cache();

    // 2) Encode the prompt. We explicitly map the error to anyhow.
    let enc = tokenizer
        .encode(prompt, /* add_special_tokens = */ true)
        .map_err(anyhow::Error::msg)?;
    let mut tokens: Vec<i64> = enc.get_ids().iter().map(|&id| id as i64).collect();
    let prompt_len = tokens.len();

    // 3) If there's a prompt, feed it all at once (offset = 0).
    //    This populates the K/V cache with the entire prefix.
    if prompt_len > 0 {
        let input_t = Tensor::new(&tokens[..], device)?.reshape((1, prompt_len))?;
        model.forward(&input_t, 0, None)?;
    }

    // Keep track of how many tokens we've fed so far.
    // If we fed `prompt_len` tokens, offset = prompt_len.
    let mut offset = prompt_len;

    // 4) Generate `max_tokens` new tokens, one by one.
    for i in 0..max_tokens {
        // We'll feed just the *last* token in shape [1,1].
        let last_token_id = if tokens.is_empty() {
            0
        } else {
            *tokens.last().unwrap()
        };

        let input_t = Tensor::new(&[last_token_id][..], device)?.reshape((1, 1))?;
        let logits = model.forward(&input_t, offset, None)?;
        offset += 1;

        // 5) The shape should be [batch=1, seq=1, vocab_size].
        let shape = logits.shape().dims();
        if shape.len() != 3 {
            eprintln!("Unexpected logits shape {:?}", shape);
            break;
        }
        let vocab_sz = shape[2];
        // Flatten the last token's logits => [vocab_sz].
        let last_slice = logits.narrow(1, shape[1] - 1, 1)?;
        let flattened = last_slice.reshape((vocab_sz,))?;
        let mut arr = flattened.to_vec1::<f32>()?;

        // Argmax (or sample).
        log_softmax_1d(&mut arr);
        let next_id = argmax_1d(&arr) as i64;
        log_line!("Step {i}/{max_tokens}");
        tokens.push(next_id);
    }

    // 6) Decode only the newly generated portion, i.e. after the prompt.
    let new_tokens = &tokens[prompt_len..];
    let new_tokens_u32: Vec<u32> = new_tokens.iter().map(|&x| x as u32).collect();
    let ans = tokenizer
        .decode(&new_tokens_u32, /* skip_special_tokens=*/ true)
        .map_err(anyhow::Error::msg)?;

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
            let enc = tokenizer.encode(prompt_text.as_str(), true)
                .map_err(|e| anyhow!("[{}:{}] Prompt encode: {}", file!(), line!(), e))?;
            let p_len = enc.get_ids().len();
            if p_len>MAX_PROMPT_LENGTH { continue; }
            used_plen = p_len;

            // Now use a single multi-completions call for all answers:
            let completions = generate_multi_completions(
                policy,
                tokenizer,
                prompt_text.as_str(),
                NUM_PRE_Q,    // how many completions
                128,          // max tokens
                device
            )?;
            for ans in completions {
                let r = local_correct_reward(&qa.answer, &ans) + local_format_reward(&ans);

                // Merge: combine prompt + answer tokens
                let cenc = tokenizer.encode(ans.as_str(), false)
                    .map_err(|e| anyhow!("[{}:{}] completion encode: {}", file!(), line!(), e))?;
                let mut row = Vec::with_capacity(p_len + cenc.get_ids().len());
                row.extend(enc.get_ids().iter().map(|&x| x as i64));
                row.extend(cenc.get_ids().iter().map(|&x| x as i64));

                if row.len() > max_len {
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
        let start_idx = bi * l;
        let end_idx = start_idx + l;
        let seq_slice = &batch.inputs[start_idx..end_idx];

        let input_t = candle_ok(Tensor::new(&seq_slice[..], device))?
            .reshape(&[1, l])?;

        let logits = candle_ok(policy.forward(&input_t, (l-1).max(0), None))?;
        let logits_dims = logits.shape().dims();

        if logits_dims.len() != 3 || logits_dims[1] < 2 {
            continue; // Skip this sequence if dimensions are wrong
        }

        let seqm1 = logits_dims[1] - 1;
        let vocab = logits_dims[2];

        let logits_for_next = candle_ok(logits.narrow(1, 0, seqm1))?;
        let next_tokens = candle_ok(input_t.narrow(1, 1, l-1))?;

        let logits_vec = candle_ok(logits_for_next.to_vec3::<f32>())?;
        let mut logprobs_vec = logits_vec.clone();

        for pos in 0..seqm1 {
            log_softmax_1d(&mut logprobs_vec[0][pos]);
        }

        let logprobs = candle_ok(Tensor::new(logprobs_vec.clone(), device))?;
        let next_ids = candle_ok(next_tokens.to_vec2::<i64>())?;

        let mut token_logprobs = Vec::with_capacity(seqm1);
        for pos in 0..seqm1 {
            let token_id = next_ids[0][pos];
            let safe_id = if token_id < 0 {
                0
            } else if token_id as usize >= vocab {
                vocab - 1
            } else {
                token_id as usize
            };

            let lp = logprobs_vec[0][pos][safe_id];
            token_logprobs.push(lp);
        }

        let start_pos = batch.plen.saturating_sub(1).min(seqm1);
        let comp_pos = seqm1 - start_pos;

        if comp_pos == 0 {
            continue;
        }

        let completion_logprobs = &token_logprobs[start_pos..];

        let ref_start = bi * comp_len;
        let ref_end = ref_start + comp_len;
        let ref_logprobs = &batch.refs[ref_start..ref_end];

        let min_len = completion_logprobs.len().min(ref_logprobs.len());
        if min_len == 0 {
            continue;
        }

        let mut kl_terms = Vec::with_capacity(min_len);
        for i in 0..min_len {
            let diff = ref_logprobs[i] - completion_logprobs[i];
            let kl = diff.exp() - diff - 1.0;
            kl_terms.push(kl);
        }

        let reward = batch.rewards[bi];
        let advantage = reward;

        let mut policy_terms = Vec::with_capacity(min_len);
        for i in 0..min_len {
            policy_terms.push(advantage * completion_logprobs[i]);
        }

        let mut combined_terms = Vec::with_capacity(min_len);
        for i in 0..min_len {
            combined_terms.push(policy_terms[i] - BETA * kl_terms[i]);
        }

        let seq_loss = -combined_terms.iter().sum::<f32>() / (min_len as f32);
        total_loss += seq_loss;
    }

    let final_loss = if b > 0 {
        total_loss / (b as f32)
    } else {
        0.0
    };

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
