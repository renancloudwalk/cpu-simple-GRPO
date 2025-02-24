//! # Rust GRPO Fine-Tuning Client (Candle 0.8.1 Edition)
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
use std::fs;

use anyhow::{Result, anyhow, Context};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use rand::{rngs::StdRng, SeedableRng, seq::SliceRandom};
use regex::Regex;

// Candle crates for v0.8.1
use candle_core::{Device, DType, Tensor, Shape};
use candle_nn::{VarMap, VarBuilder, Optimizer};
use candle_nn::optim::{AdamW, AdamWBuilder};
use candle_nn::ops::{self, mean, var, sqrt, log_softmax, slice, gather, sub, exp, neg, mul, add, div};
use candle_transformers::models::qwen2::{Model as QwenModel, Config as QwenConfig};
use candle_examples::hub_load_safetensors;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

/// The Python reference server base URL
static REF_SERVER: &str = "http://localhost:59875";
/// Qwen model ID
static MODEL_ID: &str = "Qwen/Qwen2.5-0.5B";

/// GRPO hyperparams
static BETA: f32 = 0.04;         // KL penalty
static NUM_PRE_Q: usize = 8;     // completions per question
static Q_BATCH_SIZE: usize = 1;  // # Q's per generation
static MAX_PROMPT_LENGTH: usize = 400;
static ALL_STEPS: usize = 20;
static SAVE_STEPS: usize = 5;
/// AdamW LR
static LR: f64 = 1e-6;

/// Minimal Q&A struct (fake GSM8K)
#[derive(Clone)]
struct QaPair {
    question: String,
    answer: String,
}

/// We do a small, hardcoded dataset for demonstration
fn load_fake_gsm8k() -> Vec<QaPair> {
    vec![
        QaPair { question: "What is 2+2?".to_string(), answer: "4".to_string() },
        QaPair { question: "Compute 5*3".to_string(), answer: "15".to_string() },
        QaPair { question: "Compute 19-7".to_string(), answer: "12".to_string() },
    ]
}

/// Load Qwen from HF Hub with Candle 0.8.1
fn load_qwen_model_and_tokenizer(
    device: &Device,
    dtype: DType,
) -> Result<(VarMap, QwenModel, Tokenizer, QwenConfig)> {
    // HF-hub
    let api = Api::new().context("HF Hub init")?;
    let repo = api.repo(Repo::with_revision(MODEL_ID.to_string(), RepoType::Model, "main".to_string()));

    // Tokenizer
    let tokenizer_path = repo.get("tokenizer.json").context("Fetching tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(tokenizer_path.to_str().unwrap())
        .map_err(|e| anyhow!("Tokenizer load error: {e}"))?;

    // Config
    let config_path = repo.get("config.json").context("Fetching config.json")?;
    let config_bytes = fs::read(&config_path).context("Reading config file")?;
    let config: QwenConfig = serde_json::from_slice(&config_bytes)
        .map_err(|e| anyhow!("Parsing config JSON: {e}"))?;

    // Weights
    let weight_files = if let Ok(_idx) = repo.get("model.safetensors.index.json") {
        hub_load_safetensors(&repo, "model.safetensors.index.json")?
    } else {
        vec![repo.get("model.safetensors").context("Fetching model.safetensors")?]
    };

    // Build model
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, dtype, device);
    let mut model = QwenModel::new(&config, vb)
        .map_err(|e| anyhow!("Construct Qwen: {e}"))?;

    for wf in weight_files {
        unsafe {
            varmap.load(&wf).map_err(|e| anyhow!("Loading Qwen weights: {e}"))?;
        }
    }

    Ok((varmap, model, tokenizer, config))
}

/// A small argmax for a 1D f32
fn argmax_with_val(t: &Tensor) -> Result<(usize,f32)> {
    let arr = t.to_vec1::<f32>()?;
    let mut best_idx = 0;
    let mut best_val = f32::MIN;
    for (i,&val) in arr.iter().enumerate() {
        if val>best_val {
            best_val = val;
            best_idx = i;
        }
    }
    Ok((best_idx, best_val))
}

/// A single generation pass (greedy). If you want sampling, do top-p or temp sampling.
fn generate_once(
    model: &mut QwenModel,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_new_tokens: usize,
    device: &Device,
) -> Result<String> {
    // Fix the &String issue: encode(&*prompt, true) or store prompt as string
    let enc = tokenizer.encode(&*prompt, true)
        .map_err(|e| anyhow!("Tokenize error: {e}"))?;
    let mut tokens: Vec<i64> = enc.get_ids().iter().map(|&x| x as i64).collect();
    let prompt_len = tokens.len();

    for _ in 0..max_new_tokens {
        let shape = [1, tokens.len()];
        let input_t = Tensor::new(tokens.as_slice(), device)?.reshape(&shape)?;
        // forward => shape [1, seq_len, vocab]
        let logits = model.forward(&input_t, tokens.len().saturating_sub(1), None)?;
        let sh = logits.shape();
        let seq_len = sh.dims()[1];
        if seq_len==0 { break; }

        // slice out last => ops::slice along axis=1 from seq_len-1..seq_len
        let last = slice(logits.clone(), 1, (seq_len-1) as i64, seq_len as i64)?;
        // shape [1,1,vocab] => flatten => [vocab]
        let vocab_sz = last.shape().dims()[2];
        let last = last.reshape(&[vocab_sz])?;
        // log_softmax => dimension=0
        let logp = log_softmax(last, 0)?;
        let (best_idx, _val) = argmax_with_val(&logp)?;
        tokens.push(best_idx as i64);
    }

    let new_tokens = &tokens[prompt_len..];
    let new_tokens_u32: Vec<u32> = new_tokens.iter().map(|&x| x as u32).collect();
    let ans = tokenizer.decode(new_tokens_u32.as_slice(), true)
        .map_err(|e| anyhow!("Decode error: {e}"))?;
    Ok(ans)
}

/// local correctness + format rewards for demonstration
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

/// The "generate_mode" => produce data => POST to server => server adds reference log-probs
fn generate_mode(
    ds: &[QaPair],
    policy_model: &mut QwenModel,
    tokenizer: &Tokenizer,
    device: &Device,
    client: &Client,
    iterations: usize,
) -> Result<()> {
    println!("enter generate_mode");
    let start = Instant::now();
    let mut rng = StdRng::from_entropy();

    for i in 0..iterations {
        // sample Q_BATCH_SIZE
        let mut picks = Vec::new();
        for _ in 0..Q_BATCH_SIZE {
            if ds.is_empty() { break; }
            picks.push(ds.choose(&mut rng).unwrap().clone());
        }

        let mut all_rows = Vec::new();
        let mut all_rewards = Vec::new();
        let mut max_len = 0usize;
        let mut total_bsz = 0usize;
        let mut used_plen = 0usize;

        for qa in picks {
            let system = "You are a helpful assistant. Provide <think>..</think><answer>..</answer>.";
            let prompt_text = format!("{system}\nQ: {}\nA:", qa.question);
            let enc = tokenizer.encode(&*prompt_text, true)
                .map_err(|e| anyhow!("prompt encode: {e}"))?;
            let p_len = enc.get_ids().len();
            if p_len>MAX_PROMPT_LENGTH { continue; }
            used_plen = p_len;

            // generate multiple completions
            for _ in 0..NUM_PRE_Q {
                let ans = generate_once(policy_model, tokenizer, &prompt_text, 128, device)?;
                let r = local_correct_reward(&qa.answer, &ans) + local_format_reward(&ans);

                let c_enc = tokenizer.encode(&*ans, false)
                    .map_err(|e| anyhow!("completion encode: {e}"))?;
                let mut row = Vec::with_capacity(p_len + c_enc.get_ids().len());
                row.extend(enc.get_ids().iter().map(|&x| x as i64));
                row.extend(c_enc.get_ids().iter().map(|&x| x as i64));
                if row.len()>max_len { max_len = row.len(); }
                all_rows.push(row);
                all_rewards.push(r);
            }
            total_bsz += NUM_PRE_Q;
        }

        if all_rows.is_empty() { continue; }

        // pad
        for row in all_rows.iter_mut() {
            while row.len()<max_len {
                let pad_id = tokenizer.token_to_id("<pad>").unwrap_or(0) as i64;
                row.push(pad_id);
            }
        }
        let mut flat = Vec::with_capacity(total_bsz*max_len);
        for row in &all_rows {
            flat.extend_from_slice(row);
        }
        // skip if no reward variation
        let rmax = all_rewards.iter().cloned().fold(f32::MIN, f32::max);
        let rmin = all_rewards.iter().cloned().fold(f32::MAX, f32::min);
        if (rmax - rmin).abs()<0.01 { continue; }

        let up = UploadData {
            plen: used_plen,
            inputs_shape: (total_bsz, max_len),
            inputs: flat,
            rewards: all_rewards.clone(),
        };
        let url = format!("{REF_SERVER}/upload");
        let resp = client.post(&url).json(&up).send();
        if let Err(e) = resp {
            eprintln!("upload error: {e}");
        }

        if i==0 {
            println!("sample => rewards= {:?}", all_rewards);
        }
    }

    let dt = start.elapsed().as_secs_f32();
    println!("exit generate_mode in {dt:.2}s");
    Ok(())
}

/// The batch structure returned by the python server's /get
#[derive(Deserialize)]
struct GetBatchResponseDe {
    plen: usize,
    inputs_shape: (usize, usize),
    inputs: Vec<i64>,
    rewards: Vec<f32>,
    refs: Vec<f32>,
}

fn get_batch(client: &Client) -> Option<GetBatchResponseDe> {
    let url = format!("{REF_SERVER}/get");
    let resp = client.get(&url).send().ok()?;
    let txt = resp.text().ok()?;
    if txt=="empty" {
        None
    } else {
        serde_json::from_str(&txt).ok()
    }
}

/// For shape expansions
fn broadcast_2d(t: &Tensor, rows: usize, cols: usize) -> Result<Tensor> {
    let shape = t.shape();
    let dims = shape.dims();
    if dims.len()==2 && dims[0]==rows && dims[1]==cols {
        return Ok(t.clone());
    }
    if dims.len()==1 && dims[0]==rows {
        let t2 = t.unsqueeze(1)?; // [rows,1]
        let expanded = ops::expand(&t2, &[rows, cols])?;
        return Ok(expanded);
    }
    if dims.len()==1 && dims[0]==cols {
        let t2 = t.unsqueeze(0)?; // [1, cols]
        let expanded = ops::expand(&t2, &[rows, cols])?;
        return Ok(expanded);
    }
    if dims.is_empty() {
        // scalar
        let expanded = ops::expand(t, &[rows, cols])?;
        return Ok(expanded);
    }
    Err(anyhow!("Cannot broadcast shape {:?} to [{},{}]", shape, rows, cols))
}

/// The core GRPO step => comparing new vs. ref, advantage, KL
fn grpo_step(
    policy_model: &mut QwenModel,
    batch: &GetBatchResponseDe,
    device: &Device
) -> Result<Tensor> {
    let b = batch.inputs_shape.0;
    let l = batch.inputs_shape.1;
    let comp_len = l.saturating_sub(batch.plen);

    let input_t = Tensor::new(batch.inputs.as_slice(), device)?.reshape(&[b, l])?;
    let logits_all = policy_model.forward(&input_t, (l-1).max(0), None)?;
    // shape => [b, l, vocab]
    let shape = logits_all.shape();
    let seq_len = shape.dims()[1];
    // slice out => [b, l-1, vocab]
    let logits = slice(logits_all.clone(), 1, 0, (seq_len-1) as i64)?;
    // shifted input => [b, l-1]
    let shifted_input = slice(input_t, 1, 1, l as i64)?;
    // log_softmax => dimension=2
    let log_probs = log_softmax(logits, 2)?;
    // gather
    let shifted_u = shifted_input.unsqueeze(2)?; // [b, l-1, 1]
    let gathered = gather(&log_probs, &shifted_u, 2)?; // [b, l-1,1]
    let gathered = gathered.squeeze(2)?;              // [b, l-1]
    // slice => from (plen-1).. => comp_len
    let new_lp = {
        let bdim1 = gathered.shape().dims()[1];
        let start_idx = batch.plen.saturating_sub(1) as i64;
        slice(gathered, 1, start_idx, bdim1 as i64)?
    };
    // ref
    let refs_t = Tensor::new(batch.refs.as_slice(), device)?.reshape(&[b, comp_len])?;

    // kl = exp(ref-new) - (ref-new) -1
    let diff_ = sub(&refs_t, &new_lp)?;
    let exp_ = exp(diff_.clone())?;
    let part_ = sub(&exp_, &diff_)?;
    let one = Tensor::new(&[1f32], device)?.reshape(&[])?;
    let kl_ = sub(&part_, &one)?;

    // advantage
    let rewards_t = Tensor::new(batch.rewards.as_slice(), device)?.reshape(&[b])?;
    let group_count = b / NUM_PRE_Q;
    let reshaped = rewards_t.reshape(&[group_count, NUM_PRE_Q])?;
    let m = mean(reshaped.clone(), &[1])?; // shape [group_count]
    let v = var(reshaped.clone(), &[1], false)?; // shape [group_count]
    let s = sqrt(v)?;
    let mean_bc = broadcast_2d(&m, group_count, NUM_PRE_Q)?;
    let std_bc = broadcast_2d(&s, group_count, NUM_PRE_Q)?;
    let eps = Tensor::new(&[1e-4f32], device)?.reshape(&[])?;
    let std_bc = add(&std_bc, &eps)?;
    let adv_2d = div(&sub(&reshaped, &mean_bc)?, &std_bc)?;
    let adv = adv_2d.reshape(&[b])?;
    // expand => [b, comp_len]
    let adv_u = adv.unsqueeze(1)?;
    let adv_u = broadcast_2d(&adv_u, b, comp_len)?;

    // final => -( adv*new_lp - BETA*kl )
    let pol_ = mul(&adv_u, &new_lp)?;
    let bscalar = Tensor::new(&[BETA], device)?.reshape(&[])?;
    let klb_ = mul(&kl_, &bscalar)?;
    let out_ = sub(&pol_, &klb_)?;
    let neg_ = neg(out_)?;
    let seq_mean = mean(neg_.clone(), &[1])?; // shape [b]
    let loss = mean(seq_mean, &[0])?;         // shape []
    Ok(loss)
}

/// A simple REPL for inference
fn talk_to_model(
    policy_model: &mut QwenModel,
    tokenizer: &Tokenizer,
    device: &Device
) -> Result<()> {
    println!("Interactive REPL: Type empty or 'quit' to exit.");
    let mut buf = String::new();
    loop {
        print!("\nUser> ");
        std::io::stdout().flush()?;
        buf.clear();
        let n = std::io::stdin().read_line(&mut buf)?;
        if n==0 { break; }
        let line = buf.trim();
        if line.is_empty() || line.eq_ignore_ascii_case("quit") {
            break;
        }
        let system = "You are a helpful assistant. Provide <think>..</think> <answer>..</answer>.";
        let prompt = format!("{system}\nQ: {line}\nA:");
        let ans = generate_once(policy_model, tokenizer, &prompt, 64, device)?;
        println!("Assistant> {ans}");
    }
    Ok(())
}

/// main => training loop
fn main() -> Result<()> {
    let device = Device::Cpu;
    let dtype = DType::F32;

    println!("Loading Qwen from HF with Candle 0.8.1 CPU...");
    let (mut varmap, mut policy_model, tokenizer, _cfg) =
        load_qwen_model_and_tokenizer(&device, dtype)?;
    println!("Model loaded successfully.");

    // Build AdamW with AdamWBuilder
    let vars = varmap.all_vars();
    let mut optimizer = AdamWBuilder::new()
        .with_lr(LR)
        .with_betas(0.9, 0.999)
        .with_weight_decay(0.01)
        .with_eps(1e-8)
        .build(vars)
        .map_err(|e| anyhow!("AdamW builder error: {e}"))?;

    // load dataset
    let dataset = load_fake_gsm8k();
    let client = Client::new();

    // do an initial generate pass
    generate_mode(&dataset, &mut policy_model, &tokenizer, &device, &client, 3)?;

    // main loop
    for step in 1..=ALL_STEPS {
        let mut batch_opt = get_batch(&client);
        while batch_opt.is_none() {
            // generate more if empty
            generate_mode(&dataset, &mut policy_model, &tokenizer, &device, &client, 2)?;
            batch_opt = get_batch(&client);
        }
        let batch = batch_opt.unwrap();
        // grpo step
        let loss_t = grpo_step(&mut policy_model, &batch, &device)?;
        let loss_val = f32::try_from(&loss_t)?;
        println!("Step {step}/{ALL_STEPS}, loss={loss_val:.4}");
        // do backward
        let grads = loss_t.backward()?;
        optimizer.step(&grads)?;

        // checkpoint
        if step % SAVE_STEPS == 0 {
            let fname = format!("grpo_step_{step}.safetensors");
            println!("Saving checkpoint {fname}...");
            unsafe {
                varmap.save(&fname)?;
            }
        }
    }

    // final
    println!("Saving final model to final.safetensors...");
    unsafe {
        varmap.save("final.safetensors")?;
    }
    println!("Done training. Let's chat!");
    talk_to_model(&mut policy_model, &tokenizer, &device)?;
    println!("All done!");
    Ok(())
}
