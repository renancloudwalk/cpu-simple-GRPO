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
fn candle_ok<T>(res: CandleResult<T>) -> Result<T> {
    match res {
        Ok(x) => Ok(x),
        Err(e) => Err(anyhow!("Candle error: {e}")),
    }
}

/// Load Qwen from HF, Candle 0.8.1
fn load_qwen_model_and_tokenizer(
    device: &Device,
    dtype: DType
) -> Result<(VarMap, QwenModel, Tokenizer, QwenConfig)> {
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(MODEL_ID.to_string(), RepoType::Model, "main".to_string()));
    println!("Loaded HF API and repository.");

    // tokenizer
    let tokenizer_path = repo.get("tokenizer.json")?;
    println!("Found tokenizer at: {:?}", tokenizer_path);
    let tokenizer = Tokenizer::from_file(tokenizer_path.to_str().unwrap())
        .map_err(|e| anyhow!("Tokenizer load: {e}"))?;
    println!("Tokenizer loaded successfully.");

    // config
    let config_path = repo.get("config.json")?;
    println!("Found config at: {:?}", config_path);
    let config_bytes = std::fs::read(&config_path)?;
    let config: QwenConfig = serde_json::from_slice(&config_bytes)
        .map_err(|e| anyhow!("Config parse: {e}"))?;
    println!("Configuration loaded successfully.");

    // weights
    let weight_files = if repo.get("model.safetensors.index.json").is_ok() {
        hub_load_safetensors(&repo, "model.safetensors.index.json")?
    } else {
        vec![repo.get("model.safetensors")?]
    };
    println!("Weight files retrieved: {:?}", weight_files);

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, dtype, device);
    let mut model = QwenModel::new(&config, vb)
        .map_err(|e| anyhow!("Building Qwen: {e}"))?;

    for wf in weight_files {
        println!("Loading weight file: {:?}", wf);
        unsafe {
            candle_ok(varmap.load(&wf))?;
        }
    }
    println!("Weights loaded.");
    // Convert all variables to F32 if they are not already.
    // This ensures that the model parameters match the expected F32 dtype.
    {
        let mut data_lock = varmap.data().lock().unwrap();
        for var in data_lock.values_mut() {
            if var.dtype() != DType::F32 {
                *var = Var::from_tensor(&candle_ok(var.to_dtype(DType::F32))?)?;
            }
        }
    }
    println!("All weights converted to F32.");
    

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
        .map_err(|e| anyhow!("Tokenize: {e}"))?;
    let mut tokens: Vec<i64> = enc.get_ids().iter().map(|&x| x as i64).collect();
    let prompt_len = tokens.len();

    for _ in 0..max_tokens {
        let shape = [1, tokens.len()];
        let input_t = candle_ok(Tensor::new(tokens.as_slice(), device))?
            .reshape(&shape)?;
        let logits = candle_ok(model.forward(&input_t, tokens.len().saturating_sub(1), None))?;
        let dims = logits.shape().dims();
        if dims.len()!=3 { break; }
        let seq_len = dims[1];
        let vocab_sz = dims[2];
        if seq_len<1 { break; }

        // narrow => last => axis=1 => from seq_len-1..1
        let last_slice = candle_ok(logits.narrow(1, seq_len-1, 1))?; // shape [1,1,vocab]
        // flatten => shape [vocab_sz]
        let last_1d = candle_ok(last_slice.reshape(&[vocab_sz]))?;
        let mut arr = candle_ok(last_1d.to_vec1::<f32>())?;
        // log_softmax_1d
        log_softmax_1d(&mut arr);
        let best_idx = argmax_1d(&arr) as i64;
        tokens.push(best_idx);
    }

    let new_tokens = &tokens[prompt_len..];
    let new_t_u32: Vec<u32> = new_tokens.iter().map(|&x| x as u32).collect();
    let ans = tokenizer.decode(new_t_u32.as_slice(), true)
        .map_err(|e| anyhow!("Decode: {e}"))?;
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
    println!("enter generate_mode");
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
                .map_err(|e| anyhow!("Prompt encode: {e}"))?;
            let p_len = enc.get_ids().len();
            if p_len>MAX_PROMPT_LENGTH { continue; }
            used_plen = p_len;

            for _ in 0..NUM_PRE_Q {
                let ans = generate_once(policy, tokenizer, &prompt_text, 128, device)?;
                let r = local_correct_reward(&qa.answer, &ans) + local_format_reward(&ans);
                // merge
                let cenc = tokenizer.encode(&*ans, false)
                    .map_err(|e| anyhow!("completion encode: {e}"))?;
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
            println!("sample => rewards= {:?}", all_rewards);
        }
    }

    let dt = t0.elapsed().as_secs_f32();
    println!("exit generate_mode in {dt:.2}s");
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
    if comp_len==0 || b==0 {
        // nothing to update
        return candle_ok(Tensor::zeros(&[], DType::F32, device));
    }

    // build input tensor => shape [b, l]
    let input_t = candle_ok(Tensor::new(batch.inputs.as_slice(), device))?
        .reshape(&[b, l])?;
    // forward => shape [b,l,vocab]
    let logits_all = candle_ok(policy.forward(&input_t, (l-1).max(0), None))?;
    let dims = logits_all.shape().dims();
    if dims.len()!=3 || dims[1]<1 {
        return candle_ok(Tensor::zeros(&[], DType::F32, device));
    }

    // narrow => [b, l-1, vocab]
    let seq_len = dims[1];
    let logits = candle_ok(logits_all.narrow(1, seq_len-1usize.saturating_sub(seq_len-1), seq_len-1))?;
    // Actually we want from 0..(seq_len-1)
    // Let's do `narrow(1, 0, seq_len-1)`
    let logits = candle_ok(logits_all.narrow(1, 0, seq_len-1))?;

    // build shifted => [b, l-1]
    if l<1 {
        return candle_ok(Tensor::zeros(&[], DType::F32, device));
    }
    let shifted = candle_ok(input_t.narrow(1, 1, l-1))?;

    // Convert to CPU array => do a 3D log_softmax
    let shape3 = logits.shape().dims();
    // shape3 = [b, l-1, vocab]
    let bsz = shape3[0];
    let seqm1 = shape3[1];
    let vocab = shape3[2];
    let raw3 = candle_ok(logits.to_vec3::<f32>())?;
    // raw3 => shape [b, l-1, vocab]
    // do in-place log_softmax along vocab dimension
    let mut new3 = raw3.clone();
    for bi in 0..bsz {
        for li in 0..seqm1 {
            let row = &mut new3[bi][li];
            log_softmax_1d(row);
        }
    }
    // rebuild
    let logprobs_t = candle_ok(Tensor::new(new3, device))?
        .reshape(&[bsz, seqm1, vocab])?;

    // gather => for each (b, pos) pick shifted token => out [b, l-1]
    let sh_data = candle_ok(shifted.to_vec2::<i64>())?;
    let logp_data = candle_ok(logprobs_t.to_vec3::<f32>())?;
    // gather manually
    let mut gather2d = Vec::with_capacity(bsz*seqm1);
    for bi in 0..bsz {
        for li in 0..seqm1 {
            let tk = sh_data[bi][li];
            let tkk = if tk<0 {0} else if tk as usize >= vocab {vocab-1} else {tk as usize};
            gather2d.push(logp_data[bi][li][tkk]);
        }
    }
    let gather_t = candle_ok(Tensor::new(gather2d.as_slice(), device))?
        .reshape(&[bsz, seqm1])?;

    // slice out the completion portion => from (plen-1) along axis=1
    let start_idx = (batch.plen.saturating_sub(1)).min(seqm1);
    let comp = candle_ok(gather_t.narrow(1, start_idx, seqm1 - start_idx))?;
    // shape => [b, comp_len], hopefully

    // reference => shape [b, comp_len]
    let refs_t = candle_ok(Tensor::new(batch.refs.as_slice(), device))?
        .reshape(&[b, comp_len])?;

    // kl = exp(ref-new) - (ref-new) -1
    let r_data = candle_ok(refs_t.to_vec2::<f32>())?;
    let c_data = candle_ok(comp.to_vec2::<f32>())?;
    // shape => [b, comp_len]
    let mut kl_data = Vec::with_capacity(b*comp_len);
    for bi in 0..b {
        for ci in 0..comp_len {
            let diff = r_data[bi][ci] - c_data[bi][ci];
            let ex = diff.exp();
            let part = ex - diff - 1.0;
            kl_data.push(part);
        }
    }
    let kl_t = candle_ok(Tensor::new(kl_data.as_slice(), device))?
        .reshape(&[b, comp_len])?;

    // advantage => group-based
    let reward_t = candle_ok(Tensor::new(batch.rewards.as_slice(), device))?
        .reshape(&[b])?;
    let group_count = b / NUM_PRE_Q;
    if group_count==0 {
        // no groups => zero
        return candle_ok(Tensor::zeros(&[], DType::F32, device));
    }
    let rew2d = reward_t.reshape(&[group_count, NUM_PRE_Q])?;
    let arr_rew = candle_ok(rew2d.to_vec2::<f32>())?;
    // compute mean, std for each group
    let mut means = Vec::with_capacity(group_count);
    let mut stds = Vec::with_capacity(group_count);
    for gc in 0..group_count {
        let row = &arr_rew[gc];
        let sum_ = row.iter().copied().sum::<f32>();
        let mean_ = sum_/(NUM_PRE_Q as f32);
        let var_ = row.iter().map(|&x| (x-mean_)*(x-mean_)).sum::<f32>()/(NUM_PRE_Q as f32);
        let s_ = var_.sqrt().max(1e-4);
        means.push(mean_);
        stds.push(s_);
    }
    // build adv
    let mut adv = Vec::with_capacity(b);
    let r1d = candle_ok(reward_t.to_vec1::<f32>())?;
    for gc in 0..group_count {
        let m_ = means[gc];
        let s_ = stds[gc];
        for i in 0..NUM_PRE_Q {
            let val = r1d[gc*NUM_PRE_Q + i];
            adv.push((val - m_)/s_);
        }
    }
    let adv_t = candle_ok(Tensor::new(adv.as_slice(), device))?
        .reshape(&[b])?;
    // expand => [b, comp_len]
    let mut adv2d = Vec::with_capacity(b*comp_len);
    let advf = candle_ok(adv_t.to_vec1::<f32>())?;
    for bi in 0..b {
        let a_ = advf[bi];
        for _ in 0..comp_len {
            adv2d.push(a_);
        }
    }
    let adv2d_t = candle_ok(Tensor::new(adv2d.as_slice(), device))?
        .reshape(&[b, comp_len])?;

    // pol = adv*comp
    let cd = candle_ok(comp.to_vec2::<f32>())?;
    let ad = candle_ok(adv2d_t.to_vec2::<f32>())?;
    let mut pol = Vec::with_capacity(b*comp_len);
    for bi in 0..b {
        for ci in 0..comp_len {
            pol.push(ad[bi][ci]*cd[bi][ci]);
        }
    }
    let pol_t = candle_ok(Tensor::new(pol.as_slice(), device))?
        .reshape(&[b, comp_len])?;

    // kl * beta => pol_t - kl
    let klv = candle_ok(kl_t.to_vec2::<f32>())?;
    let mut out_ = Vec::with_capacity(b*comp_len);
    for bi in 0..b {
        for ci in 0..comp_len {
            out_.push(pol[bi*comp_len + ci] - (klv[bi][ci]*BETA));
        }
    }
    // negative => final
    for val in out_.iter_mut() {
        *val = -(*val);
    }
    // average over comp_len => shape [b], then average over b
    let mut row_means = Vec::with_capacity(b);
    for bi in 0..b {
        let start = bi*comp_len;
        let end = start+comp_len;
        let slice_ = &out_[start..end];
        let mean_ = slice_.iter().sum::<f32>()/(comp_len as f32);
        row_means.push(mean_);
    }
    let final_loss = row_means.iter().sum::<f32>()/(b as f32);
    // build a scalar
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
    println!("Interactive REPL. 'quit' or empty => exit.");
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
        println!("Assistant> {ans}");
    }
    Ok(())
}

/// main => the training loop
fn main() -> Result<()> {
    let device = Device::Cpu;
    let dtype = DType::F32;

    println!("Loading Qwen with Candle 0.8.1 on CPU...");
    let (mut varmap, mut policy_model, tokenizer, _cfg) =
        load_qwen_model_and_tokenizer(&device, dtype)?;
    println!("Model loaded. AdamW...");
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
        println!("Step {step}/{ALL_STEPS}, loss={loss_val:.4}");

        let grads = candle_ok(loss_t.backward())?;
        candle_ok(optimizer.step(&grads))?;

        if step % SAVE_STEPS == 0 {
            let ckpt = format!("step_{step}.safetensors");
            println!("Saving {ckpt}...");
            unsafe {
                candle_ok(varmap.save(&ckpt))?;
            }
        }
    }

    println!("Saving final to final.safetensors...");
    unsafe {
        candle_ok(varmap.save("final.safetensors"))?;
    }
    println!("Done. Interactive REPL...");
    talk_to_model(&mut policy_model, &tokenizer, &device)?;
    println!("All done!");
    Ok(())
}
