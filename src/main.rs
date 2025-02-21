//! A demonstration of GRPO fine-tuning on a Qwen-like Candle model with commentary.

// ====================================
// 1) Crate imports
// ====================================
use std::io::Write;

use candle_core::{Device, DType, Error as CandleError, IndexOp, Result, Tensor};
use candle_nn::{Optimizer, VarMap, VarBuilder, AdamW, ParamsAdamW};
use candle_transformers::models::qwen2;
use hf_hub::{api::sync::Api, Repo, RepoType};
use reqwest::blocking::Client;
use serde::Deserialize;
use tokenizers::Tokenizer;

// If your Candle version does not have this helper, comment it out or replace
use candle_examples::hub_load_safetensors;

// ====================================
// 2) Helper data structures
// ====================================

/// The structure of the JSON response from the reference server.
#[derive(Deserialize)]
struct EvalResponse {
    reward: f32,
    log_prob_ref: f32,
}

// ====================================
// 3) Main entry point
// ====================================

fn main() -> Result<()> {
    let device = Device::Cpu;
    println!("Using device: {:?}", device);

    // 3.1) Setup HF Hub
    let model_id = "Qwen/Qwen2.5-0.5B";
    let api = Api::new().map_err(|e| CandleError::wrap(format!("HF Hub Api error: {e}")))?;
    let repo = api.repo(
        Repo::with_revision(model_id.to_string(), RepoType::Model, "main".to_string())
    );

    // 3.2) Try to fetch tokenizer, config, and weights
    let tokenizer_path = repo
        .get("tokenizer.json")
        .map_err(|e| CandleError::wrap(format!("Failed to get tokenizer: {e}")))?;
    let config_path = repo
        .get("config.json")
        .map_err(|e| CandleError::wrap(format!("Failed to get config: {e}")))?;

    // 3.3) Check if model is sharded or not
    let weight_files = if let Ok(_index) = repo.get("model.safetensors.index.json") {
        // If sharded:
        hub_load_safetensors(&repo, "model.safetensors.index.json")?
    } else {
        // If single-file:
        vec![repo
            .get("model.safetensors")
            .map_err(|e| CandleError::wrap(format!("Failed to get model weights: {e}")))?
        ]
    };

    // 3.4) Construct VarMap and parse config
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let config_bytes = std::fs::read(&config_path)
        .map_err(|e| CandleError::wrap(format!("Reading config file: {e}")))?;
    let config: qwen2::Config = serde_json::from_slice(&config_bytes)
        .map_err(|e| CandleError::wrap(format!("Config parse error: {e}")))?;

    // 3.5) Build the Qwen model and load weights
    let mut model = qwen2::Model::new(&config, vb)?;
    for wf in weight_files {
        varmap.load(&wf)?;
    }
    println!("Model weights loaded successfully!");

    // 3.6) Load tokenizer
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| CandleError::wrap(format!("Tokenizer load error: {e}")))?;
    println!("Tokenizer loaded successfully!");

    // 3.7) Setup AdamW optimizer. We define the hyperparams via `ParamsAdamW`:
    let params = ParamsAdamW {
        lr: 1e-6,          // base learning rate
        weight_decay: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
    };
    let mut optimizer = AdamW::new(varmap.all_vars(), params)?;

    // 3.8) Example fine-tuning loop
    let training_prompts = vec![
        "Question: 2+2=? Answer:",
        "Question: In 1492, who reached the Americas? Answer:",
    ];
    let group_size = 2; // 2 completions per prompt
    let kl_beta = 0.04;
    let max_steps = 3;
    let client = Client::new();

    for step in 1..=max_steps {
        println!("\n=== Step {step} ===");
        let mut total_loss = Tensor::zeros((), DType::F32, &device)?;

        for prompt in &training_prompts {
            let mut completions = Vec::new();
            let mut rewards = Vec::new();
            let mut logprobs_cur = Vec::new();
            let mut logprobs_ref = Vec::new();

            // Generate multiple completions per prompt
            for _ in 0..group_size {
                let (completion_text, neg_logprob) =
                    generate_once(&mut model, &tokenizer, prompt, &device)?;

                // Query reference server
                let eval = query_reference_server(&client, prompt, &completion_text)?;
                completions.push(completion_text);
                rewards.push(eval.reward);
                logprobs_ref.push(eval.log_prob_ref);
                logprobs_cur.push(neg_logprob); // sum of -logP
            }

            // Compute group advantage
            let mean_reward = rewards.iter().sum::<f32>() / (rewards.len() as f32);
            let advantages: Vec<f32> = rewards.iter().map(|r| r - mean_reward).collect();

            // Build GRPO loss for each completion
            for i in 0..group_size {
                // advantage * -logP
                let adv_tensor = scalar(advantages[i], &device)?;
                let nll_tensor = scalar(logprobs_cur[i], &device)?; // sum of -logP
                let ref_tensor = scalar(logprobs_ref[i], &device)?; // reference logprob

                // logprob_cur = -nll
                // => kl_term = kl_beta * ( logprob_cur - logprob_ref )
                //           = kl_beta * ( -nll - ref )
                let kl_term = scalar(-1.0, &device)?
                    .mul(&nll_tensor)?
                    .sub(&ref_tensor)?
                    .mul(&scalar(kl_beta, &device)?)?;

                let policy_loss = adv_tensor.mul(&nll_tensor)?; // advantage * nll
                let loss_i = policy_loss.add(&kl_term)?;
                total_loss = total_loss.add(&loss_i)?;
            }
        }

        // Average over total completions
        let denom = (training_prompts.len() * group_size) as f32;
        total_loss = total_loss.div(&scalar(denom, &device)?)?;

        println!("Step {step} Loss = {:.4}", total_loss.to_scalar::<f32>()?);
        optimizer.backward_step(&total_loss)?; // from the `Optimizer` trait
    }

    // 3.9) Save fine-tuned model
    let save_path = "policy_model_finetuned.safetensors";
    varmap.save(save_path)?;
    println!("Model saved to {save_path}");

    // Optional: test a quick interactive session
    talk_to_model(model_id, save_path)?;

    Ok(())
}

// ====================================
// 4) Helper functions
// ====================================

/// Create a 0D scalar tensor from a Rust float.
fn scalar(value: f32, device: &Device) -> Result<Tensor> {
    Tensor::new(&[value], device)?.reshape(())
}

/// Generate a single completion, returning (completion_text, sum_of_negative_logprobs).
///
/// *Important:* we convert `Vec<u32>` to `Vec<i64>`, because Candle Qwen typically uses i64 token IDs.
fn generate_once(
    model: &mut qwen2::Model,
    tokenizer: &Tokenizer,
    prompt: &str,
    device: &Device,
) -> Result<(String, f32)> {
    // 1) Encode the prompt
    let enc = tokenizer
        .encode(prompt, true)
        .map_err(|e| CandleError::wrap(format!("Tokenizer error: {e}")))?;
    let tokens_u32: Vec<u32> = enc.get_ids().to_vec();

    // Convert to i64
    let mut tokens: Vec<i64> = tokens_u32.iter().map(|&x| x as i64).collect();
    let prompt_len = tokens.len();

    // 2) Sample up to max_new_tokens
    let max_new_tokens = 64;
    let mut neg_logprob_total = 0f32;

    for _step in 0..max_new_tokens {
        // Candle expects [batch_size, seq_len], so unsqueeze(0).
        // Use `tokens.as_slice()`, not `&tokens`.
        let input = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;

        // Qwen forward requires a third argument: `Option<&Tensor>`; pass `None` if not doing KV-caching.
        let logits = model.forward(&input, tokens.len().saturating_sub(1), None)?;

        // shape: [1, seq_len, vocab_size]
        // Next token's logits are in the last position
        let last_pos = tokens.len() - 1;
        let next_logits = logits.i((0, last_pos))?; // shape: [vocab_size]

        // log_softmax along dimension=0 (since shape is [vocab_size])
        let next_log_probs = candle_nn::ops::log_softmax(&next_logits, 0)?;
        let (next_token_idx, logprob_val) = argmax_with_val(&next_log_probs)?;

        neg_logprob_total += -logprob_val;
        tokens.push(next_token_idx as i64);

        // Optional: break on special token
        if let Some(eot_id) = tokenizer.token_to_id("<|endoftext|>") {
            if next_token_idx == eot_id {
                break;
            }
        }
    }

    // 3) Decode the newly generated portion
    let answer_tokens = &tokens[prompt_len..];
    // We must convert back to u32 for `tokenizer.decode(&[u32], ...)`
    let answer_u32: Vec<u32> = answer_tokens.iter().map(|&x| x as u32).collect();
    let answer_text = tokenizer
        .decode(&answer_u32, true)
        .map_err(|e| CandleError::wrap(format!("Decode error: {e}")))?;

    Ok((answer_text, neg_logprob_total))
}

/// A simple argmax over a 1D log_probs distribution, returning (best_index, best_logprob).
fn argmax_with_val(log_probs: &Tensor) -> Result<(u32, f32)> {
    let values = log_probs.to_vec1::<f32>()?;
    let mut best_idx = 0usize;
    let mut best_val = f32::MIN;
    for (i, &val) in values.iter().enumerate() {
        if val > best_val {
            best_val = val;
            best_idx = i;
        }
    }
    Ok((best_idx as u32, best_val))
}

/// Query a reference server that returns a JSON {reward, log_prob_ref}.
/// We wrap `reqwest::Error` into `std::io::Error` so we can use `?`.
fn query_reference_server(
    client: &Client,
    prompt: &str,
    response: &str
) -> std::io::Result<EvalResponse>
{
    let url = "http://localhost:5000/eval";
    let body = serde_json::json!({
        "prompt": prompt,
        "completion": response
    });

    let resp = client
        .post(url)
        .json(&body)
        .send()
        .map_err(|e| map_reqwest_err(e))?
        .error_for_status()
        .map_err(|e| map_reqwest_err(e))?
        .json::<EvalResponse>()
        .map_err(|e| map_reqwest_err(e))?;

    Ok(resp)
}

/// Convert a reqwest::Error into a std::io::Error, for `?` usage in a std::io::Result.
fn map_reqwest_err(e: reqwest::Error) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::Other, e)
}

/// Demonstrates loading your newly saved model for inference.
fn talk_to_model(model_id: &str, safetensor_path: &str) -> Result<()> {
    println!("Reloading model from {safetensor_path} for interactive inference.\n");
    let device = Device::Cpu;

    // Re-initialize VarMap, config, model, etc.
    let api = Api::new().map_err(|e| CandleError::wrap(format!("HF Hub Api error: {e}")))?;
    let repo = api.repo(
        Repo::with_revision(model_id.to_string(), RepoType::Model, "main".to_string())
    );

    let config_path = repo
        .get("config.json")
        .map_err(|e| CandleError::wrap(format!("Failed to get config: {e}")))?;
    let config_bytes = std::fs::read(&config_path)
        .map_err(|e| CandleError::wrap(format!("Reading config file: {e}")))?;
    let config: qwen2::Config = serde_json::from_slice(&config_bytes)
        .map_err(|e| CandleError::wrap(format!("Config parse error: {e}")))?;

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let mut model = qwen2::Model::new(&config, vb)?;
    varmap.load(safetensor_path)?;

    println!("Model loaded. Let's talk!");

    // We'll also load the tokenizer again
    let tokenizer_path = repo
        .get("tokenizer.json")
        .map_err(|e| CandleError::wrap(format!("Failed to get tokenizer: {e}")))?;
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| CandleError::wrap(format!("Tokenizer load error: {e}")))?;

    // Simple REPL
    let mut user_input = String::new();
    loop {
        print!("\nUser> ");
        std::io::stdout().flush().unwrap();
        user_input.clear();
        if std::io::stdin().read_line(&mut user_input).is_err() {
            break;
        }
        let user_input = user_input.trim();
        if user_input.eq_ignore_ascii_case("quit") {
            break;
        }

        let (reply, _neg_logp) = generate_once(&mut model, &tokenizer, user_input, &device)?;
        println!("Model> {}", reply);
    }
    Ok(())
}
