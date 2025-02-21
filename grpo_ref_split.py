"""
Overview:
---------
This script fine-tunes a causal language model using a reinforcement learning technique
called GRPO (Group Relative Policy Optimization). It interacts with a separate reference
server to obtain log-probabilities from a stable reference model, which are used to
penalize deviations from that reference (KL-constraint). Below is an outline of the key steps:

1) **Data Generation** (generate_mode):
   - The script samples questions from a math dataset (GSM8K), then generates multiple
     answers per question with the current model parameters.
   - Each answer is scored by a custom reward function, combining correctness (did we
     get the right numerical answer?) and formatting (did we follow the <think>...</think>
     <answer>...</answer> structure?).
   - These prompt+answer sequences, plus their rewards, are sent to the reference server
     to compute reference log-probs for each token in the answer.

2) **Data Retrieval** (get_batch):
   - The script requests processed data (prompts, answers, rewards, plus reference
     log-probs) from the reference server once they are ready.

3) **Policy Optimization** (GRPO_step):
   - For each batch, the script computes:
       a) The current model’s log-probs for the answer portion,
       b) The KL divergence between current and reference log-probs,
       c) A group-based advantage, where each question has multiple answers.
          (We normalize each answer’s reward relative to the group’s mean & std.)
       d) The GRPO loss, combining advantage-weighted log-probs minus a beta-weighted KL penalty.

4) **Training Loop**:
   - Repeatedly fetch a batch from the server, compute the GRPO loss, backprop, and step
     the optimizer. Occasionally generate more data (if none is available) and save checkpoints.

In sum, this file orchestrates an RL training process where a language model is optimized
to produce correct, well-formatted answers while staying close to a reference model’s
distribution, using GRPO’s group-based advantage estimation and KL regularization.
"""
###############################################################################
# Main Script for Fine-Tuning a Causal LM with GRPO (Group Relative Policy Optimization)
# Using a Reference Server to Provide Reference Log-Probabilities
###############################################################################

from transformers import AutoTokenizer, AutoModelForCausalLM
import json, os, shutil, re, random, requests, io, sys, time
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist

# The environment variable below tells tokenizers to limit CPU parallelism,
# to avoid overhead in multi-threaded tokenization.
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

###############################################################################
# Distributed Setup
###############################################################################

# If PyTorch's distributed module is available & initialized, we get the rank;
# otherwise, default to rank=0 (a single-process scenario).
if torch.distributed.is_available() and torch.distributed.is_initialized():
    rank = torch.distributed.get_rank()
else:
    rank = 0

def barrier():
    """
    A convenience function to perform a distributed barrier (all processes wait)
    if torch.distributed is available & initialized. Otherwise, it's a no-op.
    This ensures synchronization across multiple processes in a distributed run.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        barrier()

###############################################################################
# Hyperparameters & Config
###############################################################################

# The path or name of the pretrained model to load;
# Qwen2.5-0.5B is presumably a smaller Qwen variant from a specific source.
model_path = "Qwen/Qwen2.5-0.5B"

# KL-divergence penalty coefficient in GRPO. A higher 'beta' forces the new policy
# to stay closer to the reference policy; a smaller 'beta' allows more deviation.
beta = 0.04

# Number of completions (answers) to generate per question. Used in GRPO to
# compute a group-based advantage (the relative difference from the group's mean).
num_pre_Q = 8

# How many questions to sample each time we do a generation pass.
# Typically small here because we generate multiple completions per question.
Q_batch_size = 1

# Total steps of training. Each step pulls a batch from the ref server and does one update.
all_steps = 100

# Skip prompts longer than this token count to avoid extreme memory usage.
max_prompt_length = 400

# Save model checkpoint every 'save_steps' steps.
save_steps = 10

###############################################################################
# Reference Server
###############################################################################

# The reference server URL. This server is expected to handle reference log-probs,
# meaning it runs or has access to a stable reference model for KL constraints.
ref_server = "http://localhost:59875"

# Utility functions from 'ref_server' for (de)serializing data between processes.
from ref_server import tensor_to_bytes, bytes_to_tensor, make_bytes_list, bytes_list_to_list

###############################################################################
# get_batch: Fetching Data from Reference Server
###############################################################################

def get_batch():
    """
    Attempt to retrieve a batch of data from the reference server. This batch contains:
     - JSON metadata (like prompt length 'plen'),
     - Serialized 'inputs' tensor (prompt+generated completion token IDs),
     - Serialized 'rewards' tensor,
     - Serialized 'refs' tensor (reference log-probabilities for tokens).

    Returns:
        data (dict or None): If no data is available, returns None. Otherwise:
          data['plen']   = (int) prompt length in tokens,
          data['inputs'] = (Tensor) shape [batch_size, seq_len],
          data['rewards'] = (Tensor) shape [batch_size,],
          data['refs']   = (Tensor) shape [batch_size, seq_len_of_completion].
    """
    try:
        # Request a batch from the reference server endpoint.
        r = requests.get(f"{ref_server}/get").content
        if r == b'empty':
            # 'empty' means no data is queued, so we return None to indicate that.
            return None
    except:
        # If server is unavailable or request fails, return None and we can retry.
        return None

    # Convert the raw bytes into a list of individual byte segments.
    dd = bytes_list_to_list(r)
    # The first segment is JSON-encoded metadata (contains 'plen').
    data = json.loads(dd[0])
    # Next segments are serialized tensors: inputs, rewards, reference log-probs.
    data['inputs'] = bytes_to_tensor(dd[1])
    data['rewards'] = bytes_to_tensor(dd[2])
    data['refs'] = bytes_to_tensor(dd[3])
    return data

###############################################################################
# Model & Tokenizer Setup
###############################################################################

# We load an AutoTokenizer and a causal language model (AutoModelForCausalLM)
# from the specified model_path. Using float32 for safety (less likely to overflow).
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    _attn_implementation="sdpa"
)

# We'll also reference 'gen_model' for generation.
# Right now it's the same object, but it’s named distinctly in case of extension.
gen_model = model

###############################################################################
# Dataset (GSM8K) Loading
###############################################################################

from datasets import load_dataset
dataset = load_dataset("openai/gsm8k", "main", split="train")

# GSM8K answers often have an explanation followed by '#### <final answer>'.
# We'll store question 'Q' and the final numeric solution 'A' after removing '####'.
QAs = [{'Q': x, 'A': y.split('####')[-1].strip()}
       for x, y in zip(dataset['question'], dataset['answer'])]

###############################################################################
# GenerationConfig
###############################################################################

from transformers import GenerationConfig
generation_config = GenerationConfig(
    max_new_tokens=512,            # Limit the max length of the generated answer
    do_sample=True,                # Enable sampling (rather than greedy)
    temperature=0.9,               # Temperature for sampling randomness
    num_return_sequences=num_pre_Q, # Generate this many completions per question
    pad_token_id=tokenizer.pad_token_id
)

###############################################################################
# System Prompt
###############################################################################

# A guiding prompt for the model, telling it to enclose reasoning in <think> tags
# and the final answer in <answer> tags, to keep them clearly separated.
system_prompt = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""

###############################################################################
# gen_answers: Generate Multiple Answers per Prompt
###############################################################################

def gen_answers(prompts):
    """
    Given a list of user prompts, apply the system prompt template, tokenize them,
    and generate multiple answers per prompt using the current 'gen_model'.

    We skip generation if the prompt length exceeds 'max_prompt_length' to avoid
    potential high memory usage.

    Returns:
        List of generated answer strings (length = #prompts * num_pre_Q).
        If the prompt is too long, returns [].
    """
    tip_text = []
    # Build the final prompt text for each user query, combining system & user roles.
    for x in prompts:
        tip_text.append(tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": x}
            ],
            tokenize=False,
            add_generation_prompt=True
        ))

    # Tokenize them as a batch. We use left padding so the newly generated tokens
    # come after the entire prompt.
    tip_inputs = tokenizer(
        tip_text,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=False
    )
    prompt_length = tip_inputs["input_ids"].shape[-1]

    # If the prompt is too long, skip generating (returns empty).
    if prompt_length > max_prompt_length:
        return []

    # Move input to model device for generation.
    tip_inputs = {k: v.to(model.device) for k, v in tip_inputs.items()}

    # Inference mode: generating tokens doesn't need gradient tracking.
    with torch.inference_mode():
        tip_completion_ids = gen_model.generate(**tip_inputs, generation_config=generation_config)

    # The first 'prompt_length' tokens in each generated sequence are just the prompt.
    # We only keep the newly generated completion part.
    completion_ids = tip_completion_ids[:, prompt_length:]

    # Decode them into strings, removing any <|endoftext|> placeholders.
    answers = [tokenizer.decode(x).replace('<|endoftext|>', '') for x in completion_ids]
    return answers

###############################################################################
# Reward Calculation Helpers
###############################################################################

def parse(x, extraction_config=None):
    """
    Attempts to parse a string as a float. If it fails, returns the string itself.
    Used to interpret numeric answers for correctness verification.
    """
    try:
        return float(x)
    except Exception:
        return x

def verify(a, b):
    """
    If both are float-like, compare numerically, else compare as exact strings.
    Returns True if they match, otherwise False.
    """
    try:
        return float(a) == float(b)
    except Exception:
        return a == b

class ExprExtractionConfig:
    """
    Placeholder class for the parse() function's signature.
    """
    pass

def reward_correct(item, answer):
    """
    Check if the final numeric output in 'answer' matches the ground truth 'item["A"]'.
     - We find all numeric strings (including decimals, fractions).
     - If none are found, we assign -1 (incorrect).
     - Otherwise, we parse the last one found and compare to the ground truth
       (which is also parsed). +1 if correct, -1 if not.

    This encourages the model to produce the correct numeric result in the final position.
    """
    pattern = r'\d+\.\d+|\d+/\d+|\d+'
    nums = re.findall(pattern, answer)
    if len(nums) == 0:
        return -1.0
    lastnum = nums[-1]
    ans = parse(lastnum, extraction_config=[ExprExtractionConfig()])
    ground_truth = parse(item["A"], extraction_config=[ExprExtractionConfig()])
    return 1 if verify(ans, ground_truth) else -1

def reward_format(item, answer):
    """
    Check whether the answer follows <think>...</think><answer>...</answer> exactly
    (no extra text outside). If yes, we give +1.25, else -1.
    This encourages a consistent chain-of-thought structure.
    """
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    return 1.25 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) else -1

###############################################################################
# gen_samples: Generate & Tokenize Answers, Compute Rewards
###############################################################################

def gen_samples(inputs):
    """
    For a list of question-answer dicts (keys "Q" and "A"), do:
      1) Generate 'num_pre_Q' completions per question.
      2) Compute the reward for each completion (correctness + formatting).
      3) Tokenize the prompts and the answers to produce:
         - prompt_inputs["input_ids"]
         - output_ids["input_ids"]
         - rewards (a FloatTensor)
         - the raw answer strings

    If generation fails (e.g. prompt too long), returns (None, None, None, None).
    """
    prompts = [x["Q"] for x in inputs]  # just the questions from the batch
    answers = gen_answers(prompts)
    if len(answers) == 0:
        return None, None, None, None

    # Evaluate each answer with the reward functions
    rewards = []
    for i, inp in enumerate(inputs):
        # For each question, gather its chunk of answers
        for a in answers[i*num_pre_Q : (i+1)*num_pre_Q]:
            # The total reward is correctness + format
            rewards.append(reward_correct(inp, a) + reward_format(inp, a))

    # Re-tokenize the original prompts with the system prompt
    prompts_text = [
        tokenizer.apply_chat_template([
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": x}
        ], tokenize=False, add_generation_prompt=True)
        for x in prompts
    ]
    prompt_inputs = tokenizer(
        prompts_text,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=False
    )

    # Tokenize the answers
    output_ids = tokenizer(
        answers,
        return_tensors="pt",
        padding=True,
        padding_side="right",
        add_special_tokens=False
    )

    return (
        prompt_inputs["input_ids"],
        output_ids["input_ids"],
        torch.tensor(rewards, dtype=torch.float32),
        answers
    )

###############################################################################
# generate_mode: Generate Data & Send to Reference Server
###############################################################################

def generate_mode(num=10, rank=0):
    """
    This function repeatedly samples from QAs, uses gen_samples() to generate
    completions and compute rewards, then merges the prompt & answer tokens
    and sends them to 'ref_server' for reference log-prob calculation.

    Parameters:
        num (int): number of generation iterations to run.
        rank (int): process rank, used to limit logging to rank=0 typically.
    """
    if rank == 0:
        print('enter generate mode')
    tic = time.time()

    for ii in range(num):
        # Randomly pick Q_batch_size questions from the QAs
        inputs = random.sample(QAs, Q_batch_size)
        prompt_inputs, output_ids, rewards, answers = gen_samples(inputs)
        if prompt_inputs is None:
            continue  # Generation might have been skipped if prompt too long

        if rank == 0:
            print('rewards:', rewards)
            if ii == 5:
                # As an example, print the first answer
                print('answers:', answers[0])

        # If there's no difference in reward, it won't help training
        # (no gradient signal), so skip.
        if (rewards.max() - rewards.min()).item() < 0.01:
            continue

        # Each question is repeated 'rep' times = num_pre_Q answers.
        # We replicate the prompt accordingly and concat the outputs.
        rep = output_ids.shape[0] // prompt_inputs.shape[0]
        prompt_length = prompt_inputs.shape[1]
        Qrep = prompt_inputs.repeat(1, rep).view(-1, prompt_length)
        merged_ids = torch.cat([Qrep, output_ids], dim=1)

        # Build a data package with:
        # - JSON containing prompt length,
        # - merged token IDs,
        # - rewards.
        xdata = make_bytes_list([
            json.dumps({"plen": prompt_length}).encode(),
            tensor_to_bytes(merged_ids),
            tensor_to_bytes(rewards)
        ])
        # Post to the reference server, which will add reference log-probs
        # and store the batch for us to retrieve with 'get_batch'.
        requests.post(f"{ref_server}/upload", data=xdata)

    if rank == 0:
        print('exit generate mode')
    print(f'{rank}: {time.time()-tic:.3f}s')

###############################################################################
# If 'genonly' is provided in CLI, just generate infinitely, no training
###############################################################################

if 'genonly' in sys.argv:
    # Move the model to CPU (optional for memory reasons) and do generation
    # basically forever (or until manual interruption).
    model.to('cpu')
    generate_mode(999999)
    sys.exit()

###############################################################################
# Optimizer Setup
###############################################################################

# Initialize an AdamW optimizer for the model parameters.
# AdamW is a variant of Adam that decouples weight decay from the Adam step,
# helping maintain correct L2 regularization.
# We use a low learning rate (1e-6) here because we are fine-tuning
# a pretrained model on a reinforcement learning signal,
# and large updates could destabilize or overwrite pretrained knowledge.
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

# For now, 'engine' is just the model; no advanced distributed wrapper here.
engine = model

# Also keep 'gen_model' pointing to the same model object, so generation uses updated weights.
gen_model = model

###############################################################################
# get_per_token_logps: Compute Log-Prob of Each Token in a Sequence
###############################################################################

def get_per_token_logps(logits, input_ids):
    """
    For a batch of logits (B, L, V), get the log-prob of each 'gold' token in input_ids (B, L).
    We do log_softmax over the vocab dimension and gather each token's log-prob.
    Using a loop can reduce memory usage if the sequence is long,
    rather than doing big matrix ops at once.

    Returns:
        (B, L) tensor of log-probs for each token.
    """
    per_token_logps = []
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(
            log_probs, dim=1, index=input_ids_row.unsqueeze(1)
        ).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)

###############################################################################
# GRPO_step: Core Training Step
###############################################################################

def GRPO_step(batch):
    """
    Compute the GRPO (Group Relative Policy Optimization) loss on one batch:
     1) Split out prompt vs. answer portion with 'prompt_length'.
     2) Get model's log-probs for answer tokens, and reference log-probs from 'refs'.
     3) Compute per-token KL penalty = exp(ref-new) - (ref-new) - 1.
     4) Group rewards by question (num_pre_Q answers each), compute mean & std
        to get the advantage for each answer (reward - mean) / std.
     5) Form the loss:
         per_token_loss = -( advantage * log(prob_of_token) - beta * KL ).
     6) Mask out padded tokens in the answer portion, average over valid tokens,
        then average over the batch.

    Returns:
        A scalar 'loss' that is used for backprop.
    """
    prompt_length = batch['plen']
    inputs = batch['inputs'].to(engine.device)
    rewards = batch['rewards'].to(engine.device)

    # Forward pass: get model logits of shape (B, L, V).
    logits = engine(inputs).logits
    # Drop the final logit (predicts beyond last token). Now (B, L-1, V).
    logits = logits[:, :-1, :]

    # Align input_ids with these logits. Now shape (B, L-1).
    input_ids = inputs[:, 1:]

    # Compute log-prob of the actual tokens.
    per_token_logps = get_per_token_logps(logits, input_ids)

    # We only want to handle the 'answer' portion, skipping the prompt.
    # The first answer token's log-prob is at index (prompt_length - 1).
    per_token_logps = per_token_logps[:, prompt_length-1:]

    # 'refs' holds reference log-probs for the same answer tokens.
    ref_per_token_logps = batch['refs'].to(per_token_logps.device)

    # Per-token KL:
    # if 'new_logp' = per_token_logps, 'ref_logp' = ref_per_token_logps,
    # the formula is: exp(ref - new) - (ref - new) - 1
    # This stays at 0 if new=ref and grows as they diverge.
    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) \
                   - (ref_per_token_logps - per_token_logps) - 1

    # Build a mask to exclude pads in the completion portion.
    completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()

    # Group-based advantage: shape (B,) -> reshape to (N, num_pre_Q),
    # then for each question, find mean & std to center the rewards.
    mean_grouped_rewards = rewards.view(-1, num_pre_Q).mean(dim=1)
    std_grouped_rewards  = rewards.view(-1, num_pre_Q).std(dim=1)

    # Broadcast them back to shape (B,):
    mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_pre_Q, dim=0)
    std_grouped_rewards  = std_grouped_rewards.repeat_interleave(num_pre_Q, dim=0)

    # advantage = (reward - mean) / (std + small_epsilon).
    advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

    # advantage * logpi: we do torch.exp(per_token_logps - per_token_logps.detach())
    # so it has gradient wrt logpi but value ~ 1.
    per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)

    # Subtract beta*KL (the penalty). Then add a negative sign to switch to "loss" form:
    #  final = -( advantage*logp - beta*KL ).
    per_token_loss = -(per_token_loss - beta * per_token_kl)

    # Multiply by the mask to zero out padded positions, sum over tokens per sequence,
    # then average by the number of valid tokens in that sequence.
    seq_loss = (per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)

    # Finally, average over the batch dimension -> scalar loss.
    loss = seq_loss.mean()
    return loss

###############################################################################
# Initial Data Generation for Bootstrapping
###############################################################################

generate_mode(rank=rank)

###############################################################################
# Main Training Loop
###############################################################################

from tqdm import tqdm
progress = range(1, all_steps+1)
if rank == 0:
    # Display a progress bar only for the main process
    progress = tqdm(progress)

for step in progress:
    # Try to get a batch from the reference server
    batch = get_batch()
    # If no batch is ready, generate some and retry
    while batch is None:
        generate_mode(rank=rank)
        batch = get_batch()

    # Standard PyTorch training steps
    optimizer.zero_grad()       # Reset gradients
    loss = GRPO_step(batch)     # Compute the GRPO loss
    loss.backward()             # Backprop through the model
    optimizer.step()            # Apply AdamW update

    # Show the current loss on the progress bar (if rank=0)
    if rank == 0:
        progress.set_description(f"Loss: {loss.item():.6f}")

    # Save the model periodically
    if step % save_steps == 0:
        if rank == 0:
            print('saving model')
            save_name = f"./step_{step}"
            # Move parameters to CPU before saving to reduce GPU mem usage
            state_dict = engine.state_dict()
            state_dict = type(state_dict)({k: v.cpu() for k, v in state_dict.items()})
            # Save model + tokenizer
            engine.save_pretrained(save_name, state_dict=state_dict)
            tokenizer.save_pretrained(save_name)
