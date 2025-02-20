import torch.distributed
from transformers import AutoTokenizer, AutoModelForCausalLM
import json, os, shutil, re, random, requests, io, sys, time, traceback
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

model_path = "Qwen/Qwen2.5-0.5B"
beta = 0.04
num_pre_Q = 8
Q_batch_size = 1
micro_batch_size = 4
all_steps = 1000
max_prompt_length = 400   
gen_steps = 30
save_steps = 200

ds_config = {
    "train_micro_batch_size_per_gpu": micro_batch_size,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": { "lr": 1e-6 }
    },
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 1,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
        "stage3_gather_16bit_weights_on_model_save": True,
        "offload_optimizer": {"device": "cpu"}
    }
}

ref_server = "http://localhost:59875"
from ref_server import tensor_to_bytes, bytes_to_tensor, make_bytes_list, bytes_list_to_list

def get_batch():
    try:
        r = requests.get(f"{ref_server}/get").content
        if r == b'empty': return None
    except: return None
    dd = bytes_list_to_list(r)
    data = json.loads(dd[0]) 
    data['inputs'] = bytes_to_tensor(dd[1])
    data['rewards'] = bytes_to_tensor(dd[2])
    data['refs'] = bytes_to_tensor(dd[3])
    return data

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, 
        torch_dtype=torch.float32, _attn_implementation="sdpa")
gen_model = model

from datasets import load_dataset
dataset = load_dataset("openai/gsm8k", "main", split="train")
QAs = [{'Q':x, 'A':y.split('####')[-1].strip()} for x,y in zip(dataset['question'], dataset['answer'])]

from transformers import GenerationConfig
generation_config = GenerationConfig(
            max_new_tokens=512,
            do_sample=True, temperature=0.9, 
            num_return_sequences=num_pre_Q,
            pad_token_id=tokenizer.pad_token_id,
        )

system_prompt = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""
def gen_answers(prompts):
    tip_text = []
    for x in prompts:
        tip_text.append(tokenizer.apply_chat_template([
             {"role": "system", "content": system_prompt},
             {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True))
    tip_inputs = tokenizer(tip_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
    prompt_length = tip_inputs["input_ids"].shape[-1]
    if prompt_length > max_prompt_length: return []
    tip_inputs = {k: v.to(model.device) for k, v in tip_inputs.items()}
    with torch.inference_mode():
        tip_completion_ids = gen_model.generate(**tip_inputs, generation_config=generation_config)
    completion_ids = tip_completion_ids[:, prompt_length:]
    answers = [tokenizer.decode(x).replace('<|endoftext|>', '') for x in completion_ids]
    return answers

from math_verify import parse, verify, ExprExtractionConfig
def reward_correct(item, answer):
    pattern = r'\d+\.\d+|\d+/\d+|\d+'
    nums = re.findall(pattern, answer) # 使用正则表达式在answer中查找所有数字
    if len(nums) == 0: return -1.0
    lastnum = nums[-1] # 用answer中最后一个数字和ground_truth做比较
    ans = parse(lastnum, extraction_config=[ExprExtractionConfig()])
    ground_truth = parse(item["A"], extraction_config=[ExprExtractionConfig()])
    return 1 if verify(ans, ground_truth) else -1
def reward_format(item, answer):
    # pattern = r"^<think>(?:(?!</?think>)[\s\S]*?)</think>\s*<answer>(?:(?!</?answer>)[\s\S]*?)</answer><\|im_end\|>$"
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    return 1.25 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) else -1


def gen_samples(inputs):
    prompts = [x["Q"] for x in inputs]
    answers = gen_answers(prompts)
    if len(answers) == 0: return None, None, None
    rewards = []
    for i, inp in enumerate(inputs):
        for a in answers[i*num_pre_Q:(i+1)*num_pre_Q]:
            rewards.append(reward_correct(inp, a) + reward_format(inp, a))
    prompts_text = [tokenizer.apply_chat_template([
             {"role": "system", "content": system_prompt},
             {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True) for x in prompts]
    return prompts_text, rewards, answers


def generate_mode(num=10, rank=0, gen_timeout=999999):
    if rank == 0: print('enter generate mode')
    tic = time.time()
    for ii in range(num):
        if time.time() - tic > gen_timeout: break
        inputs = random.sample(QAs, 1)
        prompts_text, rewards, answers = gen_samples(inputs)
        if prompts_text is None: continue
        xdata = make_bytes_list([
            json.dumps({"Q": prompts_text[0], "As": answers}).encode(), 
            tensor_to_bytes(torch.tensor(rewards, dtype=torch.float32))])
        requests.post(f"{ref_server}/upload", data=xdata)
        if rank != 0: continue
        print('gen:', rewards)
    if rank == 0: print('exit generate mode')
    print(f'{rank}: {time.time()-tic:.3f}s')

if 'genonly' in sys.argv:
    model.to('cpu')
    generate_mode(999999)
    sys.exit()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
engine = model
gen_model = model

def get_per_token_logps(logits, input_ids):
    per_token_logps = [] # Use a loop to reduce memory peak.
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)

from kernel.ce_kernel import fast_log_softmax_gather
get_per_token_logps = fast_log_softmax_gather

def GRPO_step(batch):
    prompt_length = batch['plen']
    inputs = batch['inputs'].to(engine.device)
    advantages = batch['rewards'].to(engine.device)

    logits = engine(inputs).logits
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
    input_ids = inputs[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
    
    per_token_logps = get_per_token_logps(logits, input_ids)
    per_token_logps = per_token_logps[:,prompt_length-1:]
    ref_per_token_logps = batch['refs'].to(engine.device)

    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
    completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()

    per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
    per_token_loss = -(per_token_loss - beta * per_token_kl)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    return loss

#generate_mode(rank=torch.distributed.get_rank())

from tqdm import tqdm
progress = range(1, all_steps+1)
if torch.distributed.get_rank() == 0: progress = tqdm(progress)
for step in progress:
    batch = get_batch()
    while batch is None:
        generate_mode(rank=torch.distributed.get_rank())
        batch = get_batch()

    if step % gen_steps == 0:
        generate_mode(rank=torch.distributed.get_rank(), gen_timeout=100)

    B = batch['inputs'].shape[0]
    assert B % micro_batch_size == 0
    optimizer.zero_grad()
    for sub in range(0, B, micro_batch_size):
        sub_batch = {'plen': batch['plen']}
        sub_batch['inputs'] = batch['inputs'][sub:sub+micro_batch_size]
        sub_batch['rewards'] = batch['rewards'][sub:sub+micro_batch_size]
        sub_batch['refs'] = batch['refs'][sub:sub+micro_batch_size]
        try:
            loss = GRPO_step(sub_batch)
            loss.backward()
        except:
            traceback.print_exc()
            print('batch:', sub_batch['inputs'].shape)
            torch.distributed.destroy_process_group()
            sys.exit()
    optimizer.step()

    if torch.distributed.get_rank() == 0:
        progress.set_description(f"Loss: {loss.item():.6f}")

    if step % save_steps == 0:
        dist.barrier()
        if torch.distributed.get_rank() == 0:
            print('saving model')
            save_name = f"./step_{step}"
            state_dict = model.state_dict()
            state_dict = type(state_dict)({k: v.cpu() for k, v in state_dict.items()})
            model.save_pretrained(save_name, state_dict=state_dict)
            tokenizer.save_pretrained(save_name)
        dist.barrier()
