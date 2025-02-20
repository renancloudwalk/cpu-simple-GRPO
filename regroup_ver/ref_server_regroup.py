
import json, os, shutil, re, random, io, time
import torch

def tensor_to_bytes(t):
    buffer = io.BytesIO()
    torch.save(t, buffer)
    return buffer.getvalue()
def bytes_to_tensor(b):
    return torch.load(io.BytesIO(b))
def make_bytes_list(blist):
    buffer = io.BytesIO()
    buffer.write(len(blist).to_bytes(4, 'big'))
    for b in blist:
        buffer.write(len(b).to_bytes(4, 'big'))
        buffer.write(b)
    return buffer.getvalue()
def bytes_list_to_list(b):
    buffer = io.BytesIO(b)
    num = int.from_bytes(buffer.read(4), 'big')
    blist = []
    for _ in range(num):
        l = int.from_bytes(buffer.read(4), 'big')
        blist.append(buffer.read(l))
    return blist

class QAPool:
    def __init__(self):
        self.lock = threading.Lock()
        self.pool = {}
        self.num = 0
    def put(self, Q, As, Rs):
        with self.lock:
            if Q not in self.pool: self.pool[Q] = []
            for A, R in zip(As, Rs):
                self.pool[Q].append((A, R))
            self.num += len(As)
            if len(self.pool[Q]) > 100:
                self.pool[Q] = self.pool[Q][-100:]

    def sample_group(self):
        while self.num < 60: time.sleep(0.5)
        while True:
            lst = list(self.pool.keys())
            random.shuffle(lst)
            for Q in lst:
                group = None
                with self.lock:
                    if len(self.pool[Q]) >= 8:
                        group = random.sample(self.pool[Q], 8)
                if group is None: continue
                As, Rs = zip(*group)
                Rs = torch.tensor(Rs, dtype=torch.float32)
                if Rs.max() <= Rs.min() + 0.1: continue
                Rs = (Rs - Rs.mean()) / Rs.std()
                yield Q, As, Rs
            time.sleep(5)

if __name__ == '__main__':   
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    import torch.nn as nn

    from bottle import request
    import bottle, threading, queue
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'

    model_path = "Qwen/Qwen2.5-0.5B"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    ref_model = AutoModelForCausalLM.from_pretrained(model_path,
            torch_dtype=torch.bfloat16, _attn_implementation="sdpa").to('cuda')
    ref_model.eval()
    ref_model.requires_grad_(False)

    def get_per_token_logps(input_ids):
        logits = ref_model(input_ids).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    raw_queue = queue.Queue()
    result_queue = queue.Queue()
    pool = QAPool()

    app = bottle.Bottle()

    @app.route('/upload', method='POST')
    def do_upload():
        dd = request.body.read()
        dd = bytes_list_to_list(dd)
        data = json.loads(dd[0])
        rewards = bytes_to_tensor(dd[1])
        pool.put(data['Q'], data['As'], rewards)
        print('receive', rewards, '  num:', pool.num, ' ready:', result_queue.qsize())

    @app.route('/get', method='GET')
    def do_get():
        if result_queue.empty(): return b'empty'
        return result_queue.get()

    def run_server(): bottle.run(app, host='0.0.0.0', port=59875, server='tornado')
    threading.Thread(target=run_server, daemon=False).start()

    gen = pool.sample_group()
    for Q, As, Rs in gen:
        prompt_inputs = tokenizer([Q], return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)["input_ids"]
        output_ids = tokenizer(As, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False)["input_ids"]
        
        rep = output_ids.shape[0] // prompt_inputs.shape[0]
        prompt_length = prompt_inputs.shape[1]
        Qrep = prompt_inputs.repeat(1, rep).view(-1, prompt_length)
        merged_ids = torch.cat([Qrep, output_ids], dim=1)

        with torch.inference_mode():
            per_token_logps = get_per_token_logps(merged_ids.to(ref_model.device))
        per_token_logps = per_token_logps[:,prompt_length-1:]

        xdata = make_bytes_list([json.dumps({'plen':prompt_length}).encode(), 
                                 tensor_to_bytes(merged_ids), 
                                 tensor_to_bytes(Rs),
                                 tensor_to_bytes(per_token_logps)])
        result_queue.put(xdata)
        if result_queue.qsize() > 100: time.sleep(1)
