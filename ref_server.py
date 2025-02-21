"""
Overview:
---------
This file implements a minimal 'reference server' that:
1) Runs a small web server (using Bottle) to accept HTTP POST requests containing
   serialized model inputs (prompt + answer tokens) and their associated rewards.
2) Uses a pretrained reference model (ref_model) to compute per-token log probabilities
   for the generated portion of each sample. This is done to provide a stable reference
   baseline in reinforcement learning tasks (e.g., GRPO).
3) Returns the newly-computed reference log-probs back to the training script via a GET
   endpoint, along with the original data (prompt length, input IDs, and rewards).

Data Flow Summary:
------------------
- Training script calls POST /upload with batch data (inputs, rewards, some metadata).
- The data is stored in a 'raw_queue'.
- A loop in this file reads from 'raw_queue', computes the reference log-probs, and
  places the result into 'result_queue'.
- Training script calls GET /get to retrieve processed data (same inputs + a new tensor
  containing reference log-probs) from 'result_queue'.

Additionally, this file provides several helper functions to:
- Serialize/deserialize tensors to bytes (tensor_to_bytes, bytes_to_tensor).
- Bundle multiple byte segments into one payload and unbundle them (make_bytes_list, bytes_list_to_list).
These utilities ensure that each batch can be efficiently transferred as a single
binary object, rather than multiple scattered parts.

The main loop keeps running, listening for new data (POST /upload),
processing it, and making processed batches available (GET /get) for the training script.
"""
import json, os, shutil, re, random, io
import torch

def tensor_to_bytes(t):
    """
    Serialize a PyTorch tensor 't' into raw bytes using torch.save.
    We first write it to an in-memory BytesIO buffer and then return the buffer contents.

    Why: This function allows us to easily convert tensors to a bytestring for
    transmission or storage (e.g., sending over HTTP).
    """
    buffer = io.BytesIO()
    torch.save(t, buffer)
    return buffer.getvalue()

def bytes_to_tensor(b):
    """
    Deserialize raw bytes 'b' back into a PyTorch tensor by loading from an in-memory BytesIO stream.

    Why: This is the reverse operation of 'tensor_to_bytes', enabling us to rebuild
    the exact tensor on the receiving end of a network or file transfer.
    """
    return torch.load(io.BytesIO(b))

def make_bytes_list(blist):
    """
    Take a list of byte-objects 'blist' and pack them into a single bytestring
    with headers indicating how many items there are and each item's length.

    Steps:
      1) Write the number of byte-items (4-byte integer).
      2) For each item, write its size (4 bytes), then the item bytes.

    Why: This creates a simple custom format to bundle multiple distinct
    byte segments together into one cohesive payload, so we can transmit them in one go.
    """
    buffer = io.BytesIO()
    # Number of items in blist (4 bytes in big-endian)
    buffer.write(len(blist).to_bytes(4, 'big'))
    for b in blist:
        # First write the size of this byte segment
        buffer.write(len(b).to_bytes(4, 'big'))
        # Then write the segment itself
        buffer.write(b)
    return buffer.getvalue()

def bytes_list_to_list(b):
    """
    Unpack a single bytestring 'b' (created by make_bytes_list) back into
    a list of byte-objects.

    Steps:
      1) Read the count of segments (4 bytes).
      2) For each segment, read its length (4 bytes), then read that many bytes.

    Why: This is the companion function to 'make_bytes_list', restoring
    the original list of byte segments after being transmitted or stored.
    """
    buffer = io.BytesIO(b)
    num = int.from_bytes(buffer.read(4), 'big')
    blist = []
    for _ in range(num):
        l = int.from_bytes(buffer.read(4), 'big')
        blist.append(buffer.read(l))
    return blist

if __name__ == '__main__':
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    import torch.nn as nn

    from bottle import request
    import bottle, threading, queue

    # Disable parallelism in tokenizers for consistent performance/avoid overhead
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'

    model_path = "Qwen/Qwen2.5-0.5B"

    # Reference model loaded on CPU, used to provide log-probabilities for a stable baseline.
    # We keep it in eval mode and disable gradient to reduce overhead.
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        _attn_implementation="sdpa"
    ).to('cpu')
    ref_model.eval()
    ref_model.requires_grad_(False)

    def get_per_token_logps(input_ids):
        """
        Compute per-token log probabilities from 'ref_model' for the sequence 'input_ids'.
        - We run the model forward to get logits.
        - We drop the final logit (because it has no corresponding next-token in input_ids).
        - Then gather the log-prob of each actual token at each position.

        Why: This is used for computing the reference policy's likelihood of tokens,
        which is crucial in KL-divergence or reward shaping for RL with reference.
        """
        logits = ref_model(input_ids).logits  # shape: (B, L, V)
        # Exclude the last logit, which corresponds to the next token after the final one in input_ids
        logits = logits[:, :-1, :]  # shape: (B, L-1, V)

        # Align with input_ids (dropping the first token, since there's no logit for it)
        input_ids = input_ids[:, 1:]  # (B, L-1)

        # We'll compute log_probs in a loop per sequence to mitigate memory usage
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            # Convert raw logits to log probabilities
            log_probs = logits_row.log_softmax(dim=-1)
            # Extract the log_prob for the token that actually appeared
            token_log_prob = torch.gather(
                log_probs, dim=1, index=input_ids_row.unsqueeze(1)
            ).squeeze(1)
            per_token_logps.append(token_log_prob)

        return torch.stack(per_token_logps)

    # Two queues for communication:
    # 1) raw_queue: where incoming requests (batches) are stored before processing
    # 2) result_queue: where processed results (with reference log-probs) go,
    #                  ready to be retrieved.
    raw_queue = queue.Queue()
    result_queue = queue.Queue()

    # Setting up a minimal Bottle web app to handle data upload and retrieval
    app = bottle.Bottle()

    @app.route('/upload', method='POST')
    def do_upload():
        """
        Endpoint to receive training batches:
          - Batches come in as a single bytestring, which we convert back into a list.
          - We parse out base metadata, 'inputs' (the tokens), and 'rewards'.
          - The data is then put into 'raw_queue' for processing.

        Why: This decouples the training script from the reference server,
        allowing us to asynchronously compute reference log-probs.
        """
        dd = request.body.read()        # Read raw bytes from request
        dd = bytes_list_to_list(dd)     # Unpack them into multiple segments
        data = {'base': json.loads(dd[0])}
        data['inputs'] = bytes_to_tensor(dd[1])
        data['rewards'] = bytes_to_tensor(dd[2])
        raw_queue.put(data)
        print('receive', data['inputs'].shape, data['rewards'])

    @app.route('/get', method='GET')
    def do_get():
        """
        Endpoint to provide processed data (with reference log-probs) back to the training script.
        If there's nothing in 'result_queue', returns b'empty'.

        Why: The training script periodically calls /get to fetch newly processed
        batches, so it can compute the final GRPO step with ref log-probs.
        """
        if result_queue.empty():
            return b'empty'
        return result_queue.get()

    def run_server():
        """
        Start the Bottle web server on port 59875 (host='0.0.0.0').
        Using 'tornado' as the server backend for improved performance.
        """
        bottle.run(app, host='0.0.0.0', port=59875, server='tornado')

    # Launch the Bottle server in a separate thread so main thread can process the queue
    threading.Thread(target=run_server, daemon=False).start()

    # Main loop: continually waits for data in raw_queue, computes reference log-probs,
    # and places the result (including the log-probs) into result_queue.
    while True:
        d = raw_queue.get()  # Wait until there's data (inputs, rewards, etc.)
        prompt_length = d['base']['plen']
        with torch.inference_mode():
            per_token_logps = get_per_token_logps(d['inputs'].to(ref_model.device))
        # Keep only the log-probs for the completion portion (exclude prompt tokens),
        # which starts at index (prompt_length - 1).
        per_token_logps = per_token_logps[:, prompt_length-1:]

        # Pack everything (including the newly computed log-probs) into a single bytes object
        xdata = make_bytes_list([
            json.dumps(d['base']).encode(),
            tensor_to_bytes(d['inputs']),
            tensor_to_bytes(d['rewards']),
            tensor_to_bytes(per_token_logps)
        ])
        # Put this bytes package on the result_queue for the training script to fetch.
        result_queue.put(xdata)
