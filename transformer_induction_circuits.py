# %%
import functools
import sys
from pathlib import Path 

import circuitsvis as cv
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from IPython.display import display
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm 
from transformer_lens import (
    ActivationCache,
    FactoredMatrix,
    HookedTransformer,
    HookedTransformerConfig,
    utils,
)
from transformer_lens.hook_points import HookPoint
import plotly.express as px

torch.set_grad_enabled(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"
# %%
gpt2_small: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")
# %%
if MAIN:
    print("Number of layers: ", gpt2_small.cfg.n_layers)
    print("Number of attention heads: ", gpt2_small.cfg.n_heads)
    print("Maximum context window: ", gpt2_small.cfg.n_ctx)
# %%
model_input_text = '''Lessgo some of Tyrion's Best Quotes Compiled.

Never forget what you are, the rest of the world will not. Wear it like armor and it can never be used to hurt you.

A mind needs books like a sword needs a whetstone.

I have a tender spot in my heart for cripples, bastards, and broken things.

It’s hard to put a leash on a dog once you’ve put a crown on its head.

The powerful have always preyed on the powerless, that's how they became powerful in the first place.
'''
if MAIN:
    loss = gpt2_small(model_input_text,return_type = "loss")
    print("Model Loss:", loss)
# %%
if MAIN:
    logits: Tensor = gpt2_small(model_input_text, return_type = "logits")
    prediction = logits.argmax(dim=-1).squeeze()[:-1]

    true_tokens = gpt2_small.to_tokens(model_input_text).squeeze()[1:]
    is_correct = (prediction == true_tokens)

    print(f"Model accuracy: {is_correct.sum()}/{len(true_tokens)}")
    print(f"Correct tokens: {gpt2_small.to_str_tokens(prediction[is_correct])}")

# %%
if MAIN: 
    print(gpt2_small.to_str_tokens("Jon Snow, King in the North",prepend_bos = False))
# %%
if MAIN:
    gpt2_text = "Natural Language Processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."
    gpt2_tokens = gpt2_small.to_tokens(gpt2_text)
    gpt2_logits, gpt2_cache = gpt2_small.run_with_cache(gpt2_tokens,remove_batch_dim=1)
# %%
if MAIN:
    attn_pattern_layer_0 = gpt2_cache["pattern",0]
    print(attn_pattern_layer_0.shape)
# %%
if MAIN:
    attn_pattern_layer_0_copy = gpt2_cache["blocks.0.attn.hook_pattern"]
    torch.testing.assert_close(attn_pattern_layer_0,attn_pattern_layer_0_copy)
# %%
if MAIN:
    layer0_pattern_from_cache = gpt2_cache["pattern", 0]

    q, k = gpt2_cache["q", 0], gpt2_cache["k", 0]
    seq, nhead, headsize = q.shape
    layer0_attn_scores = einops.einsum(q, k, "seqQ n h, seqK n h -> n seqQ seqK")
    mask = torch.triu(torch.ones((seq, seq), dtype=torch.bool), diagonal=1).to(device)
    layer0_attn_scores.masked_fill_(mask, -1e9)
    layer0_pattern_from_q_and_k = (layer0_attn_scores / headsize**0.5).softmax(-1)


    torch.testing.assert_close(layer0_pattern_from_cache, layer0_pattern_from_q_and_k)
    print("Tests passed!")
# %%
if MAIN:
    print(type(gpt2_cache))
    attention_pattern = gpt2_cache["pattern", 0]
    print(attention_pattern.shape)
    gpt2_str_tokens = gpt2_small.to_str_tokens(gpt2_text)

    print("Layer 0 Head Attention Patterns:")
    display(cv.attention.attention_patterns(
        tokens=gpt2_str_tokens,
        attention=attention_pattern
    ))
# %%
cfg = HookedTransformerConfig(
    d_model = 768,
    d_head = 64,
    n_heads = 12,
    n_layers = 2,
    n_ctx = 2048,
    d_vocab = 50278,
    attention_dir = "causal",
    attn_only = True,
    tokenizer_name = "EleutherAI/gpt-neox-20b",
    seed = 398,
    use_attn_result = True,
    normalization_type = None,
    positional_embedding_type = "shortformer"
)
# %%
from huggingface_hub import hf_hub_download
REPO_ID = "callummcdougall/attn_only_2L_half"
FILENAME = "attn_only_2L_half.pth"
weights_path = hf_hub_download(repo_id=REPO_ID,filename=FILENAME)
# %%
model = HookedTransformer(cfg)
pretrained_weights = torch.load(weights_path,map_location=device)
model.load_state_dict(pretrained_weights)
# %%
if MAIN:
    text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."
    logits, cache = model.run_with_cache(text,remove_batch_dim=True)
    str_tokens = model.to_str_tokens(text)
    for layer in range(model.cfg.n_layers):
        attention_pattern = cache["pattern", layer]
        display(cv.attention.attention_patterns(tokens=str_tokens,attention= attention_pattern))
# %%
def current_attn_detector(cache: ActivationCache) -> list[str]:
    '''
    Returns a list e.g. ["0.2","1.4","1.9"] of "layer.head" which are judged to be current-token heads.
    '''
    attn_heads = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            attention_pattern = cache["pattern",layer][head]
            score = attention_pattern.diagonal().mean()
            if score > 0.4:
                attn_heads.append(f"{layer}.{head}")
    return attn_heads

def prev_attn_detector(cache: ActivationCache) -> list[str]:
    '''
    Returns a list e.g. ["0.2","1.4","1.9"] of "layer.head" which are judged to be previous-token heads.
    '''
    attn_heads = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            attention_pattern = cache["pattern",layer][head]
            score = attention_pattern.diagonal(-1).mean()
            if score > 0.4:
                attn_heads.append(f"{layer}.{head}")
    return attn_heads

def first_attn_detector(cache: ActivationCache) -> list[str]:
    '''
    Returns a list e.g. ["0.2","1.4","1.9"] of "layer.head" which are judged to be first-token heads.
    '''
    attn_heads = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            attention_pattern = cache["pattern",layer][head]
            score = attention_pattern[:,0].mean()
            if score > 0.4:
                attn_heads.append(f"{layer}.{head}")
    return attn_heads
# %%
if MAIN:
    print("Heads attending to the current token:"), ", ".join(current_attn_detector(cache))
    print("Heads attending to the previous token:"), ", ".join(prev_attn_detector(cache))
    print("Heads attending to the first token:"), ", ".join(first_attn_detector(cache))
# %%
from plotly_utils import plot_loss_difference 
def generate_repeated_tokens(model: HookedTransformer, seq_len: int, batch: int = 1) -> Tensor:
    '''
    Generates a sequence of repeated random tokens

    Outputs are:
        rep_tokens: [batch, 1+2*seq_len]
    '''
    prefix = (torch.ones(batch,1) * model.tokenizer.bos_token_id).long()
    rep_tokens_half = torch.randint(0, model.cfg.d_vocab, (batch, seq_len), dtype = torch.int64)
    rep_tokens = torch.cat([prefix, rep_tokens_half, rep_tokens_half], dim = -1).to(device)
    return rep_tokens

def run_and_cache_model_repeated_tokens(model: HookedTransformer, seq_len: int, batch: int = 1) -> tuple[Tensor, Tensor, ActivationCache]:
    '''
    Generates a sequence of repeated random tokens, and runs the model on it, returning (tokens, logits, cache)

    Should use the `generate_repeated_tokens` function above

    Outputs are: 
        rep_tokens: [batch, 1+2*seq_len]
        rep_logits: [batch, 1+2*seq_len, d_vocab]
        rep_cache: The cache of the model run on rep_tokens
    '''
    rep_tokens = generate_repeated_tokens(model, seq_len, batch)
    rep_logits, rep_cache = model.run_with_cache(rep_tokens)
    return rep_tokens, rep_logits, rep_cache

if MAIN:
    seq_len = 50
    batch = 1
    (rep_tokens, rep_logits, rep_cache) = run_and_cache_model_repeated_tokens(model,seq_len, batch)
    rep_cache.remove_batch_dim()
    rep_str = model.to_str_tokens(rep_tokens)
    model.reset_hooks()
    log_probs =rep_logits.log_softmax(dim = -1)[:, : -1].gather(dim=-1,index=rep_tokens[:,1:].unsqueeze(-1)).squeeze(-1)

    print(f"Performance on the first half: {log_probs[:seq_len].mean():.3f}")
    print(f"Performance on the second half: {log_probs[seq_len:].mean():.3f}")

    plot_loss_difference(log_probs, seq_len)
# %%
def induction_attn_detector(cache: ActivationCache) -> list[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be induction heads

    Remember - the tokens used to generate rep_cache are (bos_token, *rand_tokens, *rand_tokens)
    '''
    attn_heads = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            attention_pattern = cache["pattern", layer][head]
            seq_len = (attention_pattern.shape[-1]-1)//2
            score = attention_pattern.diagonal(-seq_len+1).mean()
            if score > 0.4:
                attn_heads.append(f"{layer}.{head}")
    return attn_heads

if MAIN:
    print("Induction heads:"), ", ".join(induction_attn_detector(rep_cache))
# %%
