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
