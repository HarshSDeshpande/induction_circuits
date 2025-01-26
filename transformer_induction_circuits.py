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
import matplotlib.pyplot as plt
from typing import Callable
from torch import Tensor
from tqdm import tqdm 
import seaborn as sns
from transformer_lens import (
    ActivationCache,
    FactoredMatrix,
    HookedTransformer,
    HookedTransformerConfig,
    utils,
)
import plotly.express
import plotly_utils
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
seq_len = 50
batch = 10
rep_tokens_10 = generate_repeated_tokens(model, seq_len, batch)

induction_score_store = torch.zeros((model.cfg.n_layers, model.cfg.n_heads),device=model.cfg.device)

def induction_score_hook(
    pattern: Tensor,
    hook: HookPoint,
):
    '''
    Calculates the induction score, and stores it in the  [layer, head] position of the `induction_score_store` tensor.
    '''
    induction_stripe = pattern.diagonal(dim1=-2,dim2=-1,offset=1-seq_len)
    induction_score = einops.reduce(induction_stripe, "batch head_index position -> head_index", "mean")
    induction_score_store[hook.layer(), :] = induction_score

if MAIN:
    model.run_with_hooks(
        rep_tokens_10,
        return_type = None,
        fwd_hooks = [(lambda name: name.endswith("pattern"), induction_score_hook)]
    )
    f0 = plotly.express.imshow(
        induction_score_store.cpu().numpy(),
        labels={"x": "Head", "y": "Layer"},
        title = "Induction Score by Head",
        text_auto='.2f',
        width=900, height=400
    )
    f0.show()
# %%
def visualize_pattern_hook(
    pattern: Tensor,
    hook: HookPoint,
): 
    print("Layer:", hook.layer())
    display(
        cv.attention.attention_patterns(
            tokens=gpt2_small.to_str_tokens(rep_tokens[0]),
            attention = pattern.mean(0)
        )
    )
# %%
seq_len = 50
batch = 10
rep_tokens_10 = generate_repeated_tokens(gpt2_small,seq_len,batch)

induction_score_store = torch.zeros((gpt2_small.cfg.n_layers, gpt2_small.cfg.n_heads),device=gpt2_small.cfg.device)

gpt2_small.run_with_hooks(
    rep_tokens_10,
    return_type = None,
    fwd_hooks = [(
        lambda name: name.endswith("pattern"),
        induction_score_hook
    )]
)

f1 = plotly.express.imshow(
    induction_score_store.cpu().numpy(),
    labels={"x": "Head", "y": "Layer"}, 
    title="Induction Score by Head",
    text_auto=".1f",
    width=800
)
f1.show()
# %%
if MAIN:
    for induction_head_layer in [5,6,7]:
        gpt2_small.run_with_hooks(
            rep_tokens,
            return_type = None,
            fwd_hooks = [
                (utils.get_act_name("pattern",induction_head_layer),visualize_pattern_hook)
            ]
        )
# %%
def logit_attribution(
    embed: Tensor,
    l1_results: Tensor,
    l2_results: Tensor,
    W_U: Tensor,
    tokens: Tensor
) -> Tensor:
    '''
    Inputs:
        embed: the embeddings of the tokens (i.e. token + position embeddings)
        l1_results: the outputs of the attention heads at layer 1 (with head as one of the dimensions)
        l2_results: the outputs of the attention heads at layer 2 (with head as one of the dimensions)
        W_U: the unembedding matrix
        tokens: the token ids of the seequence

    Returns:
         Tensor of shape (seq_len-1, n_components)
         represents the concatenations (along dim=-1) of logit attributions from:
            - the direct path (seq-1,1)
            - layer 0 logits (seq-1,n_heads)
            - layer 1 logits (seq-1,n_heads)
         so n_components = 1+2*n_heads
    '''
    W_U_correct_tokens = W_U[:,tokens[1:]]
    direct_attributions = einops.einsum(W_U_correct_tokens,embed[:-1],"emb seq, seq emb -> seq")
    l1_attributions = einops.einsum(W_U_correct_tokens,l1_results[:-1],"emb seq, seq nhead emb -> seq nhead")
    l2_attributions = einops.einsum(W_U_correct_tokens,l2_results[:-1],"emb seq, seq nhead emb -> seq nhead")
    return torch.concat([direct_attributions.unsqueeze(-1), l1_attributions, l2_attributions],dim=-1)
# %%
if MAIN:
    text = "They say we die twice. Once when our breath leaves our body and once when the last person we know says our name."
    logits, cache = model.run_with_cache(text, remove_batch_dim=True)
    str_tokens = model.to_str_tokens(text)
    tokens = model.to_tokens(text)

    with torch.inference_mode():
        embed = cache["embed"]
        l1_results = cache["result",0]
        l2_results = cache["result",1]
        logit_attr = logit_attribution(embed,l1_results,l2_results,model.W_U,tokens[0])
        correct_token_logits = logits[0, torch.arange(len(tokens[0])-1),tokens[0,1:]]
        torch.testing.assert_close(logit_attr.sum(1),correct_token_logits,atol=1e-3,rtol=0)
        print("Tests passed!")
# %%
def get_log_probs(
    logits: Tensor, tokens: Tensor
) -> Tensor:
    log_probs = logits.log_softmax(dim=-1)
    log_probs_for_tokens = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)

    return log_probs_for_tokens
# %%
def head_zero_ablation_hook(
    z: Tensor,
    hook: HookPoint,
    head_index_to_ablate: int,
) -> None:
    z[:,:,head_index_to_ablate,:] = 0.0

def get_ablation_scores(
    model: HookedTransformer,
    tokens: Tensor,
    ablation_function: Callable = head_zero_ablation_hook,
) -> Tensor:
    '''
    Returns a tensor of shape (n_layers, n_heads) containing the increase in cross entropy loss from ablating the output of each head
    '''
    ablation_scores = torch.zeros((model.cfg.n_layers, model.cfg.n_heads),device=model.cfg.device)
    model.reset_hooks()
    seq_len = (tokens.shape[1] - 1)//2
    logits = model(tokens,return_type="logits")
    loss_no_ablation = -get_log_probs(logits, tokens)[:,-(seq_len-1):].mean()

    for layer in tqdm(range(model.cfg.n_layers)):
        for head in range(model.cfg.n_heads):
            temp_hook_fn = functools.partial(ablation_function, head_index_to_ablate=head)
            ablated_logits = model.run_with_hooks(tokens,fwd_hooks=[(utils.get_act_name("z",layer),temp_hook_fn)])
            loss = -get_log_probs(ablated_logits.log_softmax(-1), tokens)[:, -(seq_len - 1) :].mean()
            ablation_scores[layer, head] = loss - loss_no_ablation

    return ablation_scores


if MAIN:
    ablation_scores = get_ablation_scores(model,rep_tokens)
# %%
if MAIN:
    plt.imshow(
        ablation_scores.cpu().numpy(),
    )

# %%
def head_mean_ablation_hook(
    z:Tensor,
    hook: HookPoint,
    head_index_to_ablate: int,
) -> None:
    z[:,:, head_index_to_ablate, :] = z[:, :, head_index_to_ablate,:].mean(0)

if MAIN:
    rep_tokens_batch = run_and_cache_model_repeated_tokens(model,seq_len=10,batch=50)[0]
    mean_ablation_scores = get_ablation_scores(model,rep_tokens_batch,ablation_function=head_mean_ablation_hook)
    plt.imshow(mean_ablation_scores.cpu().numpy())

# %%
if MAIN:
    head_index = 4
    layer = 1

    W_O = model.W_O[layer, head_index]
    W_V = model.W_V[layer, head_index]
    W_E = model.W_E
    W_U = model.W_U

    OV_circuit = FactoredMatrix(W_V,W_O)
    full_OV_circuit = W_E @ OV_circuit @ W_U
# %%
if MAIN:
    indices = torch.randint(0,model.cfg.d_vocab,(200,))
    full_OV_circuit_sample = full_OV_circuit[indices,indices].AB

    f3 = plotly.express.imshow(
        full_OV_circuit_sample.cpu().numpy(),
        labels={"x": "Logits on output token", "y": "Input token"},
        title = "Full OV circuit for copying head",
        width = 700,
        height=600,
    )
    f3.show()
# %%
def top_1_acc(full_OV_circuit: FactoredMatrix, batch_size: int = 1000) -> float:
    """
    Compute the argmax of each column (ie over dim=0) and return the fraction of the time that the maximum value is on 
    the circuit diagonal.
    """
    total = 0
    
    for indices in torch.split(torch.arange(full_OV_circuit.shape[0], device=device),batch_size):
        AB_slice = full_OV_circuit[indices].AB
        total += (torch.argmax(AB_slice,dim=1)==indices).float().sum().item()

    return total/full_OV_circuit.shape[0]

if MAIN:
    print(top_1_acc(full_OV_circuit))
# %%
if MAIN:
    W_O_both = einops.rearrange(model.W_O[1,[4,10]],"head d_head d_model -> (head d_head) d_model")
    W_V_both = einops.rearrange(model.W_V[1,[4,10]], "head d_model d_head -> d_model (head d_head)")
    W_OV_eff = W_E @ FactoredMatrix(W_V_both,W_O_both) @ W_U
    print(top_1_acc(W_OV_eff))
# %%
if MAIN:
    layer = 0
    head_index = 7

    W_pos = model.W_pos
    W_QK = model.W_Q[layer, head_index] @ model.W_K[layer, head_index].T
    pos_by_pos_scores = W_pos @ W_QK @ W_pos.T

    mask = torch.tril(torch.ones_like(pos_by_pos_scores)).bool()
    pos_by_pos_pattern = torch.where(mask, pos_by_pos_scores / model.cfg.d_head**0.5, -1.0e6).softmax(-1)

    print(f"Avg lower-diagonal value: {pos_by_pos_pattern.diag(-1).mean():.4f}")
    f4 = plotly.express.imshow(
        utils.to_numpy(pos_by_pos_pattern[:200,:200]),
        labels={"x": "Key", "y": "Query"},
        title = "Attention patterns for prev-token QK circuit, first 100 indices",
        width = 700,
        height =600,
    )
    f4.show()
# %%
def decompose_qk_input(cache: ActivationCache) -> Tensor:
    y0 = cache["embed"].unsqueeze(0)
    y1 = cache["pos_embed"].unsqueeze(0)
    y_rest = cache["result",0].transpose(0,1)
    return torch.concat([y0,y1,y_rest],dim=0)
# %%
def decompose_q(
    decomposed_qk_input: Tensor,
    ind_head_index: int,
    model: HookedTransformer,
) -> Tensor:
    W_Q =  model.W_Q[1,ind_head_index]
    return einops.einsum(decomposed_qk_input,W_Q,"n seq d_model, d_model d_head -> n seq d_head")
# %%
def decompose_k(
    decomposed_qk_input: Tensor,
    ind_head_index: int,
    model: HookedTransformer,
) -> Tensor:
    W_K = model.W_K[1,ind_head_index]
    return einops.einsum(decomposed_qk_input,W_K,"n seq d_model, d_model d_head -> n seq d_head")
# %%
if MAIN:
    seq_len = 50
    batch_size = 1
    (rep_tokens, rep_logits, rep_cache) = run_and_cache_model_repeated_tokens(model,seq_len,batch_size)
    rep_cache.remove_batch_dim()
    
    ind_head_index = 4

    decomposed_qk_input = decompose_qk_input(rep_cache)
    decomposed_q = decompose_q(decomposed_qk_input, ind_head_index, model)
    decomposed_k = decompose_k(decomposed_qk_input,ind_head_index, model)
    torch.testing.assert_close(decomposed_qk_input.sum(0),rep_cache["resid_pre",1]+rep_cache["pos_embed"],rtol=0.01,atol=1e-05)
    torch.testing.assert_close(decomposed_q.sum(0),rep_cache["q",1][:,ind_head_index],rtol=0.01,atol=0.001)
    torch.testing.assert_close(decomposed_k.sum(0),rep_cache["k",1][:, ind_head_index],rtol=0.01,atol=0.01)

    component_labels = ["Embed", "PosEmbed"] + [f"0.{h}" for h in range(model.cfg.n_heads)]
    for decomposed_input, name in [(decomposed_q, "query"),(decomposed_k,"key")]:
        fig = plotly.express.imshow(
            utils.to_numpy(decomposed_input.pow(2).sum([-1])),
            labels={"x":"Position", "y": "Component"},
            title = f"Norms of components of {name}",
            y = component_labels,
            width = 800,
            height = 400,
        )
        fig.show()

# %%
def decompose_attn_scores(decomposed_q: torch.Tensor, decomposed_k: torch.Tensor) -> Tensor:
    return einops.einsum(
        decomposed_q,
        decomposed_k,
        "q_comp q_pos d_model, k_comp k_pos d_model -> q_comp k_comp q_pos k_pos",
    )
# %%
if MAIN:
    decomposed_scores = decompose_attn_scores(decomposed_q, decomposed_k)
    decomposed_stds = einops.reduce(
        decomposed_scores, "query_decomp key_decomp query_pos key_pos -> query_decomp key_decomp", torch.std
    )

    f10 = plotly.express.imshow(
        utils.to_numpy(torch.tril(decomposed_scores[0,9])),
        title = "Attention score contributions from query = embed, key = output of L0H7<br>(by query & key sequence positions)",
        width = 700,
    )
    f10.show()

    f11 = plotly.express.imshow(
        utils.to_numpy(decomposed_stds),
        labels={"x": "Key Compnent", "y": "Query Component"},
        title = "Standard deviations of attn score contributions across sequence positions<br> (by query and key component)",
        x = component_labels,
        y = component_labels,
        width = 700,
    )
    f11.show()
# %%
def find_K_comp_full_circuit(
    model: HookedTransformer,
    prev_token_head_index: int,
    ind_head_index: int,
) -> FactoredMatrix:
    W_E = model.W_E
    W_Q = model.W_Q[1, ind_head_index]
    W_K = model.W_K[1, ind_head_index]
    W_O = model.W_O[0, prev_token_head_index]
    W_V = model.W_V[0, prev_token_head_index]
    Q = W_E @ W_Q
    K = W_E @ W_V @ W_O @ W_K
    return FactoredMatrix(Q, K.T)
# %%
if MAIN:
    prev_token_head_index = 7
    ind_head_index = 4
    K_comp_circuit = find_K_comp_full_circuit(model, prev_token_head_index, ind_head_index)
    print(f"Fraction of tokens where the highest activating key is the same token: {top_1_acc(K_comp_circuit.T):.4f}")
# %%
def get_comp_score(W_A: Tensor, W_B: Tensor) -> float:
    W_A_norm = W_A.pow(2).sum().sqrt()
    W_B_norm = W_B.pow(2).sum().sqrt()
    W_AB_norm = (W_A @ W_B).pow(2).sum().sqrt()
    return (W_AB_norm / (W_A_norm*W_B_norm)).item()

#%%
def plot_comp_scores(model, scores, title):
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(
        scores.cpu().detach(),
        annot=True,
        fmt='.3f',  # Changed to 3 decimal places for more precision in small values
        cmap='YlOrRd',  # Yellow-Orange-Red colormap better for positive values
        vmin=0,
        vmax=0.1,
        square=True,
        cbar_kws={'label': 'Composition Score'}
    )
    
    # Configure labels
    plt.xlabel("Layer 1 Heads")
    plt.ylabel("Layer 0 Heads")
    plt.title(title)
    
    # Set ticks to show head numbers
    n_heads = model.cfg.n_heads
    plt.xticks(np.arange(n_heads) + 0.5, range(n_heads))
    plt.yticks(np.arange(n_heads) + 0.5, range(n_heads))
    
    plt.tight_layout()
    plt.show()
# %%
if MAIN:
    W_QK = model.W_Q @ model.W_K.transpose(-1,-2)
    W_OV = model.W_V @ model.W_O
    composition_scores = {
        "Q": torch.zeros(model.cfg.n_heads, model.cfg.n_heads).to(device),
        "K": torch.zeros(model.cfg.n_heads, model.cfg.n_heads).to(device),
        "V": torch.zeros(model.cfg.n_heads, model.cfg.n_heads).to(device),
    }
    
    for i in tqdm(range(model.cfg.n_heads)):
        for j in range(model.cfg.n_heads):
            composition_scores["Q"][i,j] = get_comp_score(W_OV[0,i],W_QK[1,j])
            composition_scores["K"][i,j] = get_comp_score(W_OV[0,i],W_QK[1,j].T)
            composition_scores["V"][i,j] = get_comp_score(W_OV[0,i],W_QK[1,j])
            
    for comp_type in ["Q","K","V"]:
        plot_comp_scores(model, composition_scores[comp_type], f"{comp_type} Composition Scores")
# %%
def generate_single_random_comp_score() -> float:
    W_A_left = torch.empty(model.cfg.d_model, model.cfg.d_head)
    W_B_left = torch.empty(model.cfg.d_model, model.cfg.d_head)
    W_A_right = torch.empty(model.cfg.d_model, model.cfg.d_head)
    W_B_right = torch.empty(model.cfg.d_model, model.cfg.d_head)

    for W in [W_A_left, W_B_left, W_A_right, W_B_right]:
        nn.init.kaiming_uniform_(W, a = np.sqrt(5))

    W_A = W_A_left @ W_A_right.T
    W_B = W_B_left @ W_B_right.T

    return get_comp_score(W_A, W_B)
# %%
if MAIN:
    n_samples = 300
    comp_scores_baseline = np.zeros(n_samples)
    for i in tqdm(range(n_samples)):
        comp_scores_baseline[i] = generate_single_random_comp_score()

    print("\n Mean:", comp_scores_baseline.mean())
    print("Std:", comp_scores_baseline.std())
    
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=comp_scores_baseline, 
        bins=50,
        kde=True
    )
    plt.xlabel("Composition Score")
    plt.title("Random Composition Scores")
    plt.show()
