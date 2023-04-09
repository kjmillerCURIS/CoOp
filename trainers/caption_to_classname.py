import os
import sys
import torch
import torch.nn.functional as F

#expect X to have shape (N, *, **, tkn_dim)
#pad it to get shape (N, *, gramlen, tkn_dim)
def pad(X, sentinel_embedding, gramlen):
    if X.shape[2] == gramlen:
        return X

    assert(X.shape[2] < gramlen)
    padder = sentinel_embedding.unsqueeze(0).unsqueeze(0).unsqueeze(0) #(1, 1, 1, tkn_dim)
    padder = padder.tile((X.shape[0], X.shape[1], gramlen - X.shape[2], 1))
    return torch.cat([X, padder], dim=2)

#will return processed_caption_tokens, mask
#processed_caption_tokens will have shape (N, *, gramlen, tkn_dim). No, they won't be normalized (you'll have to do that yourself).
#mask will be binary and have shape (N, *)
def process_caption(caption_tokens, gramlen, tokenized_captions, sentinel_embedding):

    #build base mask
    z = tokenized_captions.argmax(dim=-1, keepdim=True).cuda() #(N, 1)
    c = torch.arange(caption_tokens.shape[1]).unsqueeze(0).cuda() #(1, n_tkn)
    base_mask = (c >= z).type(torch.int64)

    out_list = []
    mask_list = []
    for g in range(1, gramlen+1):
        if g == 1:
            out_list.append(pad(caption_tokens.unsqueeze(2), sentinel_embedding, gramlen))
            mask_list.append(base_mask)
        else:
            stacked_caps = []
            stacked_masks = []
            for gg in range(g):
                stacked_caps.append(caption_tokens[:,gg:gg+caption_tokens.shape[1]-g+1,:])
                stacked_masks.append(base_mask[:,gg:gg+caption_tokens.shape[1]-g+1])

            stacked_caps = torch.stack(stacked_caps, dim=2) #(N, *, g, tkn_dim)
            stacked_masks = torch.stack(stacked_masks, dim=2) #(N, *, g)
            out_list.append(pad(stacked_caps, sentinel_embedding, gramlen))
            mask_list.append(stacked_masks.max(dim=2)[0])

    return torch.cat(out_list, dim=1), torch.cat(mask_list, dim=1)

#caption_tokens should be shape (N, n_tkn, tkn_dim)
#gramlen is the length of the biggest type of ngram we'll extract
#tokenized_captions should be the thing you'd pass into token embedding, have shape (N, n_tkn)
#sentinel embedding should have shape (tkn_dim,) and will be used as filler for any ngrams where n < gramlen
#selector_tokens should be shape (N, gramlen, tkn_dim)
#selector_logit_scale should be a scalar, and already exponentiated if needed
#output will be shape (N, gramlen, tkn_dim)
def caption_to_classname(caption_tokens, gramlen, tokenized_captions, sentinel_embedding, selector_tokens, selector_logit_scale):

    #check shapes
    assert(tokenized_captions.shape == caption_tokens.shape[:-1])
    assert(len(caption_tokens.shape) == 3)
    assert(len(selector_tokens.shape) == 3)
    assert(len(selector_logit_scale.shape) == 0)
    assert(caption_tokens.shape[0] == selector_tokens.shape[0])
    assert(caption_tokens.shape[2] == selector_tokens.shape[2])
    assert(selector_tokens.shape[1] == gramlen)

    #process caption so the ngrams are "stacked" as needed
    #mask will make sure we don't count certain invalid ngrams, by adding -inf to their logits
    processed_caption_tokens, mask = process_caption(caption_tokens, gramlen, tokenized_captions, sentinel_embedding)
    mask = torch.tensor([0, -torch.inf]).cuda()[mask]

    #compute cossims
    processed_caption_tokens_norm = processed_caption_tokens / torch.clamp(processed_caption_tokens.norm(dim=-1, keepdim=True), min=1e-6)
    selector_tokens_norm = selector_tokens / torch.clamp(selector_tokens.norm(dim=-1, keepdim=True), min=1e-6)
    cossims = torch.sum(processed_caption_tokens_norm * selector_tokens_norm.unsqueeze(1), dim=-1) #(N, *, gramlen)
    cossims = torch.mean(cossims, dim=-1) #(N, *)
    assert(len(cossims.shape) == 2)
    assert(cossims.shape[0] == caption_tokens.shape[0])
    assert(cossims.shape[1] ==processed_caption_tokens.shape[1])

    #compute probs
    logits = selector_logit_scale * cossims
    logits = logits + mask #keeps irrelevant tokens from affecting things by setting them to -inf, which softmax indeed can handle (I checked)
    probs = F.softmax(logits, dim=-1)

    #now do weighted sum
    probs = probs.unsqueeze(-1).unsqueeze(-1) #(N, *, 1, 1)
    out_tokens = torch.sum(probs * processed_caption_tokens, dim=1) #(N, gramlen, tkn_dim)

    #check and return
    assert(out_tokens.shape == selector_tokens.shape)
    return out_tokens
