import os
import sys
import torch
import torch.nn as nn

class ExposedResidualAttentionBlock(nn.Module):
    def __init__(self, clip_resblock):
        super().__init__()
        self.attn = clip_resblock.attn
        self.ln_1 = clip_resblock.ln_1
        self.mlp = clip_resblock.mlp
        self.ln_2 = clip_resblock.ln_2
        self.attn_mask = clip_resblock.attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask, average_attn_weights=True)

    def forward(self, x: torch.Tensor):
        meow, attn_probs = self.attention(self.ln_1(x))
        x = x + meow
        x = x + self.mlp(self.ln_2(x))
        return x, attn_probs.float()

class ExposedTransformer(nn.Module):
    def __init__(self, clip_transformer):
        super().__init__()
        self.width = clip_transformer.width
        self.layers = clip_transformer.layers #this is int
        clip_resblocks_as_list = [h for h in clip_transformer.resblocks.children()]
        self.resblocks_as_list = nn.ModuleList([ExposedResidualAttentionBlock(h) for h in clip_resblocks_as_list])

    #attn_probs will be of shape (N, T, L) where T is length of target sequence and L is length of input sequence
    #want to get shape (N, L)
    def __extract_cls_attn_probs(self, attn_probs, cls_indices=None):
        if cls_indices is not None:
            return attn_probs[torch.arange(attn_probs.shape[0]), cls_indices, :]
        else:
            return attn_probs[:,0,:]

    #each item of the list will be of shape (N, L)
    #want to get shape (N, L)
    def __average_attn_probs(self, attn_probs_list):
        return torch.mean(torch.stack(attn_probs_list), dim=0)

    #will return x, attn_probs
    #where attn_probs is only for the CLS token and has been averaged across layers
    #should be shape (N, L) where L is length of input sequence
    def forward(self, x: torch.Tensor, cls_indices=None):
        attn_probs_list = []
        for h in self.resblocks_as_list:
            x, attn_probs = h(x)
            attn_probs = self.__extract_cls_attn_probs(attn_probs, cls_indices=cls_indices)
            attn_probs_list.append(attn_probs)

        attn_probs = self.__average_attn_probs(attn_probs_list)
        return x, attn_probs

class ExposedTextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = ExposedTransformer(clip_model.transformer)
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    #will return x, attn_probs
    #where attn_probs is only for the CLS token and has been averaged across layers
    #should be shape (N, L) where L is length of input sequence
    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        #NOTE: even though x (both input and output) are LND, nn.MultiheadAttention will *always* put the batch *first* for the attention weights!
        #So attn_probs is already good to go :)
        x, attn_probs = self.transformer(x, cls_indices=tokenized_prompts.argmax(dim=-1))
        
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x, attn_probs
