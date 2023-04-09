import os
import sys
import torch
import torch.nn as nn

EFFICIENT_CONTEXT_LENGTH = 10

def build_efficient_attn_mask():
    mask = torch.empty(EFFICIENT_CONTEXT_LENGTH, EFFICIENT_CONTEXT_LENGTH)
    mask.fill_(float("-inf"))
    mask.triu_(1)  # zero out the lower diagonal
    return mask

class EfficientResidualAttentionBlock(nn.Module):
    def __init__(self, clip_resblock, attn_mask):
        super().__init__()
        self.attn = clip_resblock.attn
        self.ln_1 = clip_resblock.ln_1
        self.mlp = clip_resblock.mlp
        self.ln_2 = clip_resblock.ln_2
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        meow = self.attention(self.ln_1(x))
        x = x + meow
        x = x + self.mlp(self.ln_2(x))
        return x

class EfficientTransformer(nn.Module):
    def __init__(self, clip_transformer, attn_mask):
        super().__init__()
        self.width = clip_transformer.width
        self.layers = clip_transformer.layers #this is int
        clip_resblocks_as_list = [h for h in clip_transformer.resblocks.children()]
        self.resblocks = nn.Sequential(*[EfficientResidualAttentionBlock(h, attn_mask) for h in clip_resblocks_as_list])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class EfficientTextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        attn_mask = build_efficient_attn_mask()
        self.transformer = EfficientTransformer(clip_model.transformer, attn_mask)
        self.positional_embedding = clip_model.positional_embedding
        self.context_length = EFFICIENT_CONTEXT_LENGTH
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        assert(prompts.shape[1] == self.context_length)
        x = prompts + self.positional_embedding.type(self.dtype)[:self.context_length,:]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
