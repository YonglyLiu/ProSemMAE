import torch
import torch.nn as nn
import numpy as np
# from timm.models.layers.helpers import to_3tuple
from timm.layers import to_3tuple
#from mamba_ssm import Mamba
import torch.nn.functional as F


__all__ = ["ProSemMAE"]

def build_3d_sincos_position_embedding(grid_size, embed_dim, num_tokens=1, temperature=10000.):
    grid_size = to_3tuple(grid_size)
    h, w, d = grid_size
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_d = torch.arange(d, dtype=torch.float32)

    grid_h, grid_w, grid_d = torch.meshgrid(grid_h, grid_w, grid_d)
    assert embed_dim % 6 == 0, 'Embed dimension must be divisible by 6 for 3D sin-cos position embedding'
    pos_dim = embed_dim // 6
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature ** omega)
    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    out_d = torch.einsum('m,d->md', [grid_d.flatten(), omega])
    pos_emb = torch.cat(
        [torch.sin(out_h), torch.cos(out_h), torch.sin(out_w), torch.cos(out_w), torch.sin(out_d), torch.cos(out_d)],
        dim=1)[None, :, :]

    assert num_tokens == 1 or num_tokens == 0, "Number of tokens must be of 0 or 1"
    if num_tokens == 1:
        pe_token = torch.zeros([1, 1, embed_dim], dtype=torch.float32)
        pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
    else:
        pos_embed = nn.Parameter(pos_emb)
    pos_embed.requires_grad = False
    return pos_embed


def build_perceptron_position_embedding(grid_size, embed_dim, num_tokens=1):
    pos_emb = torch.rand([1, np.prod(grid_size), embed_dim])
    nn.init.normal_(pos_emb, std=.02)

    assert num_tokens == 1 or num_tokens == 0, "Number of tokens must be of 0 or 1"
    if num_tokens == 1:
        pe_token = torch.zeros([1, 1, embed_dim], dtype=torch.float32)
        pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
    else:
        pos_embed = nn.Parameter(pos_emb)
    return pos_embed


def patchify_image(x, patch_size):
    """
    ATTENTION!!!!!!!
    Different from 2D version patchification: The final axis follows the order of [ph, pw, pd, c] instead of [c, ph, pw, pd]
    """
    # patchify input, [B,C,H,W,D] --> [B,C,gh,ph,gw,pw,gd,pd] --> [B,gh*gw*gd,ph*pw*pd*C]
    B, C, H, W, D = x.shape
    patch_size = to_3tuple(patch_size)
    grid_size = (H // patch_size[0], W // patch_size[1], D // patch_size[2])

    x = x.reshape(B, C, grid_size[0], patch_size[0], grid_size[1], patch_size[1], grid_size[2],
                  patch_size[2])  # [B,C,gh,ph,gw,pw,gd,pd]
    x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(B, np.prod(grid_size),
                                                  np.prod(patch_size) * C)  # [B,gh*gw*gd,ph*pw*pd*C]

    return x


def batched_shuffle_indices(batch_size, length, device):
    """
    Generate random permutations of specified length for batch_size times
    Motivated by https://discuss.pytorch.org/t/batched-shuffling-of-feature-vectors/30188/4
    """
    rand = torch.rand(batch_size, length).to(device)
    batch_perm = rand.argsort(dim=1)
    return batch_perm
class Block(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = h + x

        h = x
        x = self.norm2(x)
        x = self.mlp(x)
        return h + x
class MAEDecoder(nn.Module):
    def __init__(self, embed_dim, depth, num_heads, patch_dim):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, patch_dim)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x)
class PatchEmbed3D(nn.Module):
    """ 3D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, in_chan_last=True):
        super().__init__()
        img_size = to_3tuple(img_size)
        patch_size = to_3tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = []
        for im_size, pa_size in zip(img_size, patch_size):
            self.grid_size.append(im_size // pa_size)
        # self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.in_chans = in_chans
        self.num_patches = np.prod(self.grid_size)
        self.flatten = flatten
        self.in_chan_last = in_chan_last

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # import pdb
        # pdb.set_trace()
        B, L, S = x.shape
        assert S == np.prod(self.img_size) * self.in_chans, \
            f"Input image total size {S} doesn't match model configuration"
        if self.in_chan_last:
            x = x.reshape(B * L, *self.img_size, self.in_chans).permute(0, 4, 1, 2, 3) # When patchification follows HWDC
        else:
            x = x.reshape(B * L, self.in_chans, *self.img_size)
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHWD -> BNC
        x = self.norm(x)
        return x


class Semantic_Prototype_Bank(nn.Module):
    """
    Semantic_Prototype_bank: learnable prototypes + attention from layer tokens to prototypes.
    - Input: x [B, N, D]
    - Output: y [B, N, D]
    """
    def __init__(self, num_prototypes, embed_dim, dropout=0.0, temperature=1.0):
        super().__init__()
        self.num_concepts = num_prototypes
        self.embed_dim = embed_dim
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, embed_dim) * 0.02)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.log_scale = nn.Parameter(torch.tensor(float(1.0 / (temperature + 1e-8))))

    def forward(self, x):
        # x: [B, N, D]
        B, N, D = x.shape
        # Q: [B, N, D]
        Q = self.q_proj(x)  # queries from tokens
        # K, V from prototypes: [K, D] -> expand to [B, K, D] when needed
        K = self.k_proj(self.prototypes)  # [K, D]
        V = self.v_proj(self.prototypes)  # [K, D]

        # compute attention: Q @ K^T -> [B, N, K]
        scale = (D ** -0.5) * torch.exp(self.log_scale)
        attn_logits = torch.matmul(Q, K.transpose(0, 1)) * scale  # [B, N, K]
        attn = torch.softmax(attn_logits, dim=-1)  # [B, N, K]
        attn = self.dropout(attn)

        # aggregate V by attn -> [B, N, D]
        y = torch.matmul(attn, V)  # [B, N, D]
        y = self.out_proj(y)  # [B, N, D]
        return y

class ProSemMAE(nn.Module):

    def __init__(self,
                 encoder,
                 decoder,
                 args):
        super().__init__()
        self.args = args
        input_size = to_3tuple(args.img_size)
        patch_size = to_3tuple(args.patch_size)
        self.input_size = input_size
        self.patch_size = patch_size

        out_chans = args.in_chans * np.prod(self.patch_size)
        self.out_chans = out_chans

        grid_size = []
        for in_size, pa_size in zip(input_size, patch_size):
            assert in_size % pa_size == 0, "input size and patch size are not proper"
            grid_size.append(in_size // pa_size)
        self.grid_size = grid_size

        # build encoder and decoder
        embed_layer = PatchEmbed3D
        self.encoder = encoder(patch_size=patch_size, in_chans=args.in_chans, embed_layer=embed_layer)
        self.decoder = decoder(patch_size=patch_size, num_classes=out_chans, self_attn=True)

        self.encoder_to_decoder = nn.ModuleList(
            [nn.Linear(self.encoder.embed_dim, self.decoder.embed_dim, bias=True) for i in range(4)])

        num_prototypes = getattr(args, "num_prototypes", 64)
        sp_dropout = getattr(args, "prototype_dropout", 0.0)
        sp_temp = getattr(args, "prototype_temperature", 1.0)
        self.SemProBank = Semantic_Prototype_Bank(num_prototypes=num_prototypes, embed_dim=self.encoder.embed_dim, dropout=sp_dropout,temperature=sp_temp)



        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder.embed_dim))

        self.patch_norm = nn.LayerNorm(normalized_shape=(out_chans,), eps=1e-6, elementwise_affine=False)
        self.norm = nn.LayerNorm(self.encoder.embed_dim)
        self.criterion = nn.MSELoss()

        # build positional encoding for encoder and decoder
        if args.pos_embed_type == 'sincos':
            with torch.no_grad():
                self.encoder_pos_embed = build_3d_sincos_position_embedding(grid_size,
                                                                            self.encoder.embed_dim,
                                                                            num_tokens=0)
                self.decoder_pos_embed = build_3d_sincos_position_embedding(grid_size,
                                                                            self.decoder.embed_dim,
                                                                            num_tokens=0)
        elif args.pos_embed_type == 'perceptron':
            self.encoder_pos_embed = build_perceptron_position_embedding(grid_size,
                                                                         self.encoder.embed_dim,
                                                                         num_tokens=0)
            with torch.no_grad():
                self.decoder_pos_embed = build_3d_sincos_position_embedding(grid_size,
                                                                            self.decoder.embed_dim,
                                                                            num_tokens=0)
        # initialize encoder_to_decoder and mask token
        nn.init.normal_(self.mask_token, std=.02)
        for i in range(4):
            nn.init.xavier_uniform_(self.encoder_to_decoder[i].weight)

    def forward(self, x, return_image=False):
        args = self.args
        batch_size = x.size(0)
        in_chans = x.size(1)
        assert in_chans == args.in_chans
        out_chans = self.out_chans
        x = patchify_image(x, self.patch_size)  # [B,gh*gw*gd,ph*pw*pd*C]

        # compute length for selected and masked
        length = np.prod(self.grid_size)
        sel_length = int(length * (1 - args.mask_ratio))
        msk_length = length - sel_length

        # generate batched shuffle indices
        shuffle_indices = batched_shuffle_indices(batch_size, length, device=x.device)
        unshuffle_indices = shuffle_indices.argsort(dim=1)

        # select and mask the input patches
        shuffled_x = x.gather(dim=1, index=shuffle_indices[:, :, None].expand(-1, -1, out_chans))
        sel_x = shuffled_x[:, :sel_length, :]
        msk_x = shuffled_x[:, -msk_length:, :]
        # select and mask the indices
        # shuffle_indices = F.pad(shuffle_indices + 1, pad=(1, 0), mode='constant', value=0)
        sel_indices = shuffle_indices[:, :sel_length]
        # msk_indices = shuffle_indices[:, -msk_length:]

        # select the position embedings accordingly
        sel_encoder_pos_embed = self.encoder_pos_embed.expand(batch_size, -1, -1) \
            .gather(dim=1, index=sel_indices[:, :, None].expand(-1, -1, self.encoder.embed_dim))

        # forward encoder & proj to decoder dimension
        sel_x, hidden_states_out = self.encoder(sel_x, sel_encoder_pos_embed)

        sel_dic = {}
        for i, index in enumerate([3, 6, 9, 12]):
            # choose xi: for last layer you used sel_x, else hidden_states_out[index]
            if index == 12:
                xi = sel_x  # [B, sel_length, D_enc]
            else:
                xi = hidden_states_out[index]  # [B, N_i, D_enc]

            # 1) ConceptBank enhancement (yi = CB(xi))
                yi = self.SemProBank(xi)  # [B, N_i, D_enc]
                #yi = self.conceptbank_Mamba(xi)
            # 2) residual fusion: enhanced feature
            xi_enh = self.norm(xi + yi)  # [B, N_i, D_enc]

            # 3) project to decoder embed dim
            # projected = self.encoder_to_decoder[i](xi_enh)  # [B, N_i, D_dec]
            projected = self.encoder_to_decoder[i](yi)

            if index == 12:
                # for the last (deepest) layer we need to append mask tokens to form full sequence
                all_x = torch.cat([projected, self.mask_token.expand(batch_size, msk_length, -1)],
                                  dim=1)  # [B, sel_len+msk_len, D_dec]
                # apply shuffled decoder pos embed (keep your original pattern)
                shuffled_decoder_pos_embed = self.decoder_pos_embed.expand(batch_size, -1, -1) \
                    .gather(dim=1, index=shuffle_indices[:, :, None].expand(-1, -1, self.decoder.embed_dim))
                all_x[:, 1:, :] += shuffled_decoder_pos_embed
            else:
                all_x = projected
            sel_dic[index] = all_x
        # forward decoder
        all_x = self.decoder(sel_dic)

        # loss
        loss = self.criterion(input=all_x[:, -msk_length:, :], target=self.patch_norm(msk_x.detach()))

        # if return_image:
        #     # unshuffled all the tokens
        #     masked_x = torch.cat(
        #         [shuffled_x[:, :sel_length, :], 0. * torch.ones(batch_size, msk_length, out_chans).to(x.device)],
        #         dim=1).gather(dim=1, index=unshuffle_indices[:, :, None].expand(-1, -1, out_chans))
        #     recon = all_x[:, 1:, :].gather(dim=1, index=unshuffle_indices[:, :, None].expand(-1, -1, out_chans))
        #     recon = recon * (x.var(dim=-1, unbiased=True, keepdim=True).sqrt() + 1e-6) + x.mean(dim=-1, keepdim=True)
        #     return loss, x.detach(), recon.detach(), masked_x.detach()
        # else:
        #     return loss
        if return_image:
            # Step 1: get original patches
            orig_patches = x  # [B, N, P*C]

            # Step 2: get model's reconstruction for ALL patches
            full_recon_patches = all_x[:, 1:, :]  # [B, N, P*C]
            full_recon_patches = full_recon_patches.gather(
                dim=1, index=unshuffle_indices[:, :, None].expand(-1, -1, out_chans)
            )

            # Step 3: denormalize
            full_recon_patches = full_recon_patches * (orig_patches.var(dim=-1, keepdim=True).sqrt() + 1e-6) \
                                 + orig_patches.mean(dim=-1, keepdim=True)

            # Step 4: create "masked input" for visualization: visible=original, masked=0
            mask_indicator = torch.zeros(batch_size, length, device=x.device)
            mask_indicator.scatter_(1, shuffle_indices[:, -msk_length:], 1)  # 1 for masked
            mask_indicator = mask_indicator.gather(1, unshuffle_indices)  # unshuffled mask

            masked_input = orig_patches.clone()
            masked_input[mask_indicator.bool()] = 0.0

            return loss, orig_patches.detach(), full_recon_patches.detach(), masked_input.detach()
        else:
            return loss


class HiEndMAE(nn.Module):

    def __init__(self,
                 encoder,
                 decoder,
                 args):
        super().__init__()
        self.args = args
        input_size = to_3tuple(args.img_size)
        patch_size = to_3tuple(args.patch_size)
        self.input_size = input_size
        self.patch_size = patch_size

        out_chans = args.in_chans * np.prod(self.patch_size)
        self.out_chans = out_chans

        grid_size = []
        for in_size, pa_size in zip(input_size, patch_size):
            assert in_size % pa_size == 0, "input size and patch size are not proper"
            grid_size.append(in_size // pa_size)
        self.grid_size = grid_size

        # build encoder and decoder
        embed_layer = PatchEmbed3D
        self.encoder = encoder(patch_size=patch_size, in_chans=args.in_chans, embed_layer=embed_layer)
        self.decoder = decoder(patch_size=patch_size, num_classes=out_chans, self_attn=True)

        self.encoder_to_decoder = nn.ModuleList(
            [nn.Linear(self.encoder.embed_dim, self.decoder.embed_dim, bias=True) for i in range(4)])

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder.embed_dim))

        self.patch_norm = nn.LayerNorm(normalized_shape=(out_chans,), eps=1e-6, elementwise_affine=False)

        self.criterion = nn.MSELoss()

        # build positional encoding for encoder and decoder
        if args.pos_embed_type == 'sincos':
            with torch.no_grad():
                self.encoder_pos_embed = build_3d_sincos_position_embedding(grid_size,
                                                                            self.encoder.embed_dim,
                                                                            num_tokens=0)
                self.decoder_pos_embed = build_3d_sincos_position_embedding(grid_size,
                                                                            self.decoder.embed_dim,
                                                                            num_tokens=0)
        elif args.pos_embed_type == 'perceptron':
            self.encoder_pos_embed = build_perceptron_position_embedding(grid_size,
                                                                         self.encoder.embed_dim,
                                                                         num_tokens=0)
            with torch.no_grad():
                self.decoder_pos_embed = build_3d_sincos_position_embedding(grid_size,
                                                                            self.decoder.embed_dim,
                                                                            num_tokens=0)
        # initialize encoder_to_decoder and mask token
        nn.init.normal_(self.mask_token, std=.02)
        for i in range(4):
            nn.init.xavier_uniform_(self.encoder_to_decoder[i].weight)

    def forward(self, x, return_image=False):
        args = self.args
        batch_size = x.size(0)
        in_chans = x.size(1)
        assert in_chans == args.in_chans
        out_chans = self.out_chans
        x = patchify_image(x, self.patch_size)  # [B,gh*gw*gd,ph*pw*pd*C]

        # compute length for selected and masked
        length = np.prod(self.grid_size)
        sel_length = int(length * (1 - args.mask_ratio))
        msk_length = length - sel_length

        # generate batched shuffle indices
        shuffle_indices = batched_shuffle_indices(batch_size, length, device=x.device)
        unshuffle_indices = shuffle_indices.argsort(dim=1)

        # select and mask the input patches
        shuffled_x = x.gather(dim=1, index=shuffle_indices[:, :, None].expand(-1, -1, out_chans))
        sel_x = shuffled_x[:, :sel_length, :]
        msk_x = shuffled_x[:, -msk_length:, :]
        # select and mask the indices
        # shuffle_indices = F.pad(shuffle_indices + 1, pad=(1, 0), mode='constant', value=0)
        sel_indices = shuffle_indices[:, :sel_length]
        # msk_indices = shuffle_indices[:, -msk_length:]

        # select the position embedings accordingly
        sel_encoder_pos_embed = self.encoder_pos_embed.expand(batch_size, -1, -1) \
            .gather(dim=1, index=sel_indices[:, :, None].expand(-1, -1, self.encoder.embed_dim))

        # forward encoder & proj to decoder dimension
        sel_x, hidden_states_out = self.encoder(sel_x, sel_encoder_pos_embed)

        sel_dic = {}
        for i, index in enumerate([3, 6, 9, 12]):
            if index == 12:
                all_x = torch.cat(
                    [self.encoder_to_decoder[i](sel_x), self.mask_token.expand(batch_size, msk_length, -1)], dim=1)
                shuffled_decoder_pos_embed = self.decoder_pos_embed.expand(batch_size, -1, -1) \
                    .gather(dim=1, index=shuffle_indices[:, :, None].expand(-1, -1, self.decoder.embed_dim))
                all_x[:, 1:, :] += shuffled_decoder_pos_embed
            else:
                all_x = self.encoder_to_decoder[i](hidden_states_out[index])

            # combine the selected tokens and mask tokens in the shuffled order

            # shuffle all the decoder positional encoding

            # add the shuffled positional embedings to encoder output tokens
            sel_dic[index] = all_x
        # forward decoder
        all_x = self.decoder(sel_dic)

        # loss
        loss = self.criterion(input=all_x[:, -msk_length:, :], target=self.patch_norm(msk_x.detach()))

        if return_image:
            # Step 1: get original patches
            orig_patches = x  # [B, N, P*C]

            # Step 2: get model's reconstruction for ALL patches
            full_recon_patches = all_x[:, 1:, :]  # [B, N, P*C]
            full_recon_patches = full_recon_patches.gather(
                dim=1, index=unshuffle_indices[:, :, None].expand(-1, -1, out_chans)
            )

            # Step 3: denormalize
            full_recon_patches = full_recon_patches * (orig_patches.var(dim=-1, keepdim=True).sqrt() + 1e-6) \
                                 + orig_patches.mean(dim=-1, keepdim=True)

            # Step 4: create "masked input" for visualization: visible=original, masked=0
            mask_indicator = torch.zeros(batch_size, length, device=x.device)
            mask_indicator.scatter_(1, shuffle_indices[:, -msk_length:], 1)  # 1 for masked
            mask_indicator = mask_indicator.gather(1, unshuffle_indices)  # unshuffled mask

            masked_input = orig_patches.clone()
            masked_input[mask_indicator.bool()] = 0.0

            return loss, orig_patches.detach(), full_recon_patches.detach(), masked_input.detach()
        else:
            return loss


class MAE(nn.Module):

    def __init__(self, encoder, args):
        super().__init__()
        self.args = args

        # -------------------------
        # basic settings
        # -------------------------
        self.input_size = to_3tuple(args.img_size)
        self.patch_size = to_3tuple(args.patch_size)
        self.in_chans = args.in_chans

        self.patch_dim = self.in_chans * np.prod(self.patch_size)

        self.grid_size = [
            i // p for i, p in zip(self.input_size, self.patch_size)
        ]
        self.num_patches = np.prod(self.grid_size)

        # -------------------------
        # encoder
        # -------------------------
        self.encoder = encoder(
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_layer=PatchEmbed3D,
        )

        enc_dim = self.encoder.embed_dim

        # -------------------------
        # decoder (lightweight)
        # -------------------------
        dec_dim = getattr(args, "decoder_embed_dim", 384)
        dec_depth = getattr(args, "decoder_depth", 2)
        dec_heads = getattr(args, "decoder_num_heads", 8)

        self.encoder_to_decoder = nn.Linear(enc_dim, dec_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))

        self.decoder = MAEDecoder(
            embed_dim=dec_dim,
            depth=dec_depth,
            num_heads=dec_heads,
            patch_dim=self.patch_dim,
        )

        # -------------------------
        # position embedding
        # -------------------------
        self.encoder_pos_embed = build_3d_sincos_position_embedding(
            self.grid_size, enc_dim, num_tokens=0
        )
        self.decoder_pos_embed = build_3d_sincos_position_embedding(
            self.grid_size, dec_dim, num_tokens=0
        )

        # -------------------------
        # loss
        # -------------------------
        self.patch_norm = nn.LayerNorm(self.patch_dim, eps=1e-6, elementwise_affine=False)
        self.criterion = nn.MSELoss()

        nn.init.normal_(self.mask_token, std=0.02)

    # --------------------------------------------------
    # forward
    # --------------------------------------------------
    def forward(self, x, return_image=False):
        B = x.size(0)

        # 1. patchify
        x_patch = patchify_image(x, self.patch_size)  # [B, L, patch_dim]
        L = x_patch.size(1)

        # 2. random masking
        len_keep = int(L * (1 - self.args.mask_ratio))
        noise = torch.rand(B, L, device=x.device)

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_keep = x_patch.gather(
            1, ids_keep.unsqueeze(-1).expand(-1, -1, x_patch.size(-1))
        )

        mask = torch.ones(B, L, device=x.device)
        mask[:, :len_keep] = 0
        mask = mask.gather(1, ids_restore)  # restore 原始顺序

        # 3. encoder
        pos_enc = self.encoder_pos_embed.expand(B, -1, -1)
        pos_keep = pos_enc.gather(
            1, ids_keep.unsqueeze(-1).expand(-1, -1, pos_enc.size(-1))
        )

        latent, _ = self.encoder(x_keep, pos_keep)

        # 4. project to decoder dim
        latent = self.encoder_to_decoder(latent)

        # 5. append mask tokens
        mask_tokens = self.mask_token.expand(B, L - len_keep, -1)
        dec_tokens = torch.cat([latent, mask_tokens], dim=1)

        # restore order
        dec_tokens = dec_tokens.gather(
            1, ids_restore.unsqueeze(-1).expand(-1, -1, dec_tokens.size(-1))
        )

        # add decoder pos embed
        dec_tokens = dec_tokens + self.decoder_pos_embed

        # 6. decoder
        pred = self.decoder(dec_tokens)

        # 7. loss (masked only)
        target = self.patch_norm(x_patch)
        loss = self.criterion(
            pred[:, len_keep:, :],
            target[:, len_keep:, :]
        )

        if return_image:
            recon = pred
            masked_patch = x_patch.clone()
            masked_patch[mask.bool()] = 0.0
            return loss, recon.detach()
        else:
            return loss

        return loss





def build_vit_large_prosemmae_p12_3d(args):
    from networks.vit import vit_large_patch12_96, decoder_large_patch12_96
    model = ProSemMAE(args=args, encoder=vit_large_patch12_96, decoder=decoder_large_patch12_96)
    print(model)
    return model


def build_vit_base_prosemmae_p16_3d(args):
    from networks.vit import vit_base_patch16_96, decoder_base_patch16_96
    model = ProSemMAE(args=args, encoder=vit_base_patch16_96, decoder=decoder_base_patch16_96)
    print(model)
    return model

def build_vit_large_prosemmae_p16_3d(args):
    from networks.vit import vit_large_patch16_96, decoder_large_patch16_96
    model = ProSemMAE(args=args, encoder=vit_large_patch16_96, decoder=decoder_large_patch16_96)
    print(model)
    return model


def build_vit_large_mae_p16_3d(args):
    from networks.vit import vit_large_patch16_96, decoder_large_patch16_96
    model = MAE(args=args, encoder=vit_large_patch16_96)
    print(model)
    return model

def build_vit_large_hiendmae_p12_3d(args):
    from networks.vit import vit_large_patch12_96, decoder_large_patch12_96
    model = HiEndMAE(args=args, encoder=vit_large_patch12_96, decoder=decoder_large_patch12_96)
    print(model)
    return model

def build_vit_large_hiendmae_p16_3d(args):
    from networks.vit import vit_large_patch16_96, decoder_large_patch16_96
    model = HiEndMAE(args=args, encoder=vit_large_patch16_96, decoder=decoder_large_patch16_96)
    print(model)
    return model