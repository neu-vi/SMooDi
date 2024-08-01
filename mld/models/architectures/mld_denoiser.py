import torch
import torch.nn as nn
from mld.models.architectures.tools.embeddings import (TimestepEmbedding,
                                                       Timesteps)
from mld.models.operator import PositionalEncoding
from mld.models.operator.cross_attention import (SkipTransformerEncoder,
                                                 TransformerDecoder,
                                                 TransformerEncoderTemporalLayer,
                                                 TransformerDecoderLayer,
                                                 TransformerEncoder,
                                                 TransformerEncoderLayer)

from mld.models.operator.position_encoding import build_position_encoding
from mld.utils.temos_utils import lengths_to_mask
import torch.nn.functional as F
from torch import Tensor, nn
from typing import List, Optional, Union
from torch.distributions.distribution import Distribution
from mld.models.architectures.mld_style_encoder import StyleClassification
import numpy as np
from diffusers.models.attention import Attention as CrossAttention
from functools import partial
    
def shuffle_segments(tensor, segment_length,noise_probability=0.2,mask_probability=0.4, mask_ratio=0.2):
    if tensor.dim() != 3:
        raise ValueError("Input tensor must be 3-dimensional")

    C, T, H = tensor.size()
    
    num_segments = T // segment_length

    shuffled_tensor = tensor.clone()

    for i in range(num_segments):
        start = i * segment_length
        end = start + segment_length
        shuffled_indices = torch.randperm(segment_length) + start
        shuffled_tensor[:, start:end, :] = tensor[:, shuffled_indices, :]

    return shuffled_tensor

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class MldDenoiser(nn.Module):

    def __init__(self,
                 ablation,
                 nfeats: int = 263,
                 condition: str = "text",
                 latent_dim: list = [1, 256],
                 ff_size: int = 1024,
                 num_layers: int = 6,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 flip_sin_to_cos: bool = True,
                 return_intermediate_dec: bool = False,
                 position_embedding: str = "learned",
                 arch: str = "trans_enc",
                 freq_shift: int = 0,
                 guidance_scale: float = 7.5,
                 guidance_uncondp: float = 0.1,
                 text_encoded_dim: int = 768,
                 nclasses: int = 10,
                 **kwargs) -> None:

        super().__init__()

        self.latent_dim = latent_dim[-1]
        self.text_encoded_dim = text_encoded_dim
        self.condition = condition
        self.abl_plus = False
        self.ablation_skip_connection = ablation.SKIP_CONNECT
        self.diffusion_only = ablation.VAE_TYPE == "no"
        self.arch = arch
        self.pe_type = ablation.DIFF_PE_TYPE
        
        if self.diffusion_only:
            # assert self.arch == "trans_enc", "only implement encoder for diffusion-only"
            self.pose_embd = nn.Linear(nfeats, self.latent_dim)
            self.pose_proj = nn.Linear(self.latent_dim, nfeats)

        # emb proj
        if self.condition in ["text", "text_uncond"]:
            # text condition
            # project time from text_encoded_dim to latent_dim
            self.time_proj = Timesteps(text_encoded_dim, flip_sin_to_cos,
                                       freq_shift)
            self.time_embedding = TimestepEmbedding(text_encoded_dim,
                                                    self.latent_dim)
            # project time+text to latent_dim
            if text_encoded_dim != self.latent_dim:
                # todo 10.24 debug why relu
                self.emb_proj = nn.Sequential(
                    nn.ReLU(), nn.Linear(text_encoded_dim, self.latent_dim))
     
        if self.pe_type == "mld":
            self.query_pos = build_position_encoding(
                self.latent_dim, position_embedding=position_embedding)
            self.mem_pos = build_position_encoding(
                self.latent_dim, position_embedding=position_embedding)
        else:
            raise ValueError("Not Support PE type")

        if self.arch == "trans_enc":
            if self.ablation_skip_connection:
                # use DETR transformer
                encoder_layer = TransformerEncoderLayer(
                    self.latent_dim,
                    num_heads,
                    ff_size,
                    dropout,
                    activation,
                    normalize_before,
                )
                encoder_norm = nn.LayerNorm(self.latent_dim)
                self.encoder = SkipTransformerEncoder(encoder_layer,
                                                      num_layers, encoder_norm)
            else:
                # use torch transformer
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation)
                self.encoder = nn.TransformerEncoder(encoder_layer,
                                                     num_layers=num_layers)
        

    def forward(self,
                sample,
                timestep,
                encoder_hidden_states,
                lengths=None,
                control=None,
                **kwargs):
        # 0.  dimension matching
        # sample [latent_dim[0], batch_size, latent_dim] <= [batch_size, latent_dim[0], latent_dim[1]]
        sample = sample.permute(1, 0, 2)

        # 0. check lengths for no vae (diffusion only)
        if lengths not in [None, []]:
            mask = lengths_to_mask(lengths, sample.device)

        # 1. time_embedding
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timestep.expand(sample.shape[1]).clone()
        time_emb = self.time_proj(timesteps)
        time_emb = time_emb.to(dtype=sample.dtype)
        # [1, bs, latent_dim] <= [bs, latent_dim]
        time_emb = self.time_embedding(time_emb).unsqueeze(0)

        # 2. condition + time embedding
        if self.condition in ["text", "text_uncond"]:
            # text_emb [seq_len, batch_size, text_encoded_dim] <= [batch_size, seq_len, text_encoded_dim]
            encoder_hidden_states = encoder_hidden_states.permute(1, 0, 2)
            text_emb = encoder_hidden_states  # [num_words, bs, latent_dim]
           
            if self.text_encoded_dim != self.latent_dim:
                # [1 or 2, bs, latent_dim] <= [1 or 2, bs, text_encoded_dim]
                text_emb_latent = self.emb_proj(text_emb)
            # else:
            #     text_emb_latent = text_emb
            if self.abl_plus:
                emb_latent = time_emb + text_emb_latent
            else:
                emb_latent = torch.cat((time_emb, text_emb_latent), 0)
     
        else:
            raise TypeError(f"condition type {self.condition} not supported")

        # 4. transformer
        if self.arch == "trans_enc":
            if self.diffusion_only:
                sample = self.pose_embd(sample)
                xseq = torch.cat((emb_latent, sample), axis=0)
            else:
                xseq = torch.cat((sample, emb_latent), axis=0)

            #     # todo change to query_pos_decoder
            xseq = self.query_pos(xseq)
            tokens = self.encoder(xseq,control=control)

            if self.diffusion_only:
                sample = tokens[emb_latent.shape[0]:]
                sample = self.pose_proj(sample)

                # zero for padded area
                sample[~mask.T] = 0
            else:
                sample = tokens[:sample.shape[0]]

        sample = sample.permute(1, 0, 2)

        return (sample, )

class ControlMldDenoiser(nn.Module):
    def __init__(self,
                 ablation,
                 nfeats: int = 263,
                 condition: str = "text",
                 latent_dim: list = [1, 256],
                 ff_size: int = 1024,
                 num_layers: int = 6,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 flip_sin_to_cos: bool = True,
                 return_intermediate_dec: bool = False,
                 position_embedding: str = "learned",
                 arch: str = "trans_enc",
                 freq_shift: int = 0,
                 guidance_scale: float = 7.5,
                 guidance_uncondp: float = 0.1,
                 text_encoded_dim: int = 768,
                 nclasses: int = 10,
                 **kwargs) -> None:

        super().__init__()
        input_feats = nfeats
        self.latent_dim = latent_dim[-1]
        self.text_encoded_dim = text_encoded_dim
        self.condition = condition
        self.abl_plus = False
        self.ablation_skip_connection = ablation.SKIP_CONNECT
        self.diffusion_only = ablation.VAE_TYPE == "no"
        self.arch = arch
        self.pe_type = ablation.DIFF_PE_TYPE
        self.latent_size = latent_dim[0]
        self.is_adain = ablation.IS_ADAIN
        self.alpha = 1.0
        self.is_test = ablation.TEST
        self.is_test_walk = ablation.TEST_WALK

        self.is_style_text = ablation.IS_STYLE_TEXT
        
        #MLD
        if text_encoded_dim != self.latent_dim:
                self.emb_proj = nn.Sequential(nn.ReLU(), nn.Linear(text_encoded_dim, self.latent_dim))

        self.query_pos = build_position_encoding(
            self.latent_dim, position_embedding=position_embedding)
        self.mem_pos = build_position_encoding(
            self.latent_dim, position_embedding=position_embedding)
        
        self.mld_denoiser = MldDenoiser(ablation,
                 nfeats,
                 condition,
                 latent_dim,
                 ff_size,
                 num_layers,
                 num_heads,
                 dropout,
                 normalize_before,
                 activation,
                 flip_sin_to_cos,
                 return_intermediate_dec,
                 position_embedding,
                 arch,
                 freq_shift,
                 guidance_scale,
                 guidance_uncondp,
                 text_encoded_dim,
                 nclasses,
                 **kwargs)
        
        self.style_encoder = StyleClassification(nclasses=100,use_temporal_atten=False)
        #CMLD
        self.skel_embedding = nn.Linear(input_feats, self.latent_dim)
        self.global_motion_token = nn.Parameter(
                torch.randn(self.latent_size * 2, self.latent_dim))

        self.time_proj_c = Timesteps(text_encoded_dim, flip_sin_to_cos,freq_shift)
        self.time_embedding_c = TimestepEmbedding(text_encoded_dim,self.latent_dim)
        num_block = (num_layers-1)//2
        self.zero_convs = zero_module(nn.ModuleList([nn.Linear(self.latent_dim, self.latent_dim) for _ in range(num_block)]))

        encoder_norm_c = nn.LayerNorm(self.latent_dim)

        encoder_layer_c = TransformerEncoderLayer(
            self.latent_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
        )

        self.encoder_c = SkipTransformerEncoder(encoder_layer_c,num_layers, encoder_norm_c,return_intermediate=True)
    
    def freeze_mld_parameters(self):
        print("Freeze the parameters of MLD")
        for param in self.mld_denoiser.parameters():
            param.requires_grad = False
        
        self.mld_denoiser.eval()

    def freeze_style_encoder_parameters(self):
        print("Freeze the parameters of style_encoder")
        for param in self.style_encoder.parameters():
            param.requires_grad = False
        
        self.style_encoder.eval()

    def cmld_forward(self,
                sample,
                timestep,
                encoder_hidden_states,
                lengths=None,
                reference=None,
                style_text_emb=None,
                **kwargs):
        
        sample = sample.permute(1, 0, 2)
        
        style_emb = self.style_encode(reference)

        # 0. check lengths for no vae (diffusion only)
        if lengths not in [None, []]:
            mask = lengths_to_mask(lengths, sample.device)
        timesteps = timestep.expand(sample.shape[1]).clone()
        time_emb = self.time_proj_c(timesteps)
        time_emb = time_emb.to(dtype=sample.dtype)
        # [1, bs, latent_dim] <= [bs, latent_dim]
        time_emb = self.time_embedding_c(time_emb).unsqueeze(0)

        # 2. condition + time embedding
        # text_emb [seq_len, batch_size, text_encoded_dim] <= [batch_size, seq_len, text_encoded_dim]
        encoder_hidden_states = encoder_hidden_states.permute(1, 0, 2)
        text_emb = encoder_hidden_states  # [num_words, bs, latent_dim]
        # textembedding projection
        if self.text_encoded_dim != self.latent_dim:
                # [1 or 2, bs, latent_dim] <= [1 or 2, bs, text_encoded_dim]
            text_emb_latent = self.emb_proj(text_emb)
        else:
            text_emb_latent = text_emb

        if self.abl_plus:
            emb_latent = time_emb + text_emb_latent
        else:
            emb_latent = torch.cat((time_emb, text_emb_latent), 0)

        sample += self.alpha * style_emb

        xseq = torch.cat((sample, emb_latent), axis=0)
        xseq = self.query_pos(xseq)
        output = self.encoder_c(xseq)

        control = []
        for i,module in enumerate(self.zero_convs):
            control.append(module(output[i]))
        
        control = torch.stack(control)

        return control
    
    def forward(self,
                sample,
                timestep,
                encoder_hidden_states,
                lengths=None,
                reference=None,
                style_text_emb=None,
                **kwargs):
        
        control = None

        if reference is not None or style_text_emb is not None:
            if reference is not None:
                control = self.cmld_forward(sample,
                    timestep,
                    encoder_hidden_states,
                    lengths,
                    reference=reference,
                    **kwargs)
           
        output = self.mld_denoiser(sample,
                    timestep,
                    encoder_hidden_states,
                    lengths,
                    control,
                    **kwargs)

        return output
    
    def style_encode(
            self,
            reference: Tensor,
            lengths: Optional[List[int]] = None
    ) -> Union[Tensor, Distribution]:        
        reference = shuffle_segments(reference,16)        
        output = self.style_encoder(reference,lengths,stage="Encode")

        return output