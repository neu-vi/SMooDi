import inspect
import os
from mld.transforms.rotation2xyz import Rotation2xyz
import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torchmetrics import MetricCollection
import time
from mld.config import instantiate_from_config
from mld.data.humanml.scripts.motion_process import recover_root_rot_pos
from mld.data.humanml.common.quaternion import quaternion_to_cont6d
from collections import OrderedDict
import torch.nn as nn

from os.path import join as pjoin
from mld.models.architectures import (
    mld_denoiser,
    mld_vae,
    vposert_vae,
    t2m_motionenc,
    t2m_textenc,
    vposert_vae,
)
from mld.models.losses.mld import MLDLosses
from mld.models.modeltype.base import BaseModel
from mld.utils.temos_utils import remove_padding

from .base import BaseModel
from mld.models.architectures.mld_style_encoder import StyleClassification
import torch.nn.functional as F

from torch.optim.adam import Adam
from tqdm import tqdm
import yaml


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)

def retrieve_timesteps(
    scheduler,
    num_inference_steps=None,
    device=None,
    timesteps=None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.
    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used,
            `timesteps` must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def rescale_noise_cfg(
    noise_cfg: torch.FloatTensor, noise_pred_text, guidance_rescale=0.0
):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(
        dim=list(range(1, noise_pred_text.ndim)), keepdim=True
    )
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = (
        guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    )

    return noise_cfg


def print_grad(grad):
    print(grad)

def build_dict_from_txt(filename):
    result_dict = {}
    
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split(" ")
            if len(parts) >= 3:
                key = parts[2]
                value = parts[1].split("_")[0]
                result_dict[key] = value
                
    return result_dict

class MLD(BaseModel):
    """
    Stage 1 vae
    Stage 2 diffusion
    """

    def __init__(self, cfg, datamodule, **kwargs):
        super().__init__()

        self.cfg = cfg

        self.stage = cfg.TRAIN.STAGE
        self.condition = cfg.model.condition
        self.is_vae = cfg.model.vae
        self.is_cmld = cfg.model.cmld
        self.predict_epsilon = cfg.TRAIN.ABLATION.PREDICT_EPSILON
        self.nfeats = cfg.DATASET.NFEATS
        self.njoints = cfg.DATASET.NJOINTS
        self.debug = cfg.DEBUG
        self.latent_dim = cfg.model.latent_dim
        self.guidance_scale = cfg.model.guidance_scale
        self.guidance_scale_style = cfg.model.guidance_scale_style
        self.guidance_uncodp = cfg.model.guidance_uncondp
        self.guidance_uncondp_style = cfg.model.guidance_uncondp_style
        self.datamodule = datamodule
        self.is_test = cfg.model.is_test

        self.is_control = cfg.model.is_control
        self.is_guidance = cfg.model.is_guidance
        self.is_cycle = cfg.model.is_cycle
        self.is_recon = cfg.model.is_recon

        self.guidance_mode = cfg.model.guidance_mode

        if self.is_test:
            nclass = 47
<<<<<<< HEAD
            dict_path = "./datasets/100STYLE_name_dict_Filter.txt"
        else:
            nclass = 100
            dict_path = "./datasets/100STYLE_name_dict.txt"
=======
            dict_path = "./dataset/100STYLE_name_dict_Filter.txt"
        else:
            nclass = 100
            dict_path = "./dataset//100STYLE_name_dict.txt"
>>>>>>> 045ca9590646d12c0e3a4de1ddbe6f8e20e4262c

        self.label_to_motion = build_dict_from_txt(dict_path)

        if self.is_cycle == True:
            self.stage = "cycle_diffusion"#"cycle_diffusion"
        
        if self.is_test:
<<<<<<< HEAD
            self.mean = np.load("./datasets/Mean.npy")
            self.std = np.load("./datasets/Std.npy")
=======
            self.mean = np.load("/work/vig/zhonglei/stylized_motion/dataset_all/Mean.npy")
            self.std = np.load("/work/vig/zhonglei/stylized_motion/dataset_all/Std.npy")
>>>>>>> 045ca9590646d12c0e3a4de1ddbe6f8e20e4262c

            self.mean = torch.from_numpy(self.mean).cuda()
            self.std = torch.from_numpy(self.std).cuda()
     
        try:
            self.vae_type = cfg.model.vae_type
        except:
            self.vae_type = cfg.model.motion_vae.target.split(
                ".")[-1].lower().replace("vae", "")

        self.text_encoder = instantiate_from_config(cfg.model.text_encoder)

        if self.vae_type != "no":
            self.vae = instantiate_from_config(cfg.model.motion_vae)
        
        self.denoiser = instantiate_from_config(cfg.model.denoiser)

<<<<<<< HEAD

        print("nclass",nclass)
=======
>>>>>>> 045ca9590646d12c0e3a4de1ddbe6f8e20e4262c
        self.style_function = StyleClassification(nclasses=nclass)
        self.style_function.eval()

    
        # Don't train the motion encoder and decoder
        if self.stage in ["diffusion","cycle_diffusion"]:
            if self.vae_type in ["mld", "vposert","actor"]:
                self.vae.training = False
                for p in self.vae.parameters():
                    p.requires_grad = False
                for p in self.denoiser.mld_denoiser.parameters():
                    p.requires_grad = False
                for p in self.style_function.parameters():
                    p.requires_grad = False


            elif self.vae_type == "no":
                pass
            else:
                self.motion_encoder.training = False
                for p in self.motion_encoder.parameters():
                    p.requires_grad = False
                self.motion_decoder.training = False
                for p in self.motion_decoder.parameters():
                    p.requires_grad = False

        assert(self.vae_type != "no")
        
        if not self.predict_epsilon:
            cfg.model.scheduler.params['prediction_type'] = 'sample'
            cfg.model.noise_scheduler.params['prediction_type'] = 'sample'

        self.scheduler = instantiate_from_config(cfg.model.scheduler)
        self.noise_scheduler = instantiate_from_config(cfg.model.noise_scheduler)
        self.inversion_scheduler = instantiate_from_config(cfg.model.inversion_scheduler)

        if self.condition in ["text", "text_uncond"]:
            self._get_t2m_evaluator(cfg)

        if cfg.TRAIN.OPTIM.TYPE.lower() == "adamw":
            self.optimizer = AdamW(params=filter(lambda p: p.requires_grad, self.parameters()), lr=cfg.TRAIN.OPTIM.LR)

        else:
            raise NotImplementedError(
                "Do not support other optimizer for now.")

        if cfg.LOSS.TYPE == "mld":
            self._losses = MetricCollection({
                split: MLDLosses(vae=self.is_vae, mode="xyz", cfg=cfg)
                for split in ["losses_train", "losses_test", "losses_val"]
            })
        else:
            raise NotImplementedError(
                "MotionCross model only supports mld losses.")

        self.losses = {
            key: self._losses["losses_" + key]
            for key in ["train", "test", "val"]
        }

        self.metrics_dict = cfg.METRIC.TYPE
        self.configure_metrics()

        # If we want to overide it at testing time
        self.sample_mean = False
        self.fact = None
        self.do_classifier_free_guidance = self.guidance_scale > 1.0
        if self.condition in ['text', 'text_uncond']:
            self.feats2joints_wo_norm = datamodule.feats2joints_wo_norm
            self.feats2joints = datamodule.feats2joints

    def _get_t2m_evaluator(self, cfg):
        """
        load T2M text encoder and motion encoder for evaluating
        """
        # init module
        self.t2m_textencoder = t2m_textenc.TextEncoderBiGRUCo(
            word_size=cfg.model.t2m_textencoder.dim_word,
            pos_size=cfg.model.t2m_textencoder.dim_pos_ohot,
            hidden_size=cfg.model.t2m_textencoder.dim_text_hidden,
            output_size=cfg.model.t2m_textencoder.dim_coemb_hidden,
        )

        self.t2m_moveencoder = t2m_motionenc.MovementConvEncoder(
            input_size=cfg.DATASET.NFEATS - 4,
            hidden_size=cfg.model.t2m_motionencoder.dim_move_hidden,
            output_size=cfg.model.t2m_motionencoder.dim_move_latent,
        )

        self.t2m_motionencoder = t2m_motionenc.MotionEncoderBiGRUCo(
            input_size=cfg.model.t2m_motionencoder.dim_move_latent,
            hidden_size=cfg.model.t2m_motionencoder.dim_motion_hidden,
            output_size=cfg.model.t2m_motionencoder.dim_motion_latent,
        )
        # load pretrianed
        dataname = cfg.TEST.DATASETS[0]
        dataname = "t2m" if dataname == "humanml3d" else dataname
        t2m_checkpoint = torch.load(
            os.path.join(cfg.model.t2m_path, dataname,
                         "text_mot_match/model/finest.tar"))
        self.t2m_textencoder.load_state_dict(t2m_checkpoint["text_encoder"])
        self.t2m_moveencoder.load_state_dict(
            t2m_checkpoint["movement_encoder"])
        self.t2m_motionencoder.load_state_dict(
            t2m_checkpoint["motion_encoder"])

        # freeze params
        self.t2m_textencoder.eval()
        self.t2m_moveencoder.eval()
        self.t2m_motionencoder.eval()
        for p in self.t2m_textencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_moveencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_motionencoder.parameters():
            p.requires_grad = False

    def sample_from_distribution(
        self,
        dist,
        *,
        fact=None,
        sample_mean=False,
    ) -> Tensor:
        fact = fact if fact is not None else self.fact
        sample_mean = sample_mean if sample_mean is not None else self.sample_mean

        if sample_mean:
            return dist.loc.unsqueeze(0)

        # Reparameterization trick
        if fact is None:
            return dist.rsample().unsqueeze(0)

        # Resclale the eps
        eps = dist.rsample() - dist.loc
        z = dist.loc + fact * eps

        # add latent size
        z = z.unsqueeze(0)
        return z

    def sample_timestep(self,batch,timestep=100):
        texts = batch["text"]
        lengths = batch["length"]
        feats_ref = None
        if "motion" in batch.keys():
            feats_ref = batch["motion"]

        if self.cfg.TEST.COUNT_TIME:
            self.starttime = time.time()

        if self.stage in ['diffusion', 'vae_diffusion']:
            # diffusion reverse
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(texts)
                if self.condition == 'text':
                    uncond_tokens.extend(texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                texts = uncond_tokens
            text_emb = self.text_encoder(texts)

            z,time_list = self._diffusion_reverse_timestep(text_emb, lengths,feats_ref,timestep)
        elif self.stage in ['vae']:
            motions = batch['motion']
            z, dist_m = self.vae.encode(motions, lengths)

        with torch.no_grad():
            # ToDo change mcross actor to same api
            if self.vae_type in ["mld","actor"]:
                feats_rst_list = []
                for i in range(len(z)):
                    latent = z[i]

                    feats_rst = self.vae.decode(latent, lengths)
                    feats_rst_list.append(feats_rst)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)

        if self.cfg.TEST.COUNT_TIME:
            self.endtime = time.time()
            elapsed = self.endtime - self.starttime
            self.times.append(elapsed)
            if len(self.times) % 100 == 0:
                meantime = np.mean(
                    self.times[-100:]) / self.cfg.TEST.BATCH_SIZE
                print(
                    f'100 iter mean Time (batch_size: {self.cfg.TEST.BATCH_SIZE}): {meantime}',
                )
            if len(self.times) % 1000 == 0:
                meantime = np.mean(
                    self.times[-1000:]) / self.cfg.TEST.BATCH_SIZE
                print(
                    f'1000 iter mean Time (batch_size: {self.cfg.TEST.BATCH_SIZE}): {meantime}',
                )
                with open(pjoin(self.cfg.FOLDER_EXP, 'times.txt'), 'w') as f:
                    for line in self.times:
                        f.write(str(line))
                        f.write('\n')
        
        joints_list = []
        for i in range(len(feats_rst_list)):

            feats_rst = feats_rst_list[i]
            joints = self.feats2joints(feats_rst.detach().cpu())
            joints = remove_padding(joints, lengths)
            joints_list.append(joints)

        return joints_list,time_list

    def reconstruct(self, batch,feature = "False"):
        motion = batch["motion"].cuda().float()
        lengths = batch["length"]
       
        # motion = motion[:,:196,:]
        bs,len,_ = motion.shape
        z, dist_m = self.vae.encode(motion, lengths)
        feats_rst = self.vae.decode(z, lengths)
        
        # feats_rst = motion

        error = (feats_rst - motion).mean()
        joints = self.feats2joints(feats_rst.detach().cpu())
        if feature == "False":
            return remove_padding(joints, lengths)
        else:
            return feats_rst.detach().cpu()
        
    def prev_step(self, model_output, timestep, sample):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev**0.5 * pred_original_sample + pred_sample_direction
        return prev_sample

    def next_step(self, model_output, timestep, sample):
        timestep, next_timestep = (
            min(timestep - self.scheduler.config.num_train_timesteps // self.num_inference_steps, 999),
            timestep,
        )
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next**0.5 * next_original_sample + next_sample_direction
        return next_sample

    @torch.no_grad()
    def invert(self, batch):
        texts = batch["text"]
        lengths = batch["length"]
        motion = batch["motion"].cuda().float()        
        
        # style_motion = batch["style_motion"]
        # style_motion = style_motion[:,:480,:]

        latents , dist_m = self.vae.encode(motion, lengths)

        # 5. Encode input prompt
        if self.do_classifier_free_guidance:
            uncond_tokens = [""] * len(texts)
            if self.condition == 'text':
                uncond_tokens.extend(texts)
            elif self.condition == 'text_uncond':
                uncond_tokens.extend(uncond_tokens)
            texts = uncond_tokens
            text_emb = self.text_encoder(texts)
            text_emb = text_emb.cuda()

        # 4. Prepare timesteps
        num_inference_steps = self.cfg.model.inversion_scheduler.num_inference_timesteps
        self.inversion_scheduler.set_timesteps(num_inference_steps)
        timesteps = self.inversion_scheduler.timesteps

        # 7. Denoising loop where we obtain the cross-attention maps.
        num_warmup_steps = len(timesteps) - num_inference_steps * self.inversion_scheduler.order
        intermediate_latents = []

        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            latent_model_input = self.inversion_scheduler.scale_model_input(latent_model_input, t)

            self.denoiser = self.denoiser.cuda()

            noise_pred = self.denoiser(
                sample=latent_model_input,
                timestep=t.cuda(),
                encoder_hidden_states=text_emb,
                lengths=lengths,
            )[0]

            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.inversion_scheduler.step(noise_pred, t, latents).prev_sample
            intermediate_latents.append(latents)

        start_step=20
        inverted_latents = intermediate_latents[-(start_step+1)] #latents.detach().clone()

        return inverted_latents
            
        # set timesteps

    def forward(self, batch,feature="False"):
        texts = batch["text"]
        lengths = batch["length"]
        feats_ref = None
        
        if "motion" in batch.keys():
            feats_ref = batch["motion"]
            feats_ref = feats_ref[:,:480,:]

        if self.cfg.TEST.COUNT_TIME:
            self.starttime = time.time()

        if self.stage in ['diffusion', 'vae_diffusion',"cycle_diffusion"]:
            # diffusion reverse
            if self.do_classifier_free_guidance:
                if self.guidance_mode == 'v0':
                    uncond_tokens = [""] * len(texts)
                    if self.condition == 'text':
                        uncond_tokens.extend(texts)
                    elif self.condition == 'text_uncond':
                        uncond_tokens.extend(uncond_tokens)
                    texts = uncond_tokens
                    text_emb = self.text_encoder(texts)
                    z = self._diffusion_reverse(text_emb, lengths,feats_ref)
                
                elif self.guidance_mode == 'v4':
                    batch_size = len(texts)
<<<<<<< HEAD
                    feats_ref = feats_ref.expand(batch_size,-1,-1)
                    empty_ref = torch.zeros_like(feats_ref)

                    feats_ref = torch.cat((empty_ref,empty_ref,feats_ref),dim=0)
=======
                    reference_motion = reference_motion.expand(batch_size,-1,-1)
                    empty_ref = torch.zeros_like(reference_motion)

                    reference_motion = torch.cat((empty_ref,empty_ref,reference_motion),dim=0)
>>>>>>> 045ca9590646d12c0e3a4de1ddbe6f8e20e4262c

                    uncond_tokens = [""] * len(texts)
                    uncond_tokens2 = [""] * len(texts)
                    if self.condition == 'text':
                        uncond_tokens.extend(texts)
                        uncond_tokens.extend(uncond_tokens2)
                    elif self.condition == 'text_uncond':
                        uncond_tokens.extend(uncond_tokens)
                    texts = uncond_tokens
                    text_emb = self.text_encoder(texts)
     
<<<<<<< HEAD
                    z = self._diffusion_reverse(text_emb, lengths,feats_ref, mode = "v4")
                
                elif self.guidance_mode == 'v2':
                    batch = len(texts)
                    feats_ref = feats_ref.expand(batch,-1,-1)
                    empty_ref = torch.zeros_like(feats_ref)

                    feats_ref = torch.cat((empty_ref,empty_ref,feats_ref),dim=0)
                    # uncond_tokens = [texts[i] for i in range(len(texts))]
                    uncond_tokens = [""] * len(texts)
                    uncond_tokens2 = [""] * len(texts)
                    if self.condition == 'text':
                        uncond_tokens.extend(texts)
                        uncond_tokens.extend(uncond_tokens2)

                    elif self.condition == 'text_uncond':
                        uncond_tokens.extend(uncond_tokens)
                    texts = uncond_tokens
                    text_emb = self.text_encoder(texts)
     
                    z = self._diffusion_reverse(text_emb, lengths,feats_ref,mode = "v2")
=======
                    z = self._diffusion_reverse(text_emb, lengths,feats_ref, is_v4=True)
>>>>>>> 045ca9590646d12c0e3a4de1ddbe6f8e20e4262c

        elif self.stage in ['vae']:
            motions = batch['motion']
            z, dist_m = self.vae.encode(motions, lengths)

        with torch.no_grad():
            # ToDo change mcross actor to same api
            if self.vae_type in ["mld","actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)

        if self.cfg.TEST.COUNT_TIME:
            self.endtime = time.time()
            elapsed = self.endtime - self.starttime
            self.times.append(elapsed)
            if len(self.times) % 100 == 0:
                meantime = np.mean(
                    self.times[-100:]) / self.cfg.TEST.BATCH_SIZE
                print(
                    f'100 iter mean Time (batch_size: {self.cfg.TEST.BATCH_SIZE}): {meantime}',
                )
            if len(self.times) % 1000 == 0:
                meantime = np.mean(
                    self.times[-1000:]) / self.cfg.TEST.BATCH_SIZE
                print(
                    f'1000 iter mean Time (batch_size: {self.cfg.TEST.BATCH_SIZE}): {meantime}',
                )
                with open(pjoin(self.cfg.FOLDER_EXP, 'times.txt'), 'w') as f:
                    for line in self.times:
                        f.write(str(line))
                        f.write('\n')

        joints = self.feats2joints(feats_rst.detach().cpu())

        if feature == "True":
            return remove_padding(joints, lengths),feats_rst.detach().cpu()
        else:
            return remove_padding(joints, lengths)

    def gen_from_latent(self, batch):
        z = batch["latent"]
        lengths = batch["length"]

        feats_rst = self.vae.decode(z, lengths)

        # feats => joints
        joints = self.feats2joints(feats_rst.detach().cpu())
        return remove_padding(joints, lengths)

    def recon_from_motion(self, batch):
        feats_ref = batch["motion"].float().cuda()
        length = batch["length"]

        z, dist = self.vae.encode(feats_ref, length)
        feats_rst = self.vae.decode(z, length)

        # feats => joints
        joints = self.feats2joints(feats_rst.detach().cpu())
        joints_ref = self.feats2joints(feats_ref.detach().cpu())
        return remove_padding(joints,
                              length), remove_padding(joints_ref, length)

    def _diffusion_reverse_timestep(self, encoder_hidden_states, lengths=None,reference_motion=None,time_step=100):
        # init latents
        bsz = encoder_hidden_states.shape[0]
        if self.do_classifier_free_guidance:
            bsz = bsz // 2

        if self.vae_type == "no":
            assert lengths is not None, "no vae (diffusion only) need lengths for diffusion"
            latents = torch.randn(
                (bsz, max(lengths), self.cfg.DATASET.NFEATS),
                device=encoder_hidden_states.device,
                dtype=torch.float,
            )
        else:
            latents = torch.randn(
                (bsz, self.latent_dim[0], self.latent_dim[-1]),
                device=encoder_hidden_states.device,
                dtype=torch.float,
            )

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        # set timesteps
        self.scheduler.set_timesteps(
            self.cfg.model.scheduler.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(encoder_hidden_states.device)
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs = {}
        if "eta" in set(
                inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta

        # reverse
        latent_list = []
        time_list = []
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (torch.cat(
                [latents] *
                2) if self.do_classifier_free_guidance else latents)
            lengths_reverse = (lengths * 2 if self.do_classifier_free_guidance else lengths)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # predict the noise residual

            if reference_motion is None or self.is_control == False:
                noise_pred = self.denoiser(
                    sample=latent_model_input,
                    timestep=t,
                    encoder_hidden_states=encoder_hidden_states,
                    lengths=lengths,
                )[0]
            elif self.is_control:
                noise_pred = self.denoiser(
                    sample=latent_model_input,
                    timestep=t,
                    encoder_hidden_states=encoder_hidden_states,
                    lengths=lengths,
                    reference=reference_motion,
                )[0]
            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond)
            
            # clean_sample = self.get_clean_sample(noise_pred,latents,t)
            latents = self.scheduler.step(noise_pred, t, latents,**extra_step_kwargs).prev_sample
            current_t = int(t.cpu().numpy())-1
            if current_t % time_step == 0:
                latent_list.append(latents.permute(1, 0, 2))
                time_list.append(current_t)


        # [batch_size, 1, latent_dim] -> [1, batch_size, latent_dim]
        return latent_list,time_list


<<<<<<< HEAD
    def _diffusion_reverse(self, encoder_hidden_states, lengths=None,reference_motion=None,mode = "v2"):
        # init latents
        bsz = encoder_hidden_states.shape[0]
        if self.do_classifier_free_guidance and mode == "v0":
            bsz = bsz // 2
        elif self.do_classifier_free_guidance and mode != "v0":
=======
    def _diffusion_reverse(self, encoder_hidden_states, lengths=None,reference_motion=None,is_v4 = False):
        # init latents
        bsz = encoder_hidden_states.shape[0]
        if self.do_classifier_free_guidance and is_v4 == False:
            bsz = bsz // 2
        elif self.do_classifier_free_guidance and is_v4 == True:
>>>>>>> 045ca9590646d12c0e3a4de1ddbe6f8e20e4262c
            bsz = bsz // 3

        if self.vae_type == "no":
            assert lengths is not None, "no vae (diffusion only) need lengths for diffusion"
            latents = torch.randn(
                (bsz, max(lengths), self.cfg.DATASET.NFEATS),
                device=encoder_hidden_states.device,
                dtype=torch.float,
            )
        else:
            latents = torch.randn(
                (bsz, self.latent_dim[0], self.latent_dim[-1]),
                device=encoder_hidden_states.device,
                dtype=torch.float,
            )

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
            
        # set timesteps
        self.scheduler.set_timesteps(
            self.cfg.model.scheduler.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(encoder_hidden_states.device)
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs = {}
        if "eta" in set(
                inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta

        # reverse
        # delat_pre_style_return = 0
        for i, t in enumerate(timesteps):
            current_t = int(t.cpu().numpy())-1
            # expand the latents if we are doing classifier free guidance
<<<<<<< HEAD
            if mode == "v0":
=======
            if is_v4 == False:
>>>>>>> 045ca9590646d12c0e3a4de1ddbe6f8e20e4262c
                latent_model_input = (torch.cat(
                    [latents] *
                    2) if self.do_classifier_free_guidance else latents)
            else:
                latent_model_input = (torch.cat(
                    [latents] *
                    3) if self.do_classifier_free_guidance else latents)

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # predict the noise residual

            if reference_motion is None or self.is_control == False:
                noise_pred = self.denoiser(
                    sample=latent_model_input,
                    timestep=t,
                    encoder_hidden_states=encoder_hidden_states,
                    lengths=lengths,
                )[0]
            elif self.is_control:
                noise_pred = self.denoiser(
                    sample=latent_model_input,
                    timestep=t,
                    encoder_hidden_states=encoder_hidden_states,
                    lengths=lengths,
                    reference=reference_motion,
                )[0]
            
            # perform guidance
<<<<<<< HEAD
            if self.do_classifier_free_guidance and mode == "v0":
=======
            if self.do_classifier_free_guidance and is_v4 == False:
>>>>>>> 045ca9590646d12c0e3a4de1ddbe6f8e20e4262c
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                if self.is_guidance and current_t < 300:
                    with torch.set_grad_enabled(True):
                        noise_pred = self.guide(noise_pred, t, latents, lengths,reference_motion)

<<<<<<< HEAD
            elif self.do_classifier_free_guidance and mode == "v4":
=======
            elif self.do_classifier_free_guidance and is_v4 == True:
>>>>>>> 045ca9590646d12c0e3a4de1ddbe6f8e20e4262c
               
                noise_pred_uncond,noise_pred_text,noise_pred_style = noise_pred.chunk(3)
                delat_pre_text = noise_pred_text - noise_pred_uncond
                delat_pre_style = noise_pred_style - noise_pred_uncond

                noise_pred = noise_pred_uncond + self.guidance_scale_style * delat_pre_style

                _,_,reference = reference_motion.chunk(3)

                
                if self.is_guidance and current_t < 300:
                    with torch.set_grad_enabled(True):
                        noise_pred = self.guide(noise_pred, t, latents, lengths,reference)

                noise_pred = noise_pred + self.guidance_scale * delat_pre_text
<<<<<<< HEAD
            
            elif self.do_classifier_free_guidance and mode == "v2":
               
                noise_pred_uncond,noise_pred_text,noise_pred_style = noise_pred.chunk(3)
                delat_pre_text = noise_pred_text - noise_pred_uncond
                delat_pre_style = noise_pred_style - noise_pred_uncond
                
                noise_pred = noise_pred_uncond + self.guidance_scale_style * delat_pre_style

                # if self.is_style_text == False:
                _,_,reference = reference_motion.chunk(3)

                if self.is_guidance and current_t < 300:
                    with torch.set_grad_enabled(True):
                        noise_pred = self.guide(noise_pred, t, latents, lengths,reference)
                
                # delat_pre_style_m = noise_pred - noise_pred_uncond
                noise_pred = noise_pred + self.guidance_scale * delat_pre_text
=======
>>>>>>> 045ca9590646d12c0e3a4de1ddbe6f8e20e4262c

            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
 
        # [batch_size, 1, latent_dim] -> [1, batch_size, latent_dim]
        latents = latents.permute(1, 0, 2)
        return latents

    def guide(self,noise_pred, timestep, latents,lengths,reference_motion,n_guide_steps=5,scale=0.2):
        #compute pred_original_sample -> f_theta(x_t, t) or x_0
        noise_pred = noise_pred.requires_grad_(True)
        for _ in range(n_guide_steps):
            loss, grad = self.gradients(noise_pred,latents,timestep,reference_motion,lengths)
            noise_pred = noise_pred - scale * grad
        noise_pred = noise_pred.detach().requires_grad_(True)
        return noise_pred
    

    def gradients(self,noise_pred,latents,timestep,reference_motion,lengths):
        with torch.enable_grad():
            noise_pred.requires_grad_(True)
            batch_size = noise_pred.shape[0]
            #compute pred_original_sample -> f_theta(x_t, t) or x_0
            
            clean_sample = self.get_clean_sample(noise_pred,latents,timestep)
            joints = self.vae.decode(clean_sample.permute(1, 0, 2).contiguous(), lengths)

            style_pred = self.style_function(joints,stage="intermediate") 
            style_reference = self.style_function(reference_motion,stage="intermediate") 
            loss = 0.0

            weights = [0.0, 2.0] 
            for i, (pred_feat, target_feat) in enumerate(zip(style_pred, style_reference)):
                loss += weights[i] * F.l1_loss(pred_feat[0], target_feat[0], reduction='none')
                loss += weights[i] * F.l1_loss(pred_feat[1], target_feat[1], reduction='none')

            loss_sum = loss.sum() /  batch_size
            grad = torch.autograd.grad([loss_sum], [noise_pred])[0]

            grad = grad.clone()
            grad[..., 0] = 0
        
        noise_pred.detach()
    
        return loss_sum, grad


    def get_clean_sample(self,noise_pred,latents,timestep):

        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
    
        return pred_original_sample


    def _diffusion_reverse_tsne(self, encoder_hidden_states, lengths=None):
        # init latents
        bsz = encoder_hidden_states.shape[0]
        if self.do_classifier_free_guidance:
            bsz = bsz // 2
        if self.vae_type == "no":
            assert lengths is not None, "no vae (diffusion only) need lengths for diffusion"
            latents = torch.randn(
                (bsz, max(lengths), self.cfg.DATASET.NFEATS),
                device=encoder_hidden_states.device,
                dtype=torch.float,
            )
        else:
            latents = torch.randn(
                (bsz, self.latent_dim[0], self.latent_dim[-1]),
                device=encoder_hidden_states.device,
                dtype=torch.float,
            )

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        # set timesteps
        self.scheduler.set_timesteps(
            self.cfg.model.scheduler.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(encoder_hidden_states.device)
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs = {}
        if "eta" in set(
                inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta

        # reverse
        latents_t = []
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (torch.cat(
                [latents] *
                2) if self.do_classifier_free_guidance else latents)
            lengths_reverse = (lengths * 2 if self.do_classifier_free_guidance
                               else lengths)
            # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # predict the noise residual
            noise_pred = self.denoiser(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                lengths=lengths_reverse,
            )[0]
            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond)
                # text_embeddings_for_guidance = encoder_hidden_states.chunk(
                #     2)[1] if self.do_classifier_free_guidance else encoder_hidden_states
            latents = self.scheduler.step(noise_pred, t, latents,
                                              **extra_step_kwargs).prev_sample
            # [batch_size, 1, latent_dim] -> [1, batch_size, latent_dim]
            latents_t.append(latents.permute(1,0,2))
        # [1, batch_size, latent_dim] -> [t, batch_size, latent_dim]
        latents_t = torch.cat(latents_t)
        return latents_t

    def _cycle_recon_process(self, latents_x,timesteps, encoder_hidden_states_x, lengths_x,reference_motion_x):
        # Sample noise that we'll add to the latents
        # [batch_size, n_token, latent_dim]
        noise_x = torch.randn_like(latents_x)

        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_latents_x = self.noise_scheduler.add_noise(latents_x.clone(), noise_x,timesteps)
        
        # Predict the noise residual
        noise_pred_x = self.denoiser(
                sample=noisy_latents_x,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states_x,
                lengths=lengths_x,
                reference=reference_motion_x,
                return_dict=False,
            )[0]

        return noise_x, noise_pred_x,noisy_latents_x
        
        
    def _cycle_diffusion_process(self, latents_x, latents_y, encoder_hidden_states_x,encoder_hidden_states_y, lengths_x,lengths_y,reference_motion_x,reference_motion_y):
        latents_x = latents_x.permute(1, 0, 2)
        latents_y = latents_y.permute(1, 0, 2)
        bsz = latents_x.shape[0]
        # Sample a random timestep for each motion
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz, ),
            device=latents_x.device,
        )
        timesteps = timesteps.long()

        # x from HumanML3D and y from 100STYLE
        noise_x, noise_pred_x,noisy_latents_x = self._cycle_recon_process(latents_x,timesteps, encoder_hidden_states_x, lengths_x, reference_motion_x)
        noise_y, noise_pred_y,noisy_latents_y = self._cycle_recon_process(latents_y,timesteps, encoder_hidden_states_y, lengths_y, reference_motion_y)

        noise_pred_prior = 0
        noise_prior = 0

        n_set = {
            "noise": noise_x,
            "noise_pred": noise_pred_x,

            "noise_y": noise_y,
            "noise_pred_y":noise_pred_y,

            "noise_pred_prior": noise_pred_prior,
            "noise_prior": noise_prior,
        }

        if self.is_style_recon:
            n_set["style_emb"] = style_emb
            n_set["style_emb_recon"] = style_emb_recon

        # predict_epsilon : True
        if not self.predict_epsilon:
            n_set["pred"] = noise_pred_x
            n_set["latent"] = latents_x
        return n_set
    
    def _cyclenet_process(self, x_start, y_start, encoder_hidden_states_x,encoder_hidden_states_y, lengths_x,lengths_y,reference_motion_x,reference_motion_y,reference_motion_x_same=None, cond_emb_empty = None):
        x_start = x_start.permute(1, 0, 2)
        y_start = y_start.permute(1, 0, 2)

        noise_x = torch.randn_like(x_start)
        noise_y = torch.randn_like(y_start)

        bsz = x_start.shape[0]
        # Sample a random timestep for each motion
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz, ),
            device=x_start.device,
        )

        timesteps = timesteps.long()

        # x from HumanML3D and y from 100STYLE
        # Add noise to the latents according to the noise magnitude at each timestep
        x_noise = self.noise_scheduler.add_noise(x_start.clone(), noise_x,timesteps)
        y_noise = self.noise_scheduler.add_noise(y_start.clone(), noise_y,timesteps)
        
        # # Predict the noise residual
        recon_x_output = self.denoiser(
                sample=x_noise,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states_x,
                lengths=lengths_x,
                reference=reference_motion_x,
                return_dict=False,
            )[0]
        
        recon_y_output = self.denoiser(
                sample=y_noise,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states_y,
                lengths=lengths_y,
                reference=reference_motion_y,
                return_dict=False,
            )[0]

        noise_xy_prime = self.denoiser(
                sample=x_noise,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states_x,
                lengths=lengths_x,
                reference=reference_motion_y,
                return_dict=False,
            )[0]
        
        noise_yx_prime = self.denoiser(
                sample=y_noise,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states_y,
                lengths=lengths_y,
                reference=reference_motion_x,
                return_dict=False,
            )[0]

        self.scheduler.set_timesteps(self.cfg.model.scheduler.num_inference_timesteps)

        if self.is_tri and reference_motion_x_same is not None:
           

            noise_pred_pos = self.denoiser(
                sample=x_noise,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states_x,
                lengths=lengths_x,
                reference=reference_motion_x_same,
                return_dict=False,
            )[0]


        extra_step_kwargs = {}
        if "eta" in set(
                inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta

        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.cuda()
        
        xy_prime = self.get_clean_sample(noise_xy_prime, x_noise, timesteps.unsqueeze(1).unsqueeze(2))
        xy_cond = self.vae.decode(xy_prime.permute(1, 0, 2).contiguous(), lengths_x)

        yx_prime = self.get_clean_sample(noise_yx_prime, y_noise, timesteps.unsqueeze(1).unsqueeze(2))
        yx_cond = self.vae.decode(yx_prime.permute(1, 0, 2).contiguous(), lengths_y)

        noise_xy = torch.randn_like(xy_prime.detach())
        xy_noise = self.noise_scheduler.add_noise(xy_prime.detach(), noise_xy, timesteps)

        noise_yx = torch.randn_like(yx_prime.detach())
        yx_noise = self.noise_scheduler.add_noise(yx_prime.detach(), noise_yx, timesteps)

        noise_xyx = self.denoiser(
                sample=xy_noise,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states_x,
                lengths=lengths_x,
                reference=yx_cond,
                return_dict=False,
            )[0]

        noise_yxy = self.denoiser(
                sample=yx_noise,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states_y,
                lengths=lengths_x,
                reference=xy_cond,
                return_dict=False,
            )[0]
  
        n_set = {
            "noise_pred":recon_x_output ,
            "noise": noise_x,

            "noise_pred_y":recon_y_output ,
            "noise_y": noise_y,

            "noise_pred_cycle":noise_xyx + noise_xy_prime.detach() ,
            "noise_cycle":noise_x,

            "noise_pred_cycle_y":noise_yxy + noise_yx_prime.detach() ,
            "noise_cycle_y":noise_y,
        }

        return n_set
        # #->

    def _diffusion_process(self, latents, encoder_hidden_states,lengths=None,reference_motion=None,n_encoder_hidden_states=None):
        """
        heavily from https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py
        """
        # our latent   [batch_size, n_token=1 or 5 or 10, latent_dim=256]
        # sd  latent   [batch_size, [n_token0=64,n_token1=64], latent_dim=4]
        # [n_token, batch_size, latent_dim] -> [batch_size, n_token, latent_dim]
        latents = latents.permute(1, 0, 2)
        # Sample noise that we'll add to the latents
        # [batch_size, n_token, latent_dim]
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each motion
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz, ),
            device=latents.device,
        )

        timesteps = timesteps.long()
        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_latents = self.noise_scheduler.add_noise(latents.clone(), noise, timesteps)

        # Predict the noise residual
        if reference_motion is None:
            noise_pred = self.denoiser(
                sample=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                n_encoder_hidden_states=n_encoder_hidden_states,
                lengths=lengths,
                return_dict=False,
            )[0]
        else:
            noise_pred = self.denoiser(
                sample=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                n_encoder_hidden_states=n_encoder_hidden_states,
                lengths=lengths,
                reference=reference_motion,
                return_dict=False,
            )[0]


        if self.is_guidance:
            self.scheduler.set_timesteps(self.cfg.model.scheduler.num_inference_timesteps)
            # # latents_xy = self.scheduler.step(noise_pred_xy, timesteps.cuda(), noisy_latents_x,**extra_step_kwargs).prev_sample #self.get_clean_sample(noise_pred_xy,noisy_latents_x,timesteps)
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.cuda()
            with torch.enable_grad():
                noise_pred = self.guide(noise_pred, timesteps.unsqueeze(1).unsqueeze(2), noisy_latents, lengths,reference_motion)

        noise_pre_mld = 0.0
        time_mask = torch.where(timesteps > 700, torch.tensor(1).cuda(), torch.tensor(0).cuda())#(timesteps >= 700).bool
        

        # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
        if self.cfg.LOSS.LAMBDA_PRIOR != 0.0:
            noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
            noise, noise_prior = torch.chunk(noise, 2, dim=0)
        else:
            noise_pred_prior = 0
            noise_prior = 0

        n_set = {
            "noise": noise,
            "noise_pred": noise_pred,

            "noise_pred_prior": noise_pred_prior,
            "noise_prior": noise_prior,
        }

        # predict_epsilon : True
        if not self.predict_epsilon:
            n_set["pred"] = noise_pred
            n_set["latent"] = latents
        return n_set

    def train_vae_forward(self, batch):
        feats_ref = batch["motion"]
        lengths = batch["length"]
        if self.vae_type in ["mld", "vposert", "actor"]:
            motion_z, dist_m = self.vae.encode(feats_ref, lengths)
            
            feats_rst = self.vae.decode(motion_z, lengths)
        else:
            raise TypeError("vae_type must be mcross or actor")

        # prepare for metric
        recons_z, dist_rm = self.vae.encode(feats_rst, lengths)

        # joints recover
        if self.condition == "text":
            joints_rst = self.feats2joints(feats_rst)
            joints_ref = self.feats2joints(feats_ref)
        elif self.condition == "action":
            mask = batch["mask"]
            joints_rst = self.feats2joints(feats_rst, mask)
            joints_ref = self.feats2joints(feats_ref, mask)

        if dist_m is not None:
            if self.is_vae:
                # Create a centred normal distribution to compare with
                mu_ref = torch.zeros_like(dist_m.loc)
                scale_ref = torch.ones_like(dist_m.scale)
                dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
            else:
                dist_ref = dist_m

        # cut longer part over max length
        min_len = min(feats_ref.shape[1], feats_rst.shape[1])

        rs_set = {
            "m_ref": feats_ref[:, :min_len, :],
            "m_rst": feats_rst[:, :min_len, :],
            # [bs, ntoken, nfeats]<= [ntoken, bs, nfeats]
            "lat_m": motion_z.permute(1, 0, 2),
            "lat_rm": recons_z.permute(1, 0, 2),
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "dist_m": dist_m,
            "dist_ref": dist_ref,
        }
        return rs_set

    def train_cycle_diffusion_forward(self, batch):
        feats_ref_1 = batch["motion_1"]
        feats_ref_2 = batch["motion_2"]
        lengths_1 = batch["length_1"]
        lengths_2 = batch["length_2"]
        # motion encode
        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                z_1, dist_1 = self.vae.encode(feats_ref_1, lengths_1)
                z_2, dist_2 = self.vae.encode(feats_ref_2, lengths_2)

        if self.condition in ["text", "text_uncond"]:
            text_1 = batch["text_1"]
            text_2 = batch["text_2"]

            text_1 = [
                "" if np.random.rand(1) < self.guidance_uncodp else i
                for i in text_1
            ]
            text_2 = [
                "" if np.random.rand(1) < self.guidance_uncodp else i
                for i in text_2
            ]

            if self.guidance_mode == 'v4':

                dim0_length = feats_ref_1.shape[0]
                mask1 = torch.bernoulli(torch.full((dim0_length,), self.guidance_uncondp_style)).unsqueeze(1).unsqueeze(2)
                mask1 = mask1.expand_as(feats_ref_1).cuda()

                mask2 = torch.bernoulli(torch.full((dim0_length,), self.guidance_uncondp_style)).unsqueeze(1).unsqueeze(2)
                mask2 = mask2.expand_as(feats_ref_2).cuda()

                feats_ref_1 = feats_ref_1 * mask1
                feats_ref_2 = feats_ref_2 * mask2        

            # text encode
            cond_emb_1 = self.text_encoder(text_1)
            cond_emb_2 = self.text_encoder(text_2)

        # diffusion process return with noise and noise_pred
        if self.is_recon:
            # Disable cycle loss.
            n_set = self._cycle_diffusion_process(z_1,z_2, cond_emb_1,cond_emb_2,lengths_1,lengths_2,feats_ref_1,feats_ref_2)
        else:
            # Enable cycle loss.
            n_set = self._cyclenet_process(z_1,z_2, cond_emb_1,cond_emb_2,lengths_1,lengths_2,feats_ref_1,feats_ref_2)

        return {**n_set}
    
    def train_diffusion_forward(self, batch):
        feats_ref = batch["motion"]
        lengths = batch["length"]

        # motion encode
        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                z, dist = self.vae.encode(feats_ref, lengths)
              
            elif self.vae_type == "no":
                z = feats_ref.permute(1, 0, 2)
            else:
                raise TypeError("vae_type must be mcross or actor")

        if self.condition in ["text", "text_uncond"]:
            text = batch["text"]
            text_len = len(text)

            # empty_text = [""]*text_len
            # classifier free guidance: randomly drop text during training
            text = [
                "" if np.random.rand(1) < self.guidance_uncodp else i
                for i in text
            ]
            # text encode
            cond_emb = self.text_encoder(text)
            # empty_emb = self.text_encoder(empty_text)

        elif self.condition in ['action']:
            action = batch['action']
            # text encode
            cond_emb = action
        else:
            raise TypeError(f"condition type {self.condition} not supported")

        # diffusion process return with noise and noise_pred
        # n_set = self._diffusion_process(z, cond_emb, lengths,feats_ref)
        if self.is_control or self.is_guidance:
            n_set = self._diffusion_process(z, cond_emb, lengths,feats_ref,n_encoder_hidden_states=cond_emb)
        else:
            n_set = self._diffusion_process(z, cond_emb, lengths,n_encoder_hidden_states=cond_emb)
        return {**n_set}

    def test_diffusion_forward(self, batch, finetune_decoder=False):
        lengths = batch["length"]

        if self.condition in ["text", "text_uncond"]:
            # get text embeddings
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(lengths)
                if self.condition == 'text':
                    texts = batch["text"]
                    uncond_tokens.extend(texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                texts = uncond_tokens
            cond_emb = self.text_encoder(texts)
        elif self.condition in ['action']:
            cond_emb = batch['action']
            if self.do_classifier_free_guidance:
                cond_emb = torch.cat(
                    cond_emb,
                    torch.zeros_like(batch['action'],
                                     dtype=batch['action'].dtype))
        else:
            raise TypeError(f"condition type {self.condition} not supported")

        # diffusion reverse
        with torch.no_grad():
            z = self._diffusion_reverse(cond_emb, lengths)

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)
            else:
                raise TypeError("vae_type must be mcross or actor or mld")

        joints_rst = self.feats2joints(feats_rst)

        rs_set = {
            "m_rst": feats_rst,
            # [bs, ntoken, nfeats]<= [ntoken, bs, nfeats]
            "lat_t": z.permute(1, 0, 2),
            "joints_rst": joints_rst,
        }

        # prepare gt/refer for metric
        if "motion" in batch.keys() and not finetune_decoder:
            feats_ref = batch["motion"].detach()
            with torch.no_grad():
                if self.vae_type in ["mld", "vposert", "actor"]:
                    motion_z, dist_m = self.vae.encode(feats_ref, lengths)
                    recons_z, dist_rm = self.vae.encode(feats_rst, lengths)
                elif self.vae_type == "no":
                    motion_z = feats_ref
                    recons_z = feats_rst

            joints_ref = self.feats2joints(feats_ref)

            rs_set["m_ref"] = feats_ref
            rs_set["lat_m"] = motion_z.permute(1, 0, 2)
            rs_set["lat_rm"] = recons_z.permute(1, 0, 2)
            rs_set["joints_ref"] = joints_ref
        return rs_set


    def recover_from_ric2(self,data, joints_num):
        r_rot_quat, r_pos = recover_root_rot_pos(data)

        positions = data[..., 4:(joints_num - 1) * 3 + 4]
        positions = positions.view(positions.shape[:-1] + (-1, 3))

        '''Add Y-axis rotation to local joints'''
        # positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

        '''Add root XZ to joints'''
        # positions[..., 0] += r_pos[..., 0:1]
        # positions[..., 2] += r_pos[..., 2:3]

        '''Concate root and joints'''
        positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

        return positions

    def recover_rot(self,data):
        # dataset [bs, seqlen, 263/251] HumanML/KIT

        bs,seqlen,_ = data.shape
        joints_num = 22 if data.shape[-1] == 263 else 21
        r_rot_quat, r_pos = recover_root_rot_pos(data)
        r_pos_pad = torch.cat([r_pos, torch.zeros_like(r_pos)], dim=-1).unsqueeze(-2)
        r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)
        start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
        end_indx = start_indx + (joints_num - 1) * 6
        cont6d_params = data[..., start_indx:end_indx]
        cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
        cont6d_params = cont6d_params.view(bs,-1, joints_num, 6)
        cont6d_params = torch.cat([cont6d_params, r_pos_pad], dim=-2)
        return cont6d_params

<<<<<<< HEAD
    def t2m_eval(self, batch,is_mm=False):
        texts = batch["text"]
        motions = batch["motion"].detach().clone().cuda()
        lengths = batch["length"]
        
        reference_motion = batch["reference_motion"].detach().clone().cuda()
        label = batch["label"][0].detach().clone().cuda()

        word_embs = batch["word_embs"].detach().clone().cuda()
        pos_ohot = batch["pos_ohot"].detach().clone().cuda()
        text_lengths = batch["text_len"].detach().clone().cuda()

        # start
        start = time.time()
        if is_mm:
=======
    def t2m_eval(self, batch):
        texts = batch["text"]
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]
        
        reference_motion = batch["reference_motion"].detach().clone()
        label = batch["label"][0].detach().clone()

        # start
        start = time.time()
        if self.trainer.datamodule.is_mm:
>>>>>>> 045ca9590646d12c0e3a4de1ddbe6f8e20e4262c
            texts = texts * self.cfg.TEST.MM_NUM_REPEATS
            motions = motions.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                dim=0)
            lengths = lengths * self.cfg.TEST.MM_NUM_REPEATS

            if self.is_test_walk == False:
                word_embs = word_embs.repeat_interleave(
                    self.cfg.TEST.MM_NUM_REPEATS, dim=0)
                pos_ohot = pos_ohot.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,dim=0)
            
                text_lengths = text_lengths.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)

        if self.stage in ['diffusion', 'vae_diffusion',"cycle_diffusion"]:

            if self.guidance_mode == 'v0':
                uncond_tokens = [""] * len(texts)
                if self.condition == 'text':
                    uncond_tokens.extend(texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                texts = uncond_tokens
                text_emb = self.text_encoder(texts)
<<<<<<< HEAD
                z = self._diffusion_reverse(text_emb, lengths,reference_motion,mode = "v0")
=======
                z = self._diffusion_reverse(text_emb, lengths,reference_motion)
>>>>>>> 045ca9590646d12c0e3a4de1ddbe6f8e20e4262c
            
            elif self.guidance_mode == 'v4':
                batch_size = len(texts)
                reference_motion = reference_motion.expand(batch_size,-1,-1)
                empty_ref = torch.zeros_like(reference_motion)

                reference_motion = torch.cat((empty_ref,empty_ref,reference_motion),dim=0)
                    # uncond_tokens = [texts[i] for i in range(len(texts))]
                uncond_tokens = [""] * len(texts)
                uncond_tokens2 = [""] * len(texts)

                if self.condition == 'text':
                    uncond_tokens.extend(texts)
                    uncond_tokens.extend(uncond_tokens2)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(texts)
                texts = uncond_tokens
                text_emb = self.text_encoder(texts)

<<<<<<< HEAD
                z = self._diffusion_reverse(text_emb, lengths,reference_motion,mode = "v4")

            elif self.guidance_mode == 'v2':
                    batch = len(texts)
                    reference_motion = reference_motion.expand(batch,-1,-1)
                    empty_ref = torch.zeros_like(reference_motion)

                    reference_motion = torch.cat((empty_ref,empty_ref,reference_motion),dim=0).cuda()
                    # uncond_tokens = [texts[i] for i in range(len(texts))]
                    uncond_tokens = [""] * len(texts)
                    uncond_tokens2 = [""] * len(texts)
                    if self.condition == 'text':
                        uncond_tokens.extend(texts)
                        uncond_tokens.extend(uncond_tokens2)

                    elif self.condition == 'text_uncond':
                        uncond_tokens.extend(uncond_tokens)
                    texts = uncond_tokens
                    text_emb = self.text_encoder(texts)
     
                    z = self._diffusion_reverse(text_emb, lengths,reference_motion,mode = "v2")
=======
                z = self._diffusion_reverse(text_emb, lengths,reference_motio,is_v4 = True)

>>>>>>> 045ca9590646d12c0e3a4de1ddbe6f8e20e4262c

        elif self.stage in ['vae']:
            if self.vae_type in ["mld", "vposert", "actor"]:
                z, dist_m = self.vae.encode(motions, lengths)
            else:
                raise TypeError("Not supported vae type!")
            if self.condition in ['text_uncond']:
                # uncond random sample
                z = torch.randn_like(z)

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)

        # end time
        end = time.time()
        inference_time = torch.tensor(end - start)
        print("inference_time:::",inference_time)
        self.times.append(end - start)

        logits = self.style_function(feats_rst)

        # print("score::", score)
        probabilities = F.softmax(logits, dim=1)
        predicted = torch.argmax(probabilities,dim=1)#.item()
        batch_size = probabilities.shape[0]

        if batch_size == 1:
            predicted = predicted.item()
        else:
            predicted = predicted[0].item()
        predicted = torch.tensor(predicted).cuda()

        motion_name = self.label_to_motion[str(predicted.cpu().numpy())]
        base_name = self.label_to_motion[str(label.cpu().numpy())]
        print("name:%s -> %s" % (base_name,motion_name))
        
        # joints recover
        joints_rst = self.feats2joints(feats_rst)
        joints_ref = self.feats2joints(motions)

        # renorm for t2m evaluators
        feats_rst = self.datamodule.renorm4t2m(feats_rst)
        motions = self.datamodule.renorm4t2m(motions)

        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=motions.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        feats_rst = feats_rst[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens,
                           self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")

        recons_mov = self.t2m_moveencoder(feats_rst[..., :-4]).detach()
        recons_emb = self.t2m_motionencoder(recons_mov, m_lens)
        motion_mov = self.t2m_moveencoder(motions[..., :-4]).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        # if self.is_test_walk == False:
        text_emb = self.t2m_textencoder(word_embs, pos_ohot,
                                    text_lengths)[align_idx]

        rs_set = {
            "m_ref": motions,
            "m_rst": feats_rst,
            "lat_t": text_emb,
            "lat_m": motion_emb,
            "lat_rm": recons_emb,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "predicted": logits,
            "label": label,
            "inference_time":inference_time
        }
       
        return rs_set




    def t2m_eval_test(self, batch,is_mm=False):
        texts = batch["text"]
        motions = batch["motion"].detach().clone().cuda()
        lengths = batch["length"]
        
        reference_motion = batch["reference_motion"].detach().clone().cuda()
        label = batch["label"][0].detach().clone().cuda()

        # if self.is_test_walk == False:
        word_embs = batch["word_embs"].detach().clone().cuda()
        pos_ohot = batch["pos_ohot"].detach().clone().cuda()
        text_lengths = batch["text_len"].detach().clone().cuda()

        # start
        start = time.time()
        #self.style_function
        # if self.trainer.datamodule.is_mm:
        if is_mm:
            texts = texts * self.cfg.TEST.MM_NUM_REPEATS
            motions = motions.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                dim=0)
            lengths = lengths * self.cfg.TEST.MM_NUM_REPEATS

            if self.is_test_walk == False:
                word_embs = word_embs.repeat_interleave(
                    self.cfg.TEST.MM_NUM_REPEATS, dim=0)
                pos_ohot = pos_ohot.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,dim=0)
            
                text_lengths = text_lengths.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)

        if self.stage in ['diffusion', 'vae_diffusion',"cycle_diffusion"]:
      
            if self.guidance_mode == 'v0':
                uncond_tokens = [""] * len(texts)
                if self.condition == 'text':
                    uncond_tokens.extend(texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                texts = uncond_tokens
                text_emb = self.text_encoder(texts)
<<<<<<< HEAD
                z = self._diffusion_reverse(text_emb, lengths,reference_motion,mode = "v0")
=======
                z = self._diffusion_reverse(text_emb, lengths,reference_motion)
>>>>>>> 045ca9590646d12c0e3a4de1ddbe6f8e20e4262c
            
            elif self.guidance_mode == 'v4':
                batch_size = len(texts)
                reference_motion = reference_motion.expand(batch_size,-1,-1)
                empty_ref = torch.zeros_like(reference_motion)

                reference_motion = torch.cat((empty_ref,empty_ref,reference_motion),dim=0)
                uncond_tokens = [""] * len(texts)
                uncond_tokens2 = [""] * len(texts)
                if self.condition == 'text':
                    uncond_tokens.extend(texts)
                    uncond_tokens.extend(uncond_tokens2)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(texts)
                texts = uncond_tokens
                text_emb = self.text_encoder(texts)
     
<<<<<<< HEAD
                z = self._diffusion_reverse(text_emb, lengths,reference_motion,mode = "v4")

            elif self.guidance_mode == 'v2':
                    batch = len(texts)
                    reference_motion = reference_motion.expand(batch,-1,-1)
                    empty_ref = torch.zeros_like(reference_motion)

                    reference_motion = torch.cat((empty_ref,empty_ref,reference_motion),dim=0).cuda()
                    # uncond_tokens = [texts[i] for i in range(len(texts))]
                    uncond_tokens = [""] * len(texts)
                    uncond_tokens2 = [""] * len(texts)
                    if self.condition == 'text':
                        uncond_tokens.extend(texts)
                        uncond_tokens.extend(uncond_tokens2)

                    elif self.condition == 'text_uncond':
                        uncond_tokens.extend(uncond_tokens)
                    texts = uncond_tokens
                    text_emb = self.text_encoder(texts)
     
                    z = self._diffusion_reverse(text_emb, lengths,reference_motion,mode = "v2")
=======
                z = self._diffusion_reverse(text_emb, lengths,reference_motion,is_v4 = True)
        
>>>>>>> 045ca9590646d12c0e3a4de1ddbe6f8e20e4262c
        elif self.stage in ['vae']:
            if self.vae_type in ["mld", "vposert", "actor"]:
                z, dist_m = self.vae.encode(motions, lengths)
            else:
                raise TypeError("Not supported vae type!")
            if self.condition in ['text_uncond']:
                # uncond random sample
                z = torch.randn_like(z)

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)

        # end time
        end = time.time()
        inference_time = torch.tensor(end - start)
        self.times.append(end - start)

       
        logits = self.style_function(feats_rst)

        probabilities = F.softmax(logits, dim=1)
        predicted = torch.argmax(probabilities,dim=1)
        batch_size = probabilities.shape[0]

        if batch_size == 1:
            predicted = predicted.item()
        else:
            predicted = predicted[0].item()
        predicted = torch.tensor(predicted).cuda()

        motion_name = self.label_to_motion[str(predicted.cpu().numpy())]
        base_name = self.label_to_motion[str(label.cpu().numpy())]
        print("name:%s -> %s" % (base_name,motion_name))
        
        # joints recover
        joints_rst = self.feats2joints(feats_rst)
        joints_ref = self.feats2joints(motions)

        # renorm for t2m evaluators
        feats_rst = self.datamodule.renorm4t2m(feats_rst)
        motions = self.datamodule.renorm4t2m(motions)

        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=motions.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        feats_rst = feats_rst[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens,
                           self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")

        recons_mov = self.t2m_moveencoder(feats_rst[..., :-4]).detach()
        recons_emb = self.t2m_motionencoder(recons_mov, m_lens)
        motion_mov = self.t2m_moveencoder(motions[..., :-4]).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
<<<<<<< HEAD
        # if self.is_test_walk == False:
        text_emb = self.t2m_textencoder(word_embs, pos_ohot,
                                    text_lengths)[align_idx]
=======
        if self.is_test_walk == False:
            text_emb = self.t2m_textencoder(word_embs, pos_ohot,
                                        text_lengths)[align_idx]
>>>>>>> 045ca9590646d12c0e3a4de1ddbe6f8e20e4262c

        rs_set = {
            "m_ref": motions,
            "m_rst": feats_rst,
            "lat_t": text_emb,
            "lat_m": motion_emb,
            "lat_rm": recons_emb,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "predicted": logits,
            "label": label,
            "inference_time":inference_time
        }
       
        return rs_set
    
   
  
    def a2m_eval(self, batch):
        actions = batch["action"]
        actiontexts = batch["action_text"]
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]

        if self.do_classifier_free_guidance:
            cond_emb = torch.cat((torch.zeros_like(actions), actions))

        if self.stage in ['diffusion', 'vae_diffusion']:
            z = self._diffusion_reverse(cond_emb, lengths)
        elif self.stage in ['vae']:
            if self.vae_type in ["mld", "vposert","actor"]:
                z, dist_m = self.vae.encode(motions, lengths)
            else:
                raise TypeError("vae_type must be mcross or actor")

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert","actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)
            else:
                raise TypeError("vae_type must be mcross or actor or mld")

        mask = batch["mask"]
        joints_rst = self.feats2joints(feats_rst, mask)
        joints_ref = self.feats2joints(motions, mask)
        joints_eval_rst = self.feats2joints_eval(feats_rst, mask)
        joints_eval_ref = self.feats2joints_eval(motions, mask)

        rs_set = {
            "m_action": actions,
            "m_ref": motions,
            "m_rst": feats_rst,
            "m_lens": lengths,
            "joints_rst": joints_rst,
            "joints_ref": joints_ref,
            "joints_eval_rst": joints_eval_rst,
            "joints_eval_ref": joints_eval_ref,
        }
        return rs_set

    def a2m_gt(self, batch):
        actions = batch["action"]
        actiontexts = batch["action_text"]
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]
        mask = batch["mask"]

        joints_ref = self.feats2joints(motions.to('cuda'), mask.to('cuda'))

        rs_set = {
            "m_action": actions,
            "m_text": actiontexts,
            "m_ref": motions,
            "m_lens": lengths,
            "joints_ref": joints_ref,
        }
        return rs_set

    def eval_gt(self, batch, renoem=True):
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]

        # feats_rst = self.datamodule.renorm4t2m(feats_rst)
        if renoem:
            motions = self.datamodule.renorm4t2m(motions)

        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=motions.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens,
                           self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")

        word_embs = batch["word_embs"].detach()
        pos_ohot = batch["pos_ohot"].detach()
        text_lengths = batch["text_len"].detach()

        motion_mov = self.t2m_moveencoder(motions[..., :-4]).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        text_emb = self.t2m_textencoder(word_embs, pos_ohot,
                                        text_lengths)[align_idx]

        # joints recover
        joints_ref = self.feats2joints(motions)

        rs_set = {
            "m_ref": motions,
            "lat_t": text_emb,
            "lat_m": motion_emb,
            "joints_ref": joints_ref,
        }
        return rs_set

<<<<<<< HEAD
    def allsplit_step(self, split: str, batch, batch_idx,is_mm = False):
=======
    def allsplit_step(self, split: str, batch, batch_idx):
>>>>>>> 045ca9590646d12c0e3a4de1ddbe6f8e20e4262c
        if self.is_test == False:
            if split in ["train","val"]:
                if self.stage == "vae":
                    rs_set = self.train_vae_forward(batch)
                    rs_set["lat_t"] = rs_set["lat_m"]
                elif self.stage == "diffusion":
                    rs_set = self.train_diffusion_forward(batch)
                
                elif self.stage == "cycle_diffusion":
                    rs_set = self.train_cycle_diffusion_forward(batch)
                
                elif self.stage == "vae_diffusion":
                    vae_rs_set = self.train_vae_forward(batch)
                    diff_rs_set = self.train_diffusion_forward(batch)
                    t2m_rs_set = self.test_diffusion_forward(batch,
                                                            finetune_decoder=True)
                    # merge results
                    rs_set = {
                        **vae_rs_set,
                        **diff_rs_set,
                        "gen_m_rst": t2m_rs_set["m_rst"],
                        "gen_joints_rst": t2m_rs_set["joints_rst"],
                        "lat_t": t2m_rs_set["lat_t"],
                    }
                else:
                    raise ValueError(f"Not support this stage {self.stage}!")

                loss = self.losses[split].update(rs_set)
                if loss is None:
                    raise ValueError(
                        "Loss is None, this happend with torchmetrics > 0.7")

            # Compute the metrics - currently evaluate results from text to motion

        if self.is_cmld == False or self.is_test == True:
        # if self.is_test == True:
            if split in ["val", "test"]:
                if self.condition in ['text', 'text_uncond']:
                    # use t2m evaluators
                    # if self.is_puzzle == False:
                    rs_set = self.t2m_eval(batch)
                 
                elif self.condition == 'action':
                    # use a2m evaluators
                    rs_set = self.a2m_eval(batch)
                # MultiModality evaluation sperately
<<<<<<< HEAD
                if is_mm:
=======
                if self.trainer.datamodule.is_mm:
>>>>>>> 045ca9590646d12c0e3a4de1ddbe6f8e20e4262c
                    metrics_dicts = ['MMMetrics']
                else:
                    metrics_dicts = self.metrics_dict

                for metric in metrics_dicts:
                    if metric == "TemosMetric":
                        phase = split if split != "val" else "eval"
                        if eval(f"self.cfg.{phase.upper()}.DATASETS")[0].lower(
                        ) not in [
                                "humanml3d",
                                "kit",
                        ]:
                            raise TypeError(
                                "APE and AVE metrics only support humanml3d and kit datasets now"
                            )

                        getattr(self, metric).update(rs_set["joints_rst"],
                                                    rs_set["joints_ref"],
                                                    batch["length"])
                    elif metric == "TM2TMetrics":
                        getattr(self, metric).update(
                            # lat_t, latent encoded from diffusion-based text
                            # lat_rm, latent encoded from reconstructed motion
                            # lat_m, latent encoded from gt motion
                            # rs_set['lat_t'], rs_set['lat_rm'], rs_set['lat_m'], batch["length"])
                            rs_set["lat_t"],
                            rs_set["lat_rm"],
                            rs_set["lat_m"],
                            batch["length"],
                            rs_set["predicted"],
                            rs_set["label"],
                       
                            rs_set["joints_rst"],
                            rs_set["inference_time"],
                        )
                
                        # )
                    elif metric == "UncondMetrics":
                        getattr(self, metric).update(
                            recmotion_embeddings=rs_set["lat_rm"],
                            gtmotion_embeddings=rs_set["lat_m"],
                            lengths=batch["length"],
                        )
                    elif metric == "MRMetrics":
                        getattr(self, metric).update(rs_set["joints_rst"],
                                                    rs_set["joints_ref"],
                                                    batch["length"])
                    elif metric == "MMMetrics":
                        getattr(self, metric).update(rs_set["lat_rm"].unsqueeze(0),
                                                    batch["length"])
                   
                    else:
                        raise TypeError(f"Not support this metric {metric}")

            # return forward output rather than loss during test
            if split in ["val","test"]:
                return rs_set["joints_rst"], batch["length"]
        return loss

    def allsplit_step_test(self, split: str, batch, batch_idx,is_mm = False,is_mst = False):
        if self.is_cmld == False or self.is_test == True:
            if split in ["val", "test"]:
                if self.condition in ['text', 'text_uncond']:
                    # use t2m evaluators
                    # if is_mst and self.is_puzzle == False:
                    #     rs_set = self.t2m_eval_mst(batch,is_mm)
                    # elif is_mst and self.is_puzzle:
                    #     rs_set = self.t2m_eval_puzzle_mst(batch)
                    # else:
                    rs_set = self.t2m_eval_test(batch,is_mm)
                # MultiModality evaluation sperately
                if is_mm:
                    metrics_dicts = ['MMMetrics']
                else:
                    metrics_dicts = self.metrics_dict
                for metric in metrics_dicts:
                    if metric == "TemosMetric":
                        phase = split if split != "val" else "eval"
                        if eval(f"self.cfg.{phase.upper()}.DATASETS")[0].lower(
                        ) not in [
                                "humanml3d",
                                "kit",
                        ]:
                            raise TypeError(
                                "APE and AVE metrics only support humanml3d and kit datasets now"
                            )

                        getattr(self, metric).update(rs_set["joints_rst"],
                                                    rs_set["joints_ref"],
                                                    batch["length"])
                    elif metric == "TM2TMetrics":
                        getattr(self, metric).update(
                            # lat_t, latent encoded from diffusion-based text
                            # lat_rm, latent encoded from reconstructed motion
                            # lat_m, latent encoded from gt motion
                            # rs_set['lat_t'], rs_set['lat_rm'], rs_set['lat_m'], batch["length"])
                            rs_set["lat_t"],
                            rs_set["lat_rm"],
                            rs_set["lat_m"],
                            batch["length"],
                            rs_set["predicted"],
                            rs_set["label"],

                            rs_set["joints_rst"],
                            rs_set["inference_time"],

                        )
                  
                    elif metric == "MRMetrics":
                        getattr(self, metric).update(rs_set["joints_rst"],
                                                    rs_set["joints_ref"],
                                                    batch["length"])
                    elif metric == "MMMetrics":
                        
                        getattr(self, metric).update(rs_set["lat_rm"].unsqueeze(0),
                                                    batch["length"])
                    elif metric == "HUMANACTMetrics":
                        getattr(self, metric).update(rs_set["m_action"],
                                                    rs_set["joints_eval_rst"],
                                                    rs_set["joints_eval_ref"],
                                                    rs_set["m_lens"])
                    else:
                        raise TypeError(f"Not support this metric {metric}")

            # return forward output rather than loss during test
            if split in ["val","test"]:
                return rs_set["joints_rst"], batch["length"]
    
    def allsplit_epoch_end(self, split: str, outputs):
        dico = {}

        if split in ["train", "val"]:
            losses = self.losses[split]
            loss_dict = losses.compute(split)
            losses.reset()
            dico.update({
                losses.loss2logname(loss, split): value.item()
                for loss, value in loss_dict.items() if not torch.isnan(value)
            })

        if self.is_cmld == False or self.is_test == True:
            if split in ["val", "test"]:
                if self.trainer.datamodule.is_mm and "TM2TMetrics" in self.metrics_dict:
                    metrics_dicts = ['MMMetrics']
                else:
                    metrics_dicts = self.metrics_dict
                for metric in metrics_dicts:
                    metrics_dict = getattr(self, metric).compute(sanity_flag=self.trainer.sanity_checking)
                    # reset metrics
                    getattr(self, metric).reset()
                    dico.update({
                        f"Metrics/{metric}": value.item()
                        for metric, value in metrics_dict.items()
                    })
            if split != "test":
                dico.update({
                    "epoch": float(self.trainer.current_epoch),
                    "step": float(self.trainer.current_epoch),
                })
        
        # don't write sanity check into log
        if not self.trainer.sanity_checking:
            self.log_dict(dico, sync_dist=True, rank_zero_only=True)
    
    def allsplit_epoch_end_test(self, split: str,is_mm = False):
        dico = {}

        if split in ["train", "val"]:
            losses = self.losses[split]
            loss_dict = losses.compute(split)
            losses.reset()
            dico.update({
                losses.loss2logname(loss, split): value.item()
                for loss, value in loss_dict.items() if not torch.isnan(value)
            })

        if self.is_cmld == False or self.is_test == True:
            if split in ["val", "test"]:
           
                if is_mm == True and 'TM2TMetrics' in self.metrics_dict:
                    metrics_dicts = ['MMMetrics']
                else:
                    metrics_dicts = self.metrics_dict
                for metric in metrics_dicts:
                    metrics_dict = getattr(self, metric).compute(sanity_flag=False)
                    # reset metrics
                    getattr(self, metric).reset()
                    dico.update({
                        f"Metrics/{metric}": value.item()
                        for metric, value in metrics_dict.items()
                    })
            # if split != "test":
            #     dico.update({
            #         "epoch": float(self.trainer.current_epoch),
            #         "step": float(self.trainer.current_epoch),
            #     })
        
        # don't write sanity check into log
        self.log_dict(dico, sync_dist=True, rank_zero_only=True)

        return dico