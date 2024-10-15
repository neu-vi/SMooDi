import numpy as np
import torch
import torch.nn as nn
from torchmetrics import Metric
import torch.nn.functional as F
from mld.data.humanml.scripts.motion_process import (qrot,
                                                     recover_root_rot_pos)

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, input_features, target_features):
        cosine_sim = F.cosine_similarity(input_features, target_features)
        loss = 1 - cosine_sim.mean()  
        return loss
    
class InfoNCE_with_filtering:
    def __init__(self, temperature=0.7, threshold_selfsim=0.8):
        self.temperature = temperature
        self.threshold_selfsim = threshold_selfsim

    def get_sim_matrix(self, x, y):
        x_logits = torch.nn.functional.normalize(x, dim=-1)
        y_logits = torch.nn.functional.normalize(y, dim=-1)
        sim_matrix = x_logits @ y_logits.T
        return sim_matrix

    def __call__(self, x, y, sent_emb=None):
        bs, device = len(x), x.device
        sim_matrix = self.get_sim_matrix(x, y) / self.temperature

        if sent_emb is not None and self.threshold_selfsim:
            # put the threshold value between -1 and 1
            real_threshold_selfsim = 2 * self.threshold_selfsim - 1
            # Filtering too close values
            # mask them by putting -inf in the sim_matrix
            selfsim = sent_emb @ sent_emb.T
            selfsim_nodiag = selfsim - selfsim.diag().diag()
            idx = torch.where(selfsim_nodiag > real_threshold_selfsim)
            sim_matrix[idx] = -torch.inf

        labels = torch.arange(bs, device=device)

        total_loss = (
            F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.T, labels)
        ) / 2

        return total_loss

    def __repr__(self):
        return f"Constrastive(temp={self.temp})"


class MLDLosses(Metric):
    """
    MLD Loss
    """

    def __init__(self, vae, mode, cfg):
        super().__init__(dist_sync_on_step=cfg.LOSS.DIST_SYNC_ON_STEP)

        # Save parameters
        # self.vae = vae
        self.vae_type = cfg.TRAIN.ABLATION.VAE_TYPE
        self.mode = mode
        self.cfg = cfg
        self.predict_epsilon = cfg.TRAIN.ABLATION.PREDICT_EPSILON
        self.stage = cfg.TRAIN.STAGE
        self.is_regularizer = cfg.TRAIN.ABLATION.REGULARIZER
        self.is_cycle = cfg.TRAIN.ABLATION.CYCLE
        self.is_recon = cfg.TRAIN.ABLATION.RECON
        self.is_two_dataset = cfg.TRAIN.ABLATION.TWODATASET
<<<<<<< HEAD
        # self.is_tri = cfg.TRAIN.ABLATION.IS_TRI
        # self.is_info = cfg.TRAIN.ABLATION.IS_INFONCE
        # self.is_style_recon = cfg.TRAIN.ABLATION.STYLE_RECON
        # self.is_motion_clip = cfg.TRAIN.ABLATION.IS_MOTION_CLIP
        losses = []

        # if self.is_motion_clip:
        #     self.stage = "motion_clip"
=======
        self.is_tri = cfg.TRAIN.ABLATION.IS_TRI
        self.is_info = cfg.TRAIN.ABLATION.IS_INFONCE
        self.is_style_recon = cfg.TRAIN.ABLATION.STYLE_RECON
        self.is_motion_clip = cfg.TRAIN.ABLATION.IS_MOTION_CLIP
        losses = []

        if self.is_motion_clip:
            self.stage = "motion_clip"
>>>>>>> 045ca9590646d12c0e3a4de1ddbe6f8e20e4262c
        # diffusion loss
        if self.stage in ['diffusion', 'vae_diffusion']:
            # instance noise loss
            losses.append("inst_loss")
            losses.append("cycle_loss")
            losses.append("x_loss")

            if self.cfg.LOSS.LAMBDA_PRIOR != 0.0:
                # prior noise loss
                losses.append("prior_loss")

        if self.stage in ['vae', 'vae_diffusion']:
            # reconstruction loss
            losses.append("recons_feature")
            losses.append("recons_verts")
            losses.append("recons_joints")
            losses.append("recons_limb")

            losses.append("gen_feature")
            losses.append("gen_joints")

            # KL loss
            losses.append("kl_motion")

        if self.stage not in ['vae', 'diffusion', 'vae_diffusion','motion_clip']:
            raise ValueError(f"Stage {self.stage} not supported")

        losses.append("total")

        for loss in losses:
            self.add_state(loss,
                           default=torch.tensor(0.0),
                           dist_reduce_fx="sum")
            # self.register_buffer(loss, torch.tensor(0.0))
        self.add_state("count", torch.tensor(0), dist_reduce_fx="sum")
        self.losses = losses

        self._losses_func = {}
        self._params = {}
        for loss in losses:
            if loss.split('_')[0] == 'motion':
                self._losses_func[loss] = nn.MSELoss(reduction='mean')
                self._params[loss] = 1
            if loss.split('_')[0] == 'inst':
                self._losses_func[loss] = nn.MSELoss(reduction='mean')
                self._params[loss] = 1
            if loss.split('_')[0] == 'sim':
                self._losses_func[loss] = nn.CosineSimilarity(dim=1, eps=1e-6)
                self._params[loss] = 0.01
            elif loss.split('_')[0] == 'x':
                self._losses_func[loss] = nn.MSELoss(reduction='mean')
                self._params[loss] = 1
            elif loss.split('_')[0] == 'cycle':
                self._losses_func[loss] = nn.MSELoss(reduction='mean')
                self._params[loss] = 0.1
            elif loss.split('_')[0] == 'reconfeat':
                self._losses_func[loss] = torch.nn.SmoothL1Loss(
                    reduction='mean')
                self._params[loss] = 0.01
            elif loss.split('_')[0] == 'prior':
                self._losses_func[loss] = nn.MSELoss(reduction='mean')
                self._params[loss] = cfg.LOSS.LAMBDA_PRIOR
            if loss.split('_')[0] == 'kl':
                if cfg.LOSS.LAMBDA_KL != 0.0:
                    self._losses_func[loss] = KLLoss()
                    self._params[loss] = cfg.LOSS.LAMBDA_KL
            elif loss.split('_')[0] == 'recons':
                self._losses_func[loss] = torch.nn.SmoothL1Loss(
                    reduction='mean')
                self._params[loss] = cfg.LOSS.LAMBDA_REC
            elif loss.split('_')[0] == 'gen':
                self._losses_func[loss] = torch.nn.SmoothL1Loss(
                    reduction='mean')
                self._params[loss] = cfg.LOSS.LAMBDA_GEN
            elif loss.split('_')[0] == 'latent':
                self._losses_func[loss] = torch.nn.SmoothL1Loss(
                    reduction='mean')
                self._params[loss] = cfg.LOSS.LAMBDA_LATENT
            else:
                ValueError("This loss is not recognized.")
            if loss.split('_')[-1] == 'joints':
                self._params[loss] = cfg.LOSS.LAMBDA_JOINT

    def update(self, rs_set):
        total: float = 0.0
        # Compute the losses
        # Compute instance loss
        if self.stage in ["vae", "vae_diffusion"]:
            
            total += self._update_loss("recons_feature", rs_set['m_rst'],
                                       rs_set['m_ref'])
            total += self._update_loss("recons_joints", rs_set['joints_rst'],
                                       rs_set['joints_ref'])
            total += self._update_loss("kl_motion", rs_set['dist_m'], rs_set['dist_ref'])
        
        if self.stage in ["diffusion", "vae_diffusion","cycle_diffusion"]:
            # predict noise HERE!!!
            if self.predict_epsilon:
                total += self._update_loss("inst_loss", rs_set['noise_pred'],rs_set['noise'])

                if self.is_two_dataset:
                    total += self._update_loss("inst_loss", rs_set['noise_y'],rs_set['noise_pred_y'])
                         
                if self.is_cycle:
                    total += self._update_loss("cycle_loss", rs_set['noise_cycle_y'],rs_set['noise_pred_cycle_y'])
                    total += self._update_loss("cycle_loss", rs_set['noise_cycle'],rs_set['noise_pred_cycle'])

            else:
                total += self._update_loss("x_loss", rs_set['pred'],
                                           rs_set['latent'])

            if self.cfg.LOSS.LAMBDA_PRIOR != 0.0:
                # loss - prior loss 
                # LAMBDA_PRIOR = 0.0
                total += self._update_loss("prior_loss", rs_set['noise_prior'],
                                           rs_set['dist_m1'])

        if self.stage in ["vae_diffusion"]:
            # loss
            # noise+text_emb => diff_reverse => latent => decode => motion
            total += self._update_loss("gen_feature", rs_set['gen_m_rst'],
                                       rs_set['m_ref'])
            total += self._update_loss("gen_joints", rs_set['gen_joints_rst'],
                                       rs_set['joints_ref'])

        self.total += total.detach()
        self.count += 1

        return total

    def compute(self, split):
        count = getattr(self, "count")
        return {loss: getattr(self, loss) / count for loss in self.losses}

    def _update_loss2(self, loss: str, inputs1,inputs2,inputs3):
        # Update the loss
        val = self._losses_func[loss](inputs1, inputs2, inputs3)
        getattr(self, loss).__iadd__(val.detach())
        # Return a weighted sum
        weighted_loss = self._params[loss] * val
        return weighted_loss

    def _update_loss(self, loss: str, outputs, inputs):
        # Update the loss
        val = self._losses_func[loss](outputs, inputs)
        getattr(self, loss).__iadd__(val.detach())
        # Return a weighted sum
        weighted_loss = self._params[loss] * val
        return weighted_loss
    
    def loss2logname(self, loss: str, split: str):
        if loss == "total":
            log_name = f"{loss}/{split}"
        else:
            loss_type, name = loss.split("_")
            log_name = f"{loss_type}/{name}/{split}"
        return log_name

class KLLoss:

    def __init__(self):
        pass

    def __call__(self, q, p):
        div = torch.distributions.kl_divergence(q, p)
        return div.mean()

    def __repr__(self):
        return "KLLoss()"


class KLLossMulti:

    def __init__(self):
        self.klloss = KLLoss()

    def __call__(self, qlist, plist):
        return sum([self.klloss(q, p) for q, p in zip(qlist, plist)])

    def __repr__(self):
        return "KLLossMulti()"
