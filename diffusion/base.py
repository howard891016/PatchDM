"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

from model.unet_autoenc import AutoencReturn
from config_base import BaseConfig
import enum
import math
import random
import numpy as np
import torch as th
from model import *
import sys
from model.nn import mean_flat
from typing import NamedTuple, Tuple
from utils.choices import *
from torch.cuda.amp import autocast
from scipy import integrate
import torch.nn.functional as F

from dataclasses import dataclass

from einops import rearrange,repeat

# from models.utils import from_flattened_numpy, to_flattened_numpy


@dataclass
class GaussianDiffusionBeatGansConfig(BaseConfig):
    gen_type: GenerativeType
    betas: Tuple[float]
    model_type: ModelType
    model_mean_type: ModelMeanType
    model_var_type: ModelVarType
    loss_type: LossType
    rescale_timesteps: bool
    fp16: bool
    train_pred_xstart_detach: bool = True
    cfg: bool = True
    whole_patch: bool = False # (Howard add) whether use whole patch to train the model
    use_vae: bool = False # (Howard add) whether use vae to encode the image

    def make_sampler(self):
        return GaussianDiffusionBeatGans(self)


class GaussianDiffusionBeatGans:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """
    def __init__(self, conf: GaussianDiffusionBeatGansConfig):
        self.conf = conf
        self.model_mean_type = conf.model_mean_type
        self.model_var_type = conf.model_var_type
        self.loss_type = conf.loss_type
        self.rescale_timesteps = conf.rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(conf.betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps, )

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod -
                                                   1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (betas * (1.0 - self.alphas_cumprod_prev) /
                                   (1.0 - self.alphas_cumprod))
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:]))
        self.posterior_mean_coef1 = (betas *
                                     np.sqrt(self.alphas_cumprod_prev) /
                                     (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) *
                                     np.sqrt(alphas) /
                                     (1.0 - self.alphas_cumprod))

    def training_losses(self,
                        model: Model,
                        x_start: th.Tensor, # padded original img
                        imgs: th.Tensor, # original img: use_vae => 256x256 / not use_vae => 64x64
                        t: th.Tensor, # [0,1000]
                        pos: th.Tensor,
                        loss_mask: th.Tensor,
                        next_loss_mask: th.Tensor,
                        idx = None,
                        patch_size = 32,
                        model_kwargs=None,
                        noise: th.Tensor = None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [(B*P*P) x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        
        # print("patch size: " + str(patch_size))
        # print("patch size: " + str(patch_size))

        halfp = patch_size // 2
        # print(t.shape)
        # print("img shape: " + str(imgs.shape))
        # # print(patch_size)
        T = 1.
        eps = 1e-3
        # t = th.rand((x_start.shape[0],), device=t.device) * (T - eps) + eps
        t_cur = repeat(t, 'h -> (h repeat)',repeat =int(x_start.shape[0]/t.shape[0]))
        
        # print("t_cur: " + str(t_cur))
        # print(t_cur/1000)
        # print(1-(t_cur/1000+eps))
        # Modified: Find interpolation between clean data and noise
        x_t = self.q_sample(x_start, t_cur, noise=noise)
        # x_t = th.einsum('b,bijk->bijk', ((t_cur/1000 + eps)), x_start) + th.einsum('b,bijk->bijk', 1 - (t_cur/1000 + eps), noise)
        # print("x_t: ")
        # print(x_t)



        # # 輸出結果
        # print(f"Pred (shift) - Min: {pred_min}, Max: {pred_max}")

        if loss_mask is not None:
            x_t = x_t * loss_mask

        terms = {'x_t': x_t}

        if self.conf.whole_patch:
            index = [0, 0]
            index = th.tensor(index, device = 'cuda')
            H, W = imgs.shape[2:]                       # H  = 256 ;  W = 256
            patch_num_x = H // patch_size               # 256 // 64  = 4
            patch_num_y = W // patch_size               # 256 // 64  = 4
            pos = pos.flatten(0,1).repeat(idx.shape[0], 1)                                                                      # pos.shape = [batch_size x (patch_num_x+1) x (patch_num_y+1), 2]
            x_t = rearrange(x_t, 'b c (p1 h) (p2 w) -> (b p1 p2) c h w', h = patch_size, w = patch_size)                        # x_t.shape = [batch_size x (patch_num_x+1) x (patch_num_y+1), 3, patch_size, patch_size]
            noise = rearrange(noise, 'b c (p1 h) (p2 w) -> (b p1 p2) c h w', h = patch_size, w = patch_size)                    # noise.shape = [batch_size x (patch_num_x+1) x (patch_num_y+1), 3, 64, 64]
            loss_mask = rearrange(loss_mask, 'b c (p1 h) (p2 w) -> (b p1 p2) c h w', h = patch_size, w = patch_size)            # loss_mask.shape = [batch_size x (patch_num_x+1) x (patch_num_y+1), 3, 64, 64]
        else:
            index_x = random.randrange(pos.shape[0]-1)
            index_y = random.randrange(pos.shape[1]-1)
            index = [index_x, index_y]
            index = th.tensor(index, device = 'cuda')
            pos = pos[index_x:index_x+2, index_y:index_y+2].flatten(0,1).repeat(idx.shape[0], 1)
            x_t = x_t[:,:,index_x*patch_size:(index_x+2)*patch_size, index_y*patch_size: (index_y+2)*patch_size]
            noise = noise[:,:,index_x*patch_size:(index_x+2)*patch_size, index_y*patch_size: (index_y+2)*patch_size]
            x_start = x_start[:,:,index_x*patch_size:(index_x+2)*patch_size, index_y*patch_size: (index_y+2)*patch_size]
            loss_mask = loss_mask[:,:,index_x*patch_size:(index_x+2)*patch_size, index_y*patch_size: (index_y+2)*patch_size]

            x_t = rearrange(x_t, 'b c (p1 h) (p2 w) -> (b p1 p2) c h w', h = patch_size, w = patch_size)
            noise = rearrange(noise, 'b c (p1 h) (p2 w) -> (b p1 p2) c h w', h = patch_size, w = patch_size)
            x_start = rearrange(x_start, 'b c (p1 h) (p2 w) -> (b p1 p2) c h w', h = patch_size, w = patch_size)
            loss_mask = rearrange(loss_mask, 'b c (p1 h) (p2 w) -> (b p1 p2) c h w', h = patch_size, w = patch_size)


        if self.conf.cfg:
            s_random = th.tensor(np.random.random(t.shape[0])).to(t.device)
            pos_random = th.tensor(np.random.random(t.shape[0])).to(t.device)
        else:
            s_random = None
            pos_random = None
        if self.loss_type in [
                LossType.mse,
                LossType.l1,
        ]:
            with autocast(self.conf.fp16):
                # max_val = th.max(self._scale_timesteps(t))
                # min_val = th.min(self._scale_timesteps(t))
                # print(f"t Max value: {max_val.item()}, t Min value: {min_val.item()}")

                # x_t is static wrt. to the diffusion process
                model_forward = model.forward(x=x_t.detach(),
                                              t=self._scale_timesteps(t),
                                              pos=pos.detach(),
                                              imgs=imgs.detach(),
                                              idx = idx.detach(),
                                              index = index,
                                              do_train= True,
                                              patch_size = patch_size,
                                              random = s_random,
                                              pos_random = pos_random,
                                              use_vae=self.conf.use_vae,
                                              **model_kwargs)
            model_output_shift = model_forward.pred
            model_output_no_shift = model_forward.pred2
            
           


            assert model_output_shift.size(0) != noise.size(0)
            if loss_mask is None: # current is 4*4
                noise_ori = rearrange(noise, '(b p1 p2) c h w -> b c (p1 h) (p2 w)', p1 = 2, p2 = 2)
                noise_ori_pad = F.pad(noise_ori, (halfp, halfp, halfp, halfp), 'constant')
                noise_target_shift = rearrange(noise_ori_pad, 'b c (p1 h) (p2 w) -> (b p1 p2) c h w', h = patch_size, w = patch_size)
            else:
                if self.conf.whole_patch:
                    noise_ori = rearrange(noise, '(b p1 p2) c h w -> b c (p1 h) (p2 w)', p1 = patch_num_x+1, p2 = patch_num_y+1)            # noise_ori.shape = [batch_size, 3, image_height + patch_size, image_width + patch_size]
                    noise_ori_crop = noise_ori[:, :, halfp:-halfp, halfp:-halfp]                                                            # noise_ori_crop.shape = [batch_size, 3, image_height, image_width]
                    noise_target_shift = rearrange(noise_ori_crop, 'b c (p1 h) (p2 w) -> (b p1 p2) c h w', h = patch_size, w = patch_size)  # noise_target_shift.shape = [batch_size x patch_num_x x patch_num_y, 3, patch_size, patch_size]
                else:
                    noise_ori = rearrange(noise, '(b p1 p2) c h w -> b c (p1 h) (p2 w)', p1 = 2, p2 = 2)                                    # noise_ori.shape = (batch_size, 3, patch_size x 2, patch_size x 2)
                    noise_ori_crop = noise_ori[:, :, halfp:-halfp, halfp:-halfp]                                                            # noise_ori_crop.shape = (batch_size, 3, patch_size, patch_size)
                    noise_target_shift = rearrange(noise_ori_crop, 'b c (p1 h) (p2 w) -> (b p1 p2) c h w', h = patch_size, w = patch_size)  # noise_target_shift.shape = (batch_size, 3, patch_size, patch_size)
            
            noise_target_no_shift = noise
            # x_start_no_shift = x_start

            # Modified: Change the target
            target_types = {
                # ModelMeanType.eps: {"shift": noise_target_shift - x_start_target_shift, "no_shift": noise_target_no_shift - x_start_no_shift},
                # ModelMeanType.eps: {"shift": x_start_target_shift - noise_target_shift, "no_shift": x_start_no_shift - noise_target_no_shift},
                ModelMeanType.eps: {"shift": noise_target_shift, "no_shift":noise_target_no_shift},
            }
            target = target_types[self.model_mean_type]
            assert model_output_shift.shape == target["shift"].shape 
            assert model_output_no_shift.shape == target["no_shift"].shape

            # x_start max = 1.
            # x_start min = -1.
            # noise max = 4.~5.
            # noise min = -4.~-5.
            
            # lpips_fn = lpips.LPIPS(net='vgg').to(x_t.device)
            
            # print(f"LossType: {self.loss_type}")
            if self.loss_type == LossType.mse:
                if self.model_mean_type == ModelMeanType.eps:
                    # (n, c, h, w) => (n, )
                    terms["mse"] = mean_flat((target["shift"] - model_output_shift)**2).mean()
                    terms["mse"] += mean_flat((target["no_shift"] - model_output_no_shift)**2 * loss_mask).mean()
                else:
                    raise NotImplementedError()
            elif self.loss_type == LossType.l1:
                # (n, c, h, w) => (n, )
                terms["mse"] = mean_flat((target["shift"] - model_output_shift).abs()).mean()
                terms["mse"] += mean_flat((target["no_shift"] - model_output_no_shift).abs() * loss_mask).mean()
            else:
                raise NotImplementedError()

            if "vb" in terms:
                # if learning the variance also use the vlb loss
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
            
        else:
            raise NotImplementedError(self.loss_type)

        return terms
    
    def training_latent_losses(self,
                        model: Model,
                        x_start: th.Tensor,
                        t: th.Tensor,
                        model_kwargs=None,
                        noise: th.Tensor = None):
        """
        Compute training losses for a single timestep.
        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        
        # Modified: Find interpolation between clean data and noise
        x_t = self.q_sample(x_start, t, noise=noise)
        # x_t = th.einsum('b,bijk->bijk', t, x_start) + th.einsum('b,bijk->bijk', (1 - t), noise)
        # x_t = (t / 1000) * x_start + (1 - (t / 1000)) * noise
        # print(t/1000)
        # print(1 - t/1000)
        
        terms = {'x_t': x_t}

        if self.loss_type in [
                LossType.mse,
                LossType.l1,
        ]:
            with autocast(self.conf.fp16):
                # x_t is static wrt. to the diffusion process
                model_forward = model.forward(x=x_t.detach(),
                                              t=self._scale_timesteps(t),
                                              x_start=x_start.detach(),
                                              **model_kwargs)
            model_output = model_forward.pred

            # Modified: Change the target
            target_types = {
                ModelMeanType.eps: noise,
                # ModelMeanType.eps: x_start - noise,
                # ModelMeanType.eps: noise - x_start,
            }
            target = target_types[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape

            if self.loss_type == LossType.mse:
                if self.model_mean_type == ModelMeanType.eps:
                    # (n, c, h, w) => (n, )
                    terms["mse"] = mean_flat((target - model_output)**2)
                else:
                    raise NotImplementedError()
            elif self.loss_type == LossType.l1:
                # (n, c, h, w) => (n, )
                terms["mse"] = mean_flat((target - model_output).abs())
            else:
                raise NotImplementedError()

            if "vb" in terms:
                # if learning the variance also use the vlb loss
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms
    
    def sample(self,
               model: Model,
               shape=None,
               noise=None,
               all_pos=None,
               cond=None,
               x_start=None,
               imgs=None,
               clip_denoised=True,
               idx = None,
               patch_size = 64,
               model_kwargs=None,
               progress=False):
        """
        Args:
            x_start: given for the autoencoder
        """
        if model_kwargs is None:
            model_kwargs = {}
            if self.conf.model_type.has_autoenc(): # True
                print("has_autoenc")
                model_kwargs['x_start'] = x_start
                model_kwargs['imgs'] = imgs
                model_kwargs['cond'] = cond
                if cond is None: # here
                    print("cond is None in sample.")
                else:
                    print("cond is not None in sample.")

        # log_sample => ddpm
        if self.conf.gen_type == GenerativeType.ddpm: 
            # img = self.ddim_sample_loop(model,
            #                              shape=shape,
            #                              noise=noise,
            #                              pos=all_pos,
            #                              idx=idx, # New added
            #                              clip_denoised=clip_denoised,
            #                              model_kwargs=model_kwargs,
            #                              progress=progress)
            # return img
            return self.p_sample_loop(model,
                                      shape=shape,
                                      noise=noise,
                                      all_pos=all_pos,
                                      clip_denoised=clip_denoised,
                                      idx = idx,
                                      patch_size = patch_size,
                                      model_kwargs=model_kwargs,
                                      progress=progress)
        elif self.conf.gen_type == GenerativeType.ddim:
            if len(noise.shape) == 2:
                print("use latent sample")
                img = self.ddim_latent_sample_loop(model,
                                         shape=shape,
                                         noise=noise,
                                         clip_denoised=clip_denoised,
                                         model_kwargs=model_kwargs,
                                         progress=progress)
                # print(img)
                return img 
            else:
                print("use ddim sample")
                img = self.ddim_sample_loop(model,
                                         shape=shape,
                                         noise=noise,
                                         pos=all_pos,
                                         idx=idx, # New added
                                         clip_denoised=clip_denoised,
                                         model_kwargs=model_kwargs,
                                         progress=progress)
                return img
        else:
            raise NotImplementedError()

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) *
            x_start)
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t,
                                        x_start.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod,
                                            t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) *
            x_start + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod,
                                           t, x_start.shape) * noise)

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) *
            x_start +
            _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) *
            x_t)
        posterior_variance = _extract_into_tensor(self.posterior_variance, t,
                                                  x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] ==
                posterior_log_variance_clipped.shape[0] == x_start.shape[0])
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self,
                        model: Model,
                        x,
                        t,
                        shapes = (1,3,256,256),
                        clip_denoised=True,
                        denoised_fn=None,
                        idx = None,
                        patch_size = 64,
                        model_kwargs=None):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}
        t_cur = repeat(t, 'h -> (h repeat)',repeat =int(x.shape[0]/t.shape[0]))
        if len(x.shape) == 2: # latent
                model_forward = model.forward(x=x,
                                          t=self._scale_timesteps(t),
                                          **model_kwargs)
        else:
            B,C,H,W = shapes
            patch_num_x = H // patch_size
            patch_num_y = W // patch_size
            halfp = patch_size//2
            if model_kwargs['imgs'] is None:
                model_kwargs['imgs'] = th.randn(shapes).to(x.device)
            
            with autocast(self.conf.fp16):
                model_forward = model.forward(x=x,
                                            t=self._scale_timesteps(t),
                                            idx = idx,
                                            patch_size = patch_size,
                                            **model_kwargs)
        model_output = model_forward.pred
        if len(model_output.shape) != 2:
            if model_output.size(0) % (patch_num_x * patch_num_y) == 0 and model_output.size(0) // (patch_num_x * patch_num_y) == t.shape[0]: # should be 17 * 17 originally
                noise_ori = rearrange(model_output, '(b p1 p2) c h w -> b c (p1 h) (p2 w)', p1 = patch_num_x, p2 = patch_num_y)
                noise_ori = F.pad(noise_ori, (halfp, halfp, halfp, halfp), 'constant')
                model_output = rearrange(noise_ori, 'b c (p1 h) (p2 w) -> (b p1 p2) c h w', h = patch_size, w = patch_size)

            elif model_output.size(0) % ((patch_num_x+1) * (patch_num_y+1)) == 0 and model_output.size(0) // ((patch_num_x+1) * (patch_num_y+1)) == t.shape[0]: #should be 16 * 16 
                noise_ori = rearrange(model_output, '(b p1 p2) c h w -> b c (p1 h) (p2 w)', p1 = (patch_num_x+1), p2 = (patch_num_y+1))
                noise_ori = noise_ori[:, :, halfp:-halfp, halfp:-halfp]
                model_output = rearrange(noise_ori, 'b c (p1 h) (p2 w) -> (b p1 p2) c h w', h = patch_size, w = patch_size)
            
            if self.conf.cfg:
                w=0.5
                e_t_cond, e_t_uncond = model_output.chunk(2)
                model_output = (1 + w) * e_t_cond - w * e_t_uncond
                x = x.chunk(2)[0]
                t = t.chunk(2)[0]
                t_cur = repeat(t, 'h -> (h repeat)',repeat =int(x.shape[0]/t.shape[0]))

        if self.model_var_type in [
                ModelVarType.fixed_large, ModelVarType.fixed_small
        ]:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.fixed_large: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(
                        np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.fixed_small: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t_cur, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t_cur,
                                                      x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type in [
                ModelMeanType.eps,
        ]:
            if self.model_mean_type == ModelMeanType.eps:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t_cur,
                                                  eps=model_output))
            else:
                raise NotImplementedError()
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t_cur)
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (model_mean.shape == model_log_variance.shape ==
                pred_xstart.shape == x.shape)
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            'model_forward': model_forward,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t,
                                     x_t.shape) * x_t -
                _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t,
                                     x_t.shape) * eps)

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape)
            * xprev - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                x_t.shape) * x_t)

    def _predict_xstart_from_scaled_xstart(self, t, scaled_xstart):
        return scaled_xstart * _extract_into_tensor(
            self.sqrt_recip_alphas_cumprod, t, scaled_xstart.shape)

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t,
                                     x_t.shape) * x_t -
                pred_xstart) / _extract_into_tensor(
                    self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _predict_eps_from_scaled_xstart(self, x_t, t, scaled_xstart):
        """
        Args:
            scaled_xstart: is supposed to be sqrt(alphacum) * x_0
        """
        # 1 / sqrt(1-alphabar) * (x_t - scaled xstart)
        return (x_t - scaled_xstart) / _extract_into_tensor(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            # scale t to be maxed out at 1000 steps
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = (p_mean_var["mean"].float() +
                    p_mean_var["variance"] * gradient.float())
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, self._scale_timesteps(t), **model_kwargs)

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t)
        return out

    def p_sample(
        self,
        model: Model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        idx = None,
        patch_size = 64,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            shapes = model_kwargs['imgs'].shape,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            idx = idx,
            patch_size = patch_size,
            model_kwargs=model_kwargs,
        )
        if self.conf.cfg:
            x = x.chunk(2)[0]
            t = t.chunk(2)[0]

        noise = th.randn_like(x)
        t_cur = repeat(t, 'h -> (h repeat)', repeat =int(x.shape[0]/t.shape[0]))
        nonzero_mask = ((t_cur != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
                        )  # no noise when t == 0
        # print(cond_fn) # None
        if cond_fn is not None:
            out["mean"] = self.condition_mean(cond_fn,
                                              out,
                                              x,
                                              t_cur,
                                              model_kwargs=model_kwargs)
        # else:
            # print("cond_fn is None.")
        sample = out["mean"] + nonzero_mask * th.exp(
            0.5 * out["log_variance"]) * noise
        # print(sample)
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model: Model,
        shape=None,
        noise=None,
        all_pos=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        idx = None,
        patch_size = 64,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
                model,
                shape,
                noise=noise,
                all_pos=all_pos,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                idx = idx,
                patch_size = patch_size,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model: Model,
        shapes=None,
        noise=None,
        all_pos=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        idx = None,
        patch_size = 64,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        b, c, H, W = model_kwargs['imgs'].shape
        if device is None:
            device = next(model.parameters()).device
        
        img = th.randn((b, 3, H, W), device=device)
        patch_size = patch_size
        halfp = patch_size // 2
        patch_num_x = H // patch_size
        patch_num_y = W // patch_size
        indices = list(range(self.num_timesteps))[::-1]
        model_kwargs['pos'] = all_pos[0]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)
        if self.conf.cfg: # classifier free-guidance
            pos_random = th.cat([th.tensor([1]*b), th.tensor([0]*b)]).long().to(device)
            seman_random = th.cat([th.tensor([1]*b), th.tensor([0]*b)]).long().to(device)
            model_kwargs["random"] = seman_random
            model_kwargs["pos_random"] = pos_random
            model_kwargs['imgs'] = th.cat([model_kwargs['imgs']]*2, dim = 0)
            model_kwargs['pos'] = th.cat([model_kwargs['pos']]*2, dim = 0)

        for i in indices:
            
            t = th.tensor([i] * b, device=device)
            if self.conf.cfg:
                img = th.cat([img]*2, dim = 0)
                t = th.cat([t]*2)

            img_new = F.pad(img, (halfp, halfp, halfp, halfp), 'constant')
            img_new = rearrange(img_new, 'b c (p1 h) (p2 w) -> (b p1 p2) c h w', h = patch_size, w = patch_size)

            with th.no_grad():
                out = self.p_sample(
                    model,
                    img_new,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    idx = idx,
                    patch_size = patch_size,
                    model_kwargs=model_kwargs,
                )

            img_new = rearrange(out['sample'], '(b p1 p2) c h w -> b c (p1 h) (p2 w)', p1 = patch_num_x +1, p2 = patch_num_y+1)
            img = img_new[:, :, halfp:-halfp, halfp:-halfp]

            out['sample'] = img
            yield out
            img = out["sample"]

    def ddim_sample(
        self,
        model: Model,
        x,
        t,
        shapes = (1,3,256,256),
        clip_denoised=True,
        denoised_fn=None,
        idx=None, # newly added
        cond_fn=None,
        patch_size = 64,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            idx=idx,
            shapes = shapes,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            patch_size = patch_size,
            model_kwargs=model_kwargs,
        )
        if self.conf.cfg:
            x = x.chunk(2)[0]
            t = t.chunk(2)[0]
            
        t_cur = repeat(t, 'h -> (h repeat)',repeat =int(x.shape[0]/t.shape[0]))
        if cond_fn is not None:
            out = self.condition_score(cond_fn,
                                       out,
                                       x,
                                       t,
                                       model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t_cur, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t_cur, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t_cur,
                                              x.shape)
        sigma = (eta * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) *
                 th.sqrt(1 - alpha_bar / alpha_bar_prev))
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (out["pred_xstart"] * th.sqrt(alpha_bar_prev) +
                     th.sqrt(1 - alpha_bar_prev - sigma**2) * eps)
        nonzero_mask = ((t_cur != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
                        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model: Model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        NOTE: never used ? 
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape)
               * x - out["pred_xstart"]) / _extract_into_tensor(
                   self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t,
                                              x.shape)

        # Equation 12. reversed  (DDIM paper)  (th.sqrt == torch.sqrt)
        mean_pred = (out["pred_xstart"] * th.sqrt(alpha_bar_next) +
                     th.sqrt(1 - alpha_bar_next) * eps)

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample_loop(
        self,
        model: Model,
        x,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
        device=None,
    ):
        if device is None:
            device = next(model.parameters()).device
        sample_t = []
        xstart_t = []
        T = []
        indices = list(range(self.num_timesteps))
        sample = x
        for i in indices:
            t = th.tensor([i] * len(sample), device=device)
            with th.no_grad():
                out = self.ddim_reverse_sample(model,
                                               sample,
                                               t=t,
                                               clip_denoised=clip_denoised,
                                               denoised_fn=denoised_fn,
                                               model_kwargs=model_kwargs,
                                               eta=eta)
                sample = out['sample']
                # [1, ..., T]
                sample_t.append(sample)
                # [0, ...., T-1]
                xstart_t.append(out['pred_xstart'])
                # [0, ..., T-1] ready to use
                T.append(t)

        return {
            #  xT "
            'sample': sample,
            # (1, ..., T)
            'sample_t': sample_t,
            # xstart here is a bit different from sampling from T = T-1 to T = 0
            # may not be exact
            'xstart_t': xstart_t,
            'T': T,
        }
    
    def ddim_latent_sample(
        self,
        model: Model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.
        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn,
                                       out,
                                       x,
                                       t,
                                       model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t,
                                              x.shape)
        sigma = (eta * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) *
                 th.sqrt(1 - alpha_bar / alpha_bar_prev))
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (out["pred_xstart"] * th.sqrt(alpha_bar_prev) +
                     th.sqrt(1 - alpha_bar_prev - sigma**2) * eps)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
                        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}
    
    def ddim_latent_sample_loop(
        self,
        model: Model,
        shape=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.
        Same usage as p_sample_loop().
        """
        final = None
        # img = self.ODE_sample_latent_loop_progressive(
        #         model,
        #         shape,
        #         noise=noise,
        #         clip_denoised=clip_denoised,
        #         denoised_fn=denoised_fn,
        #         cond_fn=cond_fn,
        #         model_kwargs=model_kwargs,
        #         device=device,
        #         progress=progress,
        #         eta=eta,
        # )
        # return img
        for sample in self.ddim_sample_latent_loop_progressive(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
                eta=eta,
        ):
            final = sample
        return final["sample"]
    
    def ODE_sample_latent_loop_progressive(
        self,
        model: Model,
        shape=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.
        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        if noise is not None:
            img = noise
        else:
            assert isinstance(shape, (tuple, list))
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        def ode_func(t, x, model, shapes, model_kwargs, device):
            x = from_flattened_numpy(x, shapes).to(device).type(th.float32)
            vec_t = th.ones(shapes[0], device=x.device) * t
            # drift = flow.model_forward_wrapper(model, x, vec_t, **kwargs)
   
            drift  = model(x, vec_t*999, **model_kwargs)
            # print(type(drift[0]))
            # print(drift[0].shape)
            # sys.exit()
            return to_flattened_numpy(drift[0])
        
        def ode(t, x):
            return ode_func(t, x, model, img.shape, model_kwargs, device)
        
        rtol = atol = 1e-7
        T = 1.
        eps = 1e-3
        # print("img shape:" + str(img.shape))
        # sys.exit()
        solution = integrate.solve_ivp(ode, (eps, T), to_flattened_numpy(img),
                                             rtol=rtol, atol=atol, method='RK45')
        nfe = solution.nfev
        print("steps: " + str(nfe))
        
        ret = th.tensor(solution.y[:, -1]).reshape(img.shape).to(device).type(th.float32)
        return ret
    
    def ddim_sample_latent_loop_progressive(
        self,
        model: Model,
        shape=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.
        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        if noise is not None:
            img = noise
        else:
            assert isinstance(shape, (tuple, list))
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        
        
        for i in indices:

            if isinstance(model_kwargs, list):
                # index dependent model kwargs
                # (T-1, ..., 0)
                _kwargs = model_kwargs[i]
            else:
                _kwargs = model_kwargs

            t = th.tensor([i] * len(img), device=device)
            with th.no_grad():
                out = self.ddim_latent_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=_kwargs,
                    eta=eta,
                )
                out['t'] = t
                yield out
                img = out["sample"]

    def ddim_sample_loop(
        self,
        model: Model,
        shape=None,
        noise=None,
        pos=None,
        idx=None, # New added
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        
        final = None
        # img = self.ODE_sample_loop_progressive(
        #     model,
        #     shape,
        #     noise=noise,
        #     pos=pos,
        #     clip_denoised=clip_denoised,
        #     idx=idx, # New added
        #     denoised_fn=denoised_fn,
        #     cond_fn=cond_fn,
        #     model_kwargs=model_kwargs,
        #     device=device,
        #     progress=progress,
        #     eta=eta,
        # )
        # return img
        for sample in self.ddim_sample_loop_progressive(
                model,
                shape,
                noise=noise,
                pos=pos,
                idx=idx,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
                eta=eta,
        ):
            final = sample
        return final["sample"]
        
    def ODE_sample_loop_progressive(
        self,
        model: Model,
        shapes=None,
        noise=None,
        pos=None,
        idx=None, # New added
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        if shapes is None:
            shapes = model_kwargs['imgs'].shape
        img = th.randn(shapes, device=device)
        b,c,H,W = shapes
        if noise is None:
            patch_size = 64
        else:
            patch_size = noise.shape[2]
        halfp = patch_size // 2
        patch_num_x = H // patch_size
        patch_num_y = W // patch_size
        indices = list(range(self.num_timesteps))[::-1]
        model_kwargs['pos'] = pos[0]
        
        if model_kwargs['imgs'] is None:
            print("img is None.")
            model_kwargs['imgs'] = img
        # sys.exit()
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        if self.conf.cfg: # classifier free-guidance: false
            print("Classifier free-guidance")
            pos_random = th.cat([th.tensor([1]*b), th.tensor([0]*b)]).long().to(device)
            seman_random = th.cat([th.tensor([1]*b), th.tensor([0]*b)]).long().to(device)
            model_kwargs["random"] = seman_random
            model_kwargs["pos_random"] = pos_random
            model_kwargs['pos'] = th.cat([model_kwargs['pos']]*2, dim = 0)
            model_kwargs['cond'] = th.cat([model_kwargs['cond']]*2, dim = 0)
            shapes = (2*b,c,H,W)
            
        # print("indices:" + str(indices))
        steps = 0
        
        # test_idx = th.randint(1, 5000, (shapes[0],), device=device)
            
        # print("test_idx: " + str(test_idx))
        def ode_func(t, x, model, shapes, patch_size, model_kwargs, device):
            x = from_flattened_numpy(x, shapes).to(device).type(th.float32)
            vec_t = th.ones(shapes[0], device=x.device) * t
            halfp = patch_size // 2
            x_padded = F.pad(x, (halfp, halfp, halfp, halfp), mode="constant")
            x_patched = rearrange(x_padded, 'b c (p1 h) (p2 w) -> (b p1 p2) c h w', h=patch_size, w=patch_size)

            # print("ode: before drift.")
            
            # print("test_idx size: " + str(test_idx.size())) # 
            # print("vec_t size: " + str(vec_t.size())) 
            # print("test_idx: " + str(test_idx))
            # print("idx: " + str(idx)) 
            # model_kwargs["cond"] = None
            
            # print("Model kwargs: " + str(model_kwargs["cond"]))
            # sys.exit()
            # print("steps: " + str(steps))
            model_forward = model(x_patched, 
                                   t=vec_t*999, 
                                   patch_size=patch_size, 
                                #    idx=test_idx, 
                                   **model_kwargs)
            drift_patches = model_forward[0]
            # print("ode: after drift.")
            
            # drift_padded = rearrange(drift_patches[0], '(b p1 p2) c h w -> b c (p1 h) (p2 w)', p1 = 2, p2 = 2)
            drift_padded = rearrange(drift_patches, '(b p1 p2) c h w -> b c (p1 h) (p2 w)', p1 = patch_num_x, p2 = patch_num_y)
            
            # reverse = -drift_padded
            return to_flattened_numpy(drift_padded)
        # test_idx = th.randint(1, 70000, (shapes[0],), device=img.device)
        def ode(t, x):
            return ode_func(t, x, model, shapes, patch_size, model_kwargs, device)
        
        
        
        T = 1.
        eps = 1e-3
     
        rtol = atol = 1e-5

        
        solution = integrate.solve_ivp(ode, (eps, T), to_flattened_numpy(img),
                                            rtol=rtol, atol=atol, method='RK45')
        nfe = solution.nfev
        print("steps: " + str(nfe))
        ret = th.tensor(solution.y[:, -1]).reshape(shapes).to(device).type(th.float32)
        
        # ret = (ret + 1.) / 2.
        # ret = th.clamp(ret, 0., 1.)
        ret_min = ret.min().item()
        ret_max = ret.max().item()
        
        print("ret min: " + str(ret_min))
        print("ret max: " + str(ret_max))
        # sys.exit()
        return ret
        

    def ddim_sample_loop_progressive(
        self,
        model: Model,
        shapes=None,
        noise=None,
        pos=None,
        idx=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        
        img = th.randn(shapes, device=device)
        b,c,H,W = shapes
        patch_size = noise.shape[2]
        halfp = patch_size // 2
        patch_num_x = H // patch_size
        patch_num_y = W // patch_size
        indices = list(range(self.num_timesteps))[::-1]
        model_kwargs['pos'] = pos[0]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        if self.conf.cfg: # classifier free-guidance
            pos_random = th.cat([th.tensor([1]*b), th.tensor([0]*b)]).long().to(device)
            seman_random = th.cat([th.tensor([1]*b), th.tensor([0]*b)]).long().to(device)
            model_kwargs["random"] = seman_random
            model_kwargs["pos_random"] = pos_random
            model_kwargs['pos'] = th.cat([model_kwargs['pos']]*2, dim = 0)
            model_kwargs['cond'] = th.cat([model_kwargs['cond']]*2, dim = 0)
            shapes = (2*b,c,H,W)
            
        
        print(img.shape)
        # sys.exit()
            # Succeed when cond is None: (given index)
                # test_idx = th.randint(6000, 6001, (shapes[0], ), device=device)
                # test_idx = th.ones((shapes[0],), device=device) * 2
                # if model_kwargs["cond"] is not None:
                #     print("cond is not none.")
                # else:
                #     print("cond is none.")
                # model_kwargs["cond"] = None
        for i in indices:
            
            t = th.tensor([i] * b, device=device) # change_code_note

            if self.conf.cfg:
                img = th.cat([img]*2, dim = 0)
                t = th.cat([t]*2)

            img_new = F.pad(img, (halfp, halfp, halfp, halfp), 'constant')
            img_new = rearrange(img_new, 'b c (p1 h) (p2 w) -> (b p1 p2) c h w', h = patch_size, w = patch_size)

            if isinstance(model_kwargs, list):
                # index dependent model kwargs
                # (T-1, ..., 0)
                _kwargs = model_kwargs[i]
            else:
                _kwargs = model_kwargs

            # print("Model kwargs keys: " + str(model_kwargs.keys()))

            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img_new,
                    t,
                    idx = idx,
                    shapes = shapes,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    patch_size = patch_size,
                    model_kwargs=_kwargs,
                    eta=eta,
                )
            img_new = rearrange(out['sample'], '(b p1 p2) c h w -> b c (p1 h) (p2 w)', p1 = patch_num_x+1, p2 = patch_num_y+1)
            img = img_new[:, :, halfp:-halfp, halfp:-halfp]
            out['sample'] = img
            yield out
            img = out["sample"]

    def _vb_terms_bpd(self,
                      model: Model,
                      x_start,
                      x_t,
                      t,
                      clip_denoised=True,
                      model_kwargs=None):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t)
        out = self.p_mean_variance(model,
                                   x_t,
                                   t,
                                   clip_denoised=clip_denoised,
                                   model_kwargs=model_kwargs)
        kl = normal_kl(true_mean, true_log_variance_clipped, out["mean"],
                       out["log_variance"])
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"])
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {
            "output": output,
            "pred_xstart": out["pred_xstart"],
            'model_forward': out['model_forward'],
        }

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size,
                      device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean,
                             logvar1=qt_log_variance,
                             mean2=0.0,
                             logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self,
                      model: Model,
                      x_start,
                      clip_denoised=True,
                      model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start)**2))
            eps = self._predict_eps_from_xstart(x_t, t_batch,
                                                out["pred_xstart"])
            mse.append(mean_flat((eps - noise)**2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start,
                           beta_end,
                           num_diffusion_timesteps,
                           dtype=np.float64)
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2)**2,
        )
    elif schedule_name == "const0.01":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.01] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.015":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.015] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.008":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.008] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.0065":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.0065] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.0055":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.0055] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.0045":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.0045] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.0035":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.0035] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.0025":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.0025] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.0015":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.0015] * num_diffusion_timesteps,
                        dtype=np.float64)
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (-1.0 + logvar2 - logvar1 + th.exp(logvar1 - logvar2) +
                  ((mean1 - mean2)**2) * th.exp(-logvar2))


def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return th.from_numpy(x.reshape(shape))

def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (
        1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min,
                 th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs


class DummyModel(th.nn.Module):
    def __init__(self, pred):
        super().__init__()
        self.pred = pred

    def forward(self, *args, **kwargs):
        return DummyReturn(pred=self.pred)


class DummyReturn(NamedTuple):
    pred: th.Tensor