import math
from typing import Literal

import torch
from diffusers import Flux2KleinPipeline
from diffusers.training_utils import compute_loss_weighting_for_sd3

from trainer.train import Trainer


class Flux2KleinT2ITrainer(Trainer):
    def __init__(
        self,
        weighting_scheme: Literal["logit_normal", "uniform"] = "uniform",
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        noise_shift: float = 1.0,
        guidance_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.weighting_scheme = weighting_scheme
        self.logit_mean = logit_mean
        self.logit_std = logit_std
        self.noise_shift = noise_shift
        self.guidance_scale = guidance_scale

    def forward_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # text_encoding_pipeline.encode_prompt
        prompt_embeds = batch["text_embedding"]
        text_ids = Flux2KleinPipeline._prepare_text_ids(prompt_embeds)

        # vae.encode(pixel_values).latent_dist.mode()
        model_input = batch["image_latents"]
        # model_input = Flux2KleinPipeline._patchify_latents(
        #     model_input
        # )
        model_input_ids = Flux2KleinPipeline._prepare_latent_ids(model_input).to(
            device=model_input.device
        )

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(model_input)
        bsz = model_input.shape[0]

        # Sample a random timestep for each image
        # for weighting schemes where we sample timesteps non-uniformly
        if self.weighting_scheme == "logit_normal":
            timesteps = torch.normal(
                mean=self.logit_mean + math.log(self.noise_shift),
                std=self.logit_std,
                size=(bsz,),
                device=model_input.device,
            )
            timesteps = torch.sigmoid(timesteps)
        else:
            timesteps = torch.rand(size=(bsz,), device=model_input.device)

        # [B] -> [B, 1, 1, 1]
        sigmas = timesteps.view(-1, 1, 1, 1)
        # Add noise according to flow matching.
        # zt = (1 - texp) * x + texp * z1
        noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

        # [B, C, H, W] -> [B, H*W, C]
        packed_noisy_model_input = Flux2KleinPipeline._pack_latents(noisy_model_input)

        # handle guidance
        if self.model.config.guidance_embeds:
            guidance = torch.full([1], self.guidance_scale, device=self.device)
            guidance = guidance.expand(model_input.shape[0])
        else:
            guidance = None

        # Predict the noise residual
        model_pred = self.model(
            hidden_states=packed_noisy_model_input,  # (B, image_seq_len, C)
            timestep=timesteps,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,  # B, text_seq_len, 4
            img_ids=model_input_ids,  # B, image_seq_len, 4
            return_dict=False,
        )[0]
        model_pred = model_pred[:, : packed_noisy_model_input.size(1) :]
        model_pred = Flux2KleinPipeline._unpack_latents_with_ids(
            model_pred, model_input_ids
        )

        # # these weighting schemes use a uniform timestep sampling
        # # and instead post-weight the loss
        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme=self.weighting_scheme, sigmas=sigmas
        )
        # flow matching loss
        target = noise - model_input
        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(
                target.shape[0], -1
            ),
            1,
        )
        loss = loss.mean()
        return loss


# from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
# def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
#     sigmas = self.noise_scheduler_copy.sigmas.to(device=self.device, dtype=dtype)
#     schedule_timesteps = self.noise_scheduler_copy.timesteps.to(self.device)
#     timesteps = timesteps.to(self.device)
#     step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

#     sigma = sigmas[step_indices].flatten()
#     while len(sigma.shape) < n_dim:
#         sigma = sigma.unsqueeze(-1)
#     return sigma

# u = compute_density_for_timestep_sampling(
#     weighting_scheme=self.weighting_scheme,
#     batch_size=bsz,
#     logit_mean=self.logit_mean,
#     logit_std=self.logit_std,
#     mode_scale=self.mode_scale,
# )
# indices = (u * self.noise_scheduler_copy.config.num_train_timesteps).long()
# timesteps = self.noise_scheduler_copy.timesteps[indices].to(
#     device=model_input.device
# )

# # Add noise according to flow matching.
# # zt = (1 - texp) * x + texp * z1
# sigmas = self.get_sigmas(
#     timesteps, n_dim=model_input.ndim, dtype=model_input.dtype
# )
# noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
