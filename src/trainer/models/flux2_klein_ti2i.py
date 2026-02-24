import math
from typing import Literal

import torch
from diffusers import Flux2KleinPipeline
from diffusers.training_utils import compute_loss_weighting_for_sd3
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)

from trainer.train import Trainer
from trainer.distributed import ParallelDims


class Flux2KleinTI2ITrainer(Trainer):
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
        prompt_embeds = batch["text_embeddings"]
        text_ids = Flux2KleinPipeline._prepare_text_ids(prompt_embeds).to(
            device=self.device
        )

        # vae.encode(pixel_values).latent_dist.mode()
        model_input = batch["image_latents"]
        # model_input = Flux2KleinPipeline._patchify_latents(
        #     model_input
        # )
        model_input_ids = Flux2KleinPipeline._prepare_latent_ids(model_input).to(
            device=model_input.device
        )

        cond_model_input = batch["condition_image_latents"]
        cond_model_input_ids = torch.cat(
            [
                Flux2KleinPipeline._prepare_image_ids(
                    [cond_model_input[i].unsqueeze(0)]
                ).to(  # [(1, C, H, W)] -> [1, N_total, 4]
                    device=cond_model_input.device
                )
                for i in range(cond_model_input.shape[0])
            ],
            dim=0,
        )  # [B, N_total, 4]

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
        packed_cond_model_input = Flux2KleinPipeline._pack_latents(cond_model_input)
        orig_input_shape = packed_noisy_model_input.shape
        orig_input_ids_shape = model_input_ids.shape

        # concatenate the model inputs with the cond inputs
        packed_noisy_model_input = torch.cat(
            [packed_noisy_model_input, packed_cond_model_input], dim=1
        )
        model_input_ids = torch.cat([model_input_ids, cond_model_input_ids], dim=1)

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
        # pruning the condition information
        model_pred = model_pred[:, : orig_input_shape[1], :]
        model_input_ids = model_input_ids[:, : orig_input_ids_shape[1], :]

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

    @staticmethod
    def parallelize_model(
        model: torch.nn.Module,
        parallel_dims: ParallelDims,
    ) -> torch.nn.Module:
        # pyright: ignore [missing-attribute]
        for layer_id, block in model.transformer_blocks.named_children():
            block = ptd_checkpoint_wrapper(block, preserve_rng_state=True)
            # pyrefly: ignore [missing-attribute]
            model.transformer_blocks.register_module(layer_id, block)

        # pyrefly: ignore [missing-attribute]
        for layer_id, block in model.single_transformer_blocks.named_children():
            block = ptd_checkpoint_wrapper(block, preserve_rng_state=True)
            # pyrefly: ignore [missing-attribute]
            model.single_transformer_blocks.register_module(layer_id, block)

        return model.cuda()
