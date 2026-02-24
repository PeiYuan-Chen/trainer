from typing import Any

import torch
from streaming import StreamingDataset

from .utils import make_streams, numpy_to_tensor


class StreamingImageCaptionLatentsDataset(StreamingDataset):
    def __init__(
        self,
        batch_size: int,
        remote: str | list[str] | None = None,
        local: str | list[str] | None = None,
        proportion: list | None = None,
        repeat: list | None = None,
        choose: list | None = None,
        image_latent_key: str = "image_latents",
        image_latent_dtype: torch.dtype = torch.bfloat16,
        caption_keys: tuple[str, ...] = ("caption",),
        caption_selection_probs: tuple[float, ...] = (1.0,),
        caption_drop_prob: float = 0.0,
        text_embedding_keys: tuple[str, ...] = ("text_embeddings"),
        text_embedding_dtype: torch.dtype = torch.bfloat16,
        uncond_text_embedding_path: str | None = None,
        **streaming_kwargs,
    ):
        # Set defaults for vision-friendly streaming args.
        streaming_kwargs.setdefault("shuffle_block_size", 1 << 18)
        streaming_kwargs.setdefault("shuffle_algo", "py1s")
        streams = make_streams(
            remote, local=local, proportion=proportion, repeat=repeat, choose=choose
        )
        super().__init__(streams=streams, batch_size=batch_size, **streaming_kwargs)

        # validate
        if len(caption_keys) != len(caption_selection_probs):
            raise ValueError(
                "Length of caption_keys and caption_selection_probs must be the same"
            )
        if len(caption_keys) == 0:
            raise ValueError("caption_keys must be non-empty")
        if any(p < 0 for p in caption_selection_probs):
            raise ValueError("caption_selection_probs must be non-negative")
        s = float(sum(caption_selection_probs))
        if s <= 0:
            raise ValueError("caption_selection_probs must sum to a positive value")
        if caption_drop_prob < 0.0 or caption_drop_prob > 1.0:
            raise ValueError("caption_drop_prob must be between 0 and 1")
        if caption_drop_prob > 0.0 and uncond_text_embedding_path is None:
            raise ValueError(
                "uncond_text_embedding_path must be provided if caption_drop_prob > 0.0"
            )

        self.image_latent_key = image_latent_key
        self.image_latent_dtype = image_latent_dtype
        self.caption_keys = caption_keys
        self.caption_selection_probs = torch.tensor(
            [p / s for p in caption_selection_probs], dtype=torch.float32
        )
        self.caption_drop_prob = float(caption_drop_prob)
        self.text_embedding_keys = text_embedding_keys
        self.text_embedding_dtype = text_embedding_dtype

        # load uncond text embeddings
        self.uncond_text_embeddings = {}
        if uncond_text_embedding_path is not None:
            obj = torch.load(uncond_text_embedding_path, map_location="cpu")
            if not isinstance(obj, dict):
                raise ValueError(
                    f"uncond_text_embeddings must be a dictionary, but got {type(obj)}"
                )
            for k in self.text_embedding_keys:
                if k not in obj:
                    raise ValueError(
                        f"Missing unconditional embedding for key '{k}' in "
                        f"{uncond_text_embedding_path}. Available keys: {list(obj.keys())}"
                    )
                v = obj[k]
                if not isinstance(v, torch.Tensor):
                    raise ValueError(
                        f"uncond embedding '{k}' must be a torch.Tensor, got {type(v)}"
                    )
                # cast dtype
                self.uncond_text_embeddings[k] = v.to(dtype=self.text_embedding_dtype)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        out = {}

        # image latents
        out[self.image_latent_key] = numpy_to_tensor(
            sample[self.image_latent_key],
            self.image_latent_dtype,
        )

        # random select a caption according to the selection probabilities
        # torch.multinomial uses torch RNG (seeded via worker_init_fn/torch.initial_seed)
        caption_key = self.caption_keys[
            torch.multinomial(self.caption_selection_probs, num_samples=1).item()
        ]
        # load text embedding
        drop_caption = torch.rand(1).item() < self.caption_drop_prob
        for name in self.text_embedding_keys:
            if drop_caption:
                out[name] = self.uncond_text_embeddings[name]
            else:
                out[name] = numpy_to_tensor(
                    sample[f"{caption_key}_{name}"],
                    self.text_embedding_dtype,
                )

        return out
