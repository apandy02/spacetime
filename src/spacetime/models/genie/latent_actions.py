import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from spacetime.models.genie.config import LamConfig
from spacetime.models.tokenizer.model import VQVAEVideoEncoder
from spacetime.modules.quantizers import EMAVectorQuantizer, QuantizerType, VanillaVectorQuantizer
from spacetime.modules.transformer import STTransformerLayer
from spacetime.modules.utils import build_anti_causal_mask, patchify, unpatchify


class LatentActionModel(nn.Module):
    """
    Latent action model (VAE with quantized latent actions).
    """

    def __init__(
        self,
        lam_cfg: LamConfig,
    ):
        super(LatentActionModel, self).__init__()
        self.patch_size = lam_cfg.patch_size
        self.frame_height = lam_cfg.frame_height
        self.frame_width = lam_cfg.frame_width
        self.num_frames = lam_cfg.num_frames
        self.n_patch_h = lam_cfg.frame_height // lam_cfg.patch_size
        self.n_patch_w = lam_cfg.frame_width // lam_cfg.patch_size
        self.n_patches = self.n_patch_h * self.n_patch_w

        d_patches = 3 * lam_cfg.patch_size * lam_cfg.patch_size  # hardcoded for RGB

        self.d_model_projection = nn.Linear(d_patches, lam_cfg.d_model)

        self.pos_embed_space = nn.Parameter(torch.zeros(1, 1, self.n_patches, lam_cfg.d_model))
        self.pos_embed_time = nn.Parameter(torch.zeros(1, self.num_frames, 1, lam_cfg.d_model))

        self.anti_causal_encoder = VQVAEVideoEncoder(
            num_heads=lam_cfg.num_heads,
            d_model=lam_cfg.d_model,
            num_layers=lam_cfg.num_layers,
            d_linear=lam_cfg.d_linear,
            codebook_dim=lam_cfg.codebook_dim,
            num_linear_layers=lam_cfg.num_linear_layers,
            num_groups=lam_cfg.num_groups,
            dropout=lam_cfg.dropout,
            is_causal=False,
            mask=build_anti_causal_mask,
            gradient_checkpointing=lam_cfg.gradient_checkpointing,
        )

        if lam_cfg.quantizer_type == QuantizerType.EMA:
            self.vector_quantizer = EMAVectorQuantizer(
                lam_cfg.num_discrete_actions,
                lam_cfg.codebook_dim,
                decay=lam_cfg.quantizer_decay,
                eps=lam_cfg.quantizer_eps,
                dead_code_threshold=lam_cfg.dead_code_threshold,
                dead_code_noise=lam_cfg.dead_code_noise,
            )
        elif lam_cfg.quantizer_type == QuantizerType.VANILLA:
            self.vector_quantizer = VanillaVectorQuantizer(
                lam_cfg.num_discrete_actions, lam_cfg.codebook_dim
            )
        else:
            raise ValueError(f"Invalid quantizer type: {lam_cfg.quantizer_type}")
        self.decoder = LatentActionDecoder(
            num_heads=lam_cfg.num_heads,
            d_model=lam_cfg.d_model,
            num_layers=lam_cfg.num_layers,
            d_linear=lam_cfg.d_linear,
            codebook_dim=lam_cfg.codebook_dim,
            input_image_channels=3,
            patch_size=lam_cfg.patch_size,
            num_linear_layers=lam_cfg.num_linear_layers,
            num_groups=lam_cfg.num_groups,
            dropout=lam_cfg.dropout,
            gradient_checkpointing=lam_cfg.gradient_checkpointing,
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the LatentActionModel (VAE with quantized latent actions).
        Takes as input:
            - x: [B, C, F, H, W] (the input video)
        Returns:
            - frame_reconstructions: [B, C, F-1, H, W], predictions for frames 2..F
              (shifted target: position t predicts frame t+1)
            - z_e: [B, F, NUM_P, D_codebook], continuous encoder outputs before quantization
            - action_codes: [B, F, NUM_P, D_codebook], quantized action codes
            - action_indices: [B, F, NUM_P], codebook indices
        """
        x = patchify(x, self.patch_size)  # [B, F, NUM_P, DIM_P]
        frame_embeddings = self.d_model_projection(x)
        encoder_input = frame_embeddings + self.pos_embed_space + self.pos_embed_time
        encoder_output = self.anti_causal_encoder(encoder_input)
        action_codes, action_indices, z_e = self.vector_quantizer(encoder_output)

        st_estimator_action_codes = z_e + (action_codes - z_e).detach()
        frame_reconstructions = self.decoder(
            st_estimator_action_codes,
            frame_embeddings,
            self.pos_embed_space,
            self.pos_embed_time,
        )
        return (
            unpatchify(frame_reconstructions, self.patch_size, self.n_patch_h, self.n_patch_w),
            z_e,
            action_codes,
            action_indices,
        )


class LatentActionDecoder(nn.Module):
    """
    Latent action decoder block.
    """

    def __init__(
        self,
        num_heads,
        d_model: int,
        num_layers: int,
        d_linear: int,
        codebook_dim: int,
        input_image_channels: int,
        patch_size: int,
        num_linear_layers: int = 2,
        num_groups: int = 8,
        dropout: float = 0.1,
        gradient_checkpointing: bool = False,
    ):
        super(LatentActionDecoder, self).__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.d_model_projection = nn.Linear(codebook_dim, d_model)
        self.st_decoder = nn.ModuleList(
            [
                STTransformerLayer(
                    num_heads,
                    d_model,
                    d_linear,
                    num_linear_layers,
                    num_groups,
                    dropout,
                    is_causal=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.token_type_embed = nn.Parameter(torch.zeros(2, d_model))
        self.reconstruction_projector = nn.Linear(
            d_model, input_image_channels * patch_size * patch_size
        )

    def forward(
        self,
        action_embeddings: torch.Tensor,
        frame_embeddings: torch.Tensor,
        pos_embed_space: torch.Tensor,
        pos_embed_time: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the LatentActionDecoder.
        Takes as input:
            - action_embeddings: [B, T, N, D_codebook] (the latent action tokens)
            - frame_embeddings: [B, T, N, D_model] (embedded frames)

        Does:
            - Projects action embeddings to d_model: [B, T, N, D_model]
            - Interleaves frame and action tokens: [f_1, a_1, f_2, a_2, ..., f_T, a_T]
            - Uses causal masking with is_causal=True (enables flash attention)
            - Extracts action positions 1, 3, ..., 2T-3 to predict frames 2..T
              Position 2t-1 (a_t slot) sees f_1, a_1, ..., f_t, a_t → predicts f_{t+1}
        Outputs:
            - [B, T-1, N, patch_dim], predictions for frames 2..T
        """
        action_embeddings = self.d_model_projection(action_embeddings)

        frame_embeddings = (
            frame_embeddings
            + pos_embed_space
            + pos_embed_time
            + self.token_type_embed[0].view(1, 1, 1, -1)
        )
        action_embeddings = (
            action_embeddings
            + pos_embed_space
            + pos_embed_time
            + self.token_type_embed[1].view(1, 1, 1, -1)
        )

        # Interleave frame and action tokens: [f_1, a_1, f_2, a_2, ..., f_T, a_T]
        # This ordering ensures that with causal masking, position 2t (f_{t+1} slot)
        # sees f_1, a_1, ..., f_t, a_t - exactly what's needed to predict f_{t+1}
        batch_size, T, N, D = frame_embeddings.shape
        x = torch.stack([frame_embeddings, action_embeddings], dim=2)  # [B, T, 2, N, D]
        x = x.reshape(
            batch_size, 2 * T, N, D
        )  # [f_1, a_1, f_2, a_2, ..., f_T, a_T] -> [B, 2T, N, D]

        for layer in self.st_decoder:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        x = self.layer_norm(x)

        # Extract action positions 1, 3, ..., 2T-3 to predict frames 2, 3, ..., T.
        # With causal mask, position 2t-1 (a_t slot) sees f_1, a_1, ..., f_t, a_t
        # This is exactly the context needed to predict f_{t+1}.
        x = x[:, 1::2, :, :][:, :-1, :, :]  # [B, T-1, N, D] - positions 1, 3, ..., 2T-3
        x = self.reconstruction_projector(x)
        return x
