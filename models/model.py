import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR


def _build_mlp(
    input_dim: int,
    hidden_dims: list[int],
    output_dim: int,
    dropout: float,
    batch_norm: bool,
    pre_norm: bool,
) -> nn.Module:
    layers: list[nn.Module] = []
    if pre_norm:
        layers.append(nn.LayerNorm(input_dim))
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.extend([nn.GELU(), nn.Dropout(dropout)])
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class UnionSupConModel(pl.LightningModule):
    """Late fusion model for multimodal fusion.
    Fuses two embedding modalities with a soft-union contrastive loss.

    A pair is a positive when either modality's cosine similarity is high.
    The synergy term keeps the fused vector from matching a fusion that shuffles
    one modality hence prevents modality collapse.

    Args:
        ps_input_dim: Dimensionality of the Perturb-seq input embeddings.
        cp_input_dim: Dimensionality of the Cell Painting input embeddings.
        encoder_hidden_dims: Hidden widths of each modality encoder.
        encoder_latent_dim: Output width of each modality encoder.
        fusion_hidden_dims: Hidden widths of the fusion MLP.
        fusion_output_dim: Dimensionality of the fused embedding.
        encoder_dropout: Dropout probability inside each modality encoder.
        encoder_batch_norm: Apply batch norm after each encoder hidden linear layer.
        encoder_pre_norm: Apply layer norm to each encoder's input.
        fusion_dropout: Dropout probability inside the fusion MLP.
        fusion_batch_norm: Apply batch norm after each fusion hidden linear layer.
        double_enc1: Give the Perturb-seq encoder more capacity by doubling each
            hidden width and repeating that list twice.
        ps_pos_quantile: Quantile of the Perturb-seq cosine matrix used as the
            soft-positive threshold.
        cp_pos_quantile: Quantile of the Cell Painting cosine matrix used as the
            soft-positive threshold.
        modality_dropout_p: Training probability of replacing the Cell Painting
            input with uniform noise.
        normalize_encoder_output: L2-normalize each encoder output before concatenation.
        tau: Temperature of the supervised contrastive loss. Stored as ``log(tau)``.
        temp: Temperature of the sigmoid that turns cosine similarities into soft positives.
        degree_power: Exponent in the symmetric degree normalization of the positive mask.
        synergy_weight: Weight of the shuffled-modality synergy penalty. ``0`` disables it.
        lr: Peak AdamW learning rate.
        weight_decay: AdamW weight decay.
    """

    def __init__(
        self,
        ps_input_dim: int,
        cp_input_dim: int,
        encoder_hidden_dims: list[int],
        encoder_latent_dim: int,
        fusion_hidden_dims: list[int],
        fusion_output_dim: int,
        encoder_dropout: float = 0.0,
        encoder_batch_norm: bool = True,
        encoder_pre_norm: bool = True,
        fusion_dropout: float = 0.0,
        fusion_batch_norm: bool = False,
        double_enc1: bool = True,
        ps_pos_quantile: float = 0.99,
        cp_pos_quantile: float = 0.99,
        modality_dropout_p: float = 0.5,
        normalize_encoder_output: bool = True,
        tau: float = 0.1,
        temp: float = 1e-4,
        degree_power: float = 1.0,
        synergy_weight: float = 1.0,
        lr: float = 1e-4,
        weight_decay: float = 1e-8,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = lr
        self.weight_decay = weight_decay
        self.synergy_weight = synergy_weight
        self.modality_dropout_p = modality_dropout_p
        self.normalize_encoder_output = normalize_encoder_output
        self.ps_pos_quantile = ps_pos_quantile
        self.cp_pos_quantile = cp_pos_quantile
        self.temp = temp
        self.degree_power = degree_power
        self.log_tau = nn.Parameter(torch.log(torch.tensor(tau)))

        enc_h = list(encoder_hidden_dims)
        # doubling the hidden dims for perturb-seq encoder to give it more capacity
        ps_encoder_hidden_dims = (
            [hidden * 2 for hidden in enc_h] * 2 if double_enc1 else enc_h
        )
        self.ps_encoder = _build_mlp(
            ps_input_dim,
            ps_encoder_hidden_dims,
            encoder_latent_dim,
            encoder_dropout,
            encoder_batch_norm,
            encoder_pre_norm,
        )
        self.cp_encoder = _build_mlp(
            cp_input_dim,
            enc_h,
            encoder_latent_dim,
            encoder_dropout,
            encoder_batch_norm,
            encoder_pre_norm,
        )
        self.fusion_head = _build_mlp(
            encoder_latent_dim
            * 2,  # input is the concatenation of the two modalities' latent representations
            list(fusion_hidden_dims),
            fusion_output_dim,
            fusion_dropout,
            fusion_batch_norm,
            pre_norm=False,
        )

    @property
    def tau(self) -> torch.Tensor:
        return torch.exp(self.log_tau)

    def _fuse(self, ps_h: torch.Tensor, cp_h: torch.Tensor) -> torch.Tensor:
        if self.normalize_encoder_output:
            ps_h = F.normalize(ps_h, p=2, dim=1)
            cp_h = F.normalize(cp_h, p=2, dim=1)
        return self.fusion_head(torch.cat([ps_h, cp_h], dim=1))

    def _encode_and_fuse(self, x_ps: torch.Tensor, x_cp: torch.Tensor):
        if (
            self.training
            and self.modality_dropout_p > 0.0
            and torch.rand(1) < self.modality_dropout_p
        ):
            # apply dropout to the Cell-Paint embedding
            x_cp = torch.rand_like(x_cp)
        ps_h = self.ps_encoder(x_ps)
        cp_h = self.cp_encoder(x_cp)
        return ps_h, cp_h, self._fuse(ps_h, cp_h)

    def _shuffled_fusions(
        self, ps_h: torch.Tensor, cp_h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cp_shuffled = cp_h[torch.randperm(cp_h.size(0))]
        ps_shuffled = ps_h[torch.randperm(ps_h.size(0))]
        return self._fuse(ps_h, cp_shuffled), self._fuse(ps_shuffled, cp_h)

    def _supcon(self, z: torch.Tensor, pos_mask: torch.Tensor) -> torch.Tensor:
        batch_size = z.size(0)
        z_norm = F.normalize(z, p=2, dim=1)
        sim_z = (z_norm @ z_norm.t()) / self.tau
        eye = torch.eye(batch_size, device=z.device, dtype=torch.bool)
        sim_z = sim_z.masked_fill(eye, -1e9)
        log_prob = sim_z - torch.logsumexp(sim_z, dim=1, keepdim=True)
        pos_mask_f = pos_mask.float()
        pos_count = pos_mask_f.sum(dim=1)
        has_positives = (pos_count > 0).float()
        loss_per_anchor = -(pos_mask_f * log_prob).sum(dim=1) / pos_count.clamp(min=1.0)
        return (loss_per_anchor * has_positives).sum() / has_positives.sum().clamp(
            min=1.0
        )

    def _shared_step(self, batch, batch_idx):
        ps, cp = batch
        x_ps = ps["embedding"]
        x_cp = cp["embedding"]
        ps_h, cp_h, z = self._encode_and_fuse(x_ps, x_cp)

        # finding pseudo-labels for the positive pairs
        with torch.no_grad():
            x_ps_norm = F.normalize(x_ps, p=2, dim=1)
            x_cp_norm = F.normalize(x_cp, p=2, dim=1)
            sim_ps_mat = x_ps_norm @ x_ps_norm.t()
            sim_cp_mat = x_cp_norm @ x_cp_norm.t()
            eye = torch.eye(z.size(0), device=z.device, dtype=torch.bool)
            thresh_ps = torch.quantile(sim_ps_mat.detach(), self.ps_pos_quantile)
            thresh_cp = torch.quantile(sim_cp_mat.detach(), self.cp_pos_quantile)
            pos_ps = torch.sigmoid((sim_ps_mat - thresh_ps) / self.temp).masked_fill(
                eye, 0.0
            )
            pos_cp = torch.sigmoid((sim_cp_mat - thresh_cp) / self.temp).masked_fill(
                eye, 0.0
            )
            pos_mask = pos_ps + pos_cp - pos_ps * pos_cp
            # degree normalizing the pseudo-labels affinity graph
            degree = pos_mask.sum(dim=1).clamp(min=1.0)
            deg_inv_sqrt = 1.0 / torch.pow(degree, self.degree_power / 2)
            pos_mask = pos_mask * (
                deg_inv_sqrt.unsqueeze(1) * deg_inv_sqrt.unsqueeze(0)
            )

        loss = self._supcon(z, pos_mask)
        return_dict = {"supcon_loss": loss}

        with torch.no_grad():
            sim_z = F.normalize(z, p=2, dim=1)
            sim_z = sim_z @ sim_z.t()
            zero = torch.zeros((), device=z.device)
            return_dict["z_sim_at_ps_pos"] = (
                sim_z[pos_ps > 0].mean() if pos_ps.any() else zero
            )
            return_dict["z_sim_at_cp_pos"] = (
                sim_z[pos_cp > 0].mean() if pos_cp.any() else zero
            )
            return_dict["z_sim_at_union"] = (
                sim_z[pos_mask > 0].mean() if pos_mask.any() else zero
            )
            return_dict["z_sim_off_diag_mean"] = sim_z[~eye].mean()

        if self.synergy_weight > 0:
            z_ps_only, z_cp_only = self._shuffled_fusions(ps_h, cp_h)
            sim_ps = F.cosine_similarity(z, z_ps_only, dim=1).mean()
            sim_cp = F.cosine_similarity(z, z_cp_only, dim=1).mean()
            synergy_loss = torch.clamp(sim_ps + sim_cp, min=0.0)
            loss = loss + self.synergy_weight * synergy_loss
            return_dict["synergy_loss"] = synergy_loss
            return_dict["shuffled_cp_sim"] = sim_ps
            return_dict["shuffled_ps_sim"] = sim_cp

        return_dict["loss"] = loss
        return return_dict

    def _log_metrics(self, res, prefix: str):
        for key, value in res.items():
            self.log(
                f"{prefix}_{key}",
                value,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        if prefix == "train":
            learning_rate = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("train_lr", learning_rate, on_step=True, prog_bar=False)

    def training_step(self, batch, batch_idx):
        res = self._shared_step(batch, batch_idx)
        self._log_metrics(res, "train")
        return res["loss"]

    def validation_step(self, batch, batch_idx):
        res = self._shared_step(batch, batch_idx)
        self._log_metrics(res, "val")
        return res["loss"]

    def predict_step(self, batch, batch_idx):
        ps, cp = batch
        x_ps = ps["embedding"]
        x_cp = cp["embedding"]
        _, _, fused = self._encode_and_fuse(x_ps, x_cp)
        return {"embeddings": fused}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=int(self.trainer.estimated_stepping_batches),
            pct_start=0.02,
            div_factor=10.0,
            final_div_factor=10.0,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
