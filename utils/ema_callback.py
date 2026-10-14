"""Manual exponential moving average of Lightning module weights."""

from typing import Any, Dict

import lightning.pytorch as L
import torch
from lightning.pytorch.callbacks import Callback


class ManualEMACallback(Callback):
    """Track an EMA copy and swap it in for validation, test, and checkpoints."""

    def __init__(self, decay: float = 0.999, update_after_step: int = 100):
        super().__init__()
        self.decay = decay
        self.update_after_step = update_after_step
        self.ema_state_dict = {}
        self.original_state_dict = {}

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        self.ema_state_dict = {
            key: value.clone().detach().to(pl_module.device)
            for key, value in pl_module.state_dict().items()
        }

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx,
    ):
        if trainer.global_step < self.update_after_step:
            with torch.no_grad():
                for key, value in pl_module.state_dict().items():
                    self.ema_state_dict[key].copy_(value)
            return
        with torch.no_grad():
            for key, value in pl_module.state_dict().items():
                if value.dtype.is_floating_point:
                    self.ema_state_dict[key].mul_(self.decay).add_(
                        value, alpha=1.0 - self.decay
                    )
                else:
                    self.ema_state_dict[key].copy_(value)

    def _swap_to_ema_weights(self, pl_module: L.LightningModule):
        self.original_state_dict = {
            key: value.clone().detach().to(pl_module.device)
            for key, value in pl_module.state_dict().items()
        }
        pl_module.load_state_dict(self.ema_state_dict)

    def _restore_training_weights(self, pl_module: L.LightningModule):
        if self.original_state_dict:
            pl_module.load_state_dict(self.original_state_dict)
            self.original_state_dict = {}

    def on_validation_start(self, trainer, pl_module):
        self._swap_to_ema_weights(pl_module)

    def on_validation_end(self, trainer, pl_module):
        self._restore_training_weights(pl_module)

    def on_test_start(self, trainer, pl_module):
        self._swap_to_ema_weights(pl_module)

    def on_test_end(self, trainer, pl_module):
        self._restore_training_weights(pl_module)

    def on_save_checkpoint(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        checkpoint: Dict[str, Any],
    ):
        checkpoint["state_dict"] = {
            key: value.clone().cpu() for key, value in self.ema_state_dict.items()
        }
