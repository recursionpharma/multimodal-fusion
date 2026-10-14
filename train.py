"""Train the multimodal-fusion model."""

from typing import Any

import hydra
import lightning as L
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig

torch.set_float32_matmul_precision("medium")


def _instantiate_group(cfg) -> list[Any]:
    if not cfg:
        return []
    return [
        hydra.utils.instantiate(item)
        for item in cfg.values()
        if isinstance(item, DictConfig) and "_target_" in item
    ]


def train(cfg: DictConfig) -> None:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)
    callbacks = _instantiate_group(cfg.get("callbacks"))
    loggers = _instantiate_group(cfg.get("logger"))
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=loggers,
        enable_checkpointing=any(
            isinstance(callback, ModelCheckpoint) for callback in callbacks
        ),
    )
    if cfg.get("train"):
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
    if cfg.get("test"):
        ckpt_path = trainer.checkpoint_callback.best_model_path or None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
