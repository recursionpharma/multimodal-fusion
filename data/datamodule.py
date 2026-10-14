"""Gene-stratified loader for multimodal-fusion parquet embeddings."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from data.dataset import (
    MultiModalDataset,
    PerturbSeqEmbeddingDataset,
    CellPaintEmbeddingDataset,
)


def _mean_by_gene(df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    feature_cols = [column for column in df.columns if re.match(r"^d\d+$", column)]
    metadata_cols = [
        column
        for column in df.columns
        if column not in feature_cols and column != group_column
    ]
    if not feature_cols:
        raise ValueError(
            "df must contain feature columns matching pattern d0001, d0002, ..."
        )
    agg = {column: "first" for column in metadata_cols}
    agg.update({column: "mean" for column in feature_cols})
    return df.groupby(group_column).agg(agg).reset_index().dropna(subset=feature_cols)


class EmbeddingDataModule(LightningDataModule):
    """Multimodal embedding datamodule with gene stratified train/val split.

    Args:
        ps_data_path: Perturb-seq embeddings path, or a list of paths to concatenate.
        cp_data_path: Cell Painting embeddings path, or a list of paths to concatenate.
        seed: Random seed for the train/validation split.
        val_fraction: Fraction of rows held out for validation. The split is stratified
            by ``ps_pair_key`` and ``cp_pair_key``.
        batch_size: Number of paired items per batch.
        num_workers: DataLoader worker processes. Values above ``0`` keep workers alive
            between epochs.
        pin_memory: Copy batches into pinned memory before moving them to the GPU.
        ps_pair_key: Perturb-seq column used to match genes and stratify the split.
        cp_pair_key: Cell Painting column used to match genes and stratify the split.
        ps_cols: Perturb-seq metadata columns to keep. ``"all"`` keeps every metadata
            column with no missing values.
        cp_cols: Cell Painting metadata columns to keep. ``"all"`` keeps every metadata
            column with no missing values.
        num_pairs: Number of replicates averaged into each train and validation embedding.
    """

    def __init__(
        self,
        ps_data_path: list[str] | str,
        cp_data_path: list[str] | str,
        seed: int = 42,
        val_fraction: float = 0.2,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        ps_pair_key: str = "perturb_gene_level",
        cp_pair_key: str = "gene",
        ps_cols: list[str] | str = "all",
        cp_cols: list[str] | str = "all",
        num_pairs: int = 4,
    ) -> None:
        super().__init__()
        self.ps_data_path = (
            ps_data_path if not isinstance(ps_data_path, str) else [ps_data_path]
        )
        self.cp_data_path = (
            cp_data_path if not isinstance(cp_data_path, str) else [cp_data_path]
        )
        self.seed = seed
        self.val_fraction = val_fraction
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.ps_pair_key = ps_pair_key
        self.cp_pair_key = cp_pair_key
        self.ps_cols = ps_cols
        self.cp_cols = cp_cols
        self.num_pairs = num_pairs

    def _pair(
        self, ps_df: pd.DataFrame, cp_df: pd.DataFrame, ps_cols, cp_cols
    ) -> MultiModalDataset:
        return MultiModalDataset(
            ps_dataset=PerturbSeqEmbeddingDataset(
                df=ps_df, num_pairs=self.num_pairs, cols_to_keep=ps_cols
            ),
            cp_dataset=CellPaintEmbeddingDataset(
                df=cp_df, num_pairs=self.num_pairs, cols_to_keep=cp_cols
            ),
        )

    def setup(self, stage: str | None = None) -> None:
        ps_frames = [pd.read_parquet(path) for path in self.ps_data_path]
        cp_frames = [pd.read_parquet(path) for path in self.cp_data_path]
        ps_df = pd.concat(ps_frames, ignore_index=True)
        cp_df = pd.concat(cp_frames, ignore_index=True)
        ps_df["perturbation_id"] = pd.factorize(ps_df[self.ps_pair_key])[0]

        # we need to filter to common gene KO labels to ensure that the train/val split is stratified
        common = set(ps_df[self.ps_pair_key]).intersection(set(cp_df[self.cp_pair_key]))
        if not common:
            raise ValueError("No common pairing keys found between modalities.")
        ps_df = ps_df[ps_df[self.ps_pair_key].isin(common)].reset_index(drop=True)
        cp_df = cp_df[cp_df[self.cp_pair_key].isin(common)].reset_index(drop=True)

        try:
            ps_train_idx, ps_val_idx = train_test_split(
                np.arange(len(ps_df)),
                test_size=self.val_fraction,
                stratify=ps_df[self.ps_pair_key],
                random_state=self.seed,
            )
            cp_train_idx, cp_val_idx = train_test_split(
                np.arange(len(cp_df)),
                test_size=self.val_fraction,
                stratify=cp_df[self.cp_pair_key],
                random_state=self.seed,
            )
        except ValueError:
            print(
                "Warning: stratified split failed. Using non-stratified split for train and validation."
            )
            ps_train_idx, ps_val_idx = train_test_split(
                np.arange(len(ps_df)),
                test_size=self.val_fraction,
                stratify=False,
                random_state=self.seed,
            )
            cp_train_idx, cp_val_idx = train_test_split(
                np.arange(len(cp_df)),
                test_size=self.val_fraction,
                stratify=False,
                random_state=self.seed,
            )

        self.train_dataset = self._pair(
            ps_df.iloc[ps_train_idx].reset_index(drop=True),
            cp_df.iloc[cp_train_idx].reset_index(drop=True),
            self.ps_cols,
            self.cp_cols,
        )
        self.val_dataset = self._pair(
            ps_df.iloc[ps_val_idx].reset_index(drop=True),
            cp_df.iloc[cp_val_idx].reset_index(drop=True),
            self.ps_cols,
            self.cp_cols,
        )
        ps_test = ps_frames[0]
        cp_test = cp_frames[0]
        ps_test = ps_test[ps_test[self.ps_pair_key].isin(common)].reset_index(drop=True)
        cp_test = cp_test[cp_test[self.cp_pair_key].isin(common)].reset_index(drop=True)
        self.test_dataset = self._pair(
            _mean_by_gene(ps_test, "perturb_gene_level"),
            _mean_by_gene(cp_test, "gene"),
            self.ps_cols,
            self.cp_cols,
        )
        print(
            f"Dataset ready: {len(self.train_dataset)} train rows, {len(self.val_dataset)} val rows."
        )

    def _loader(self, dataset, shuffle: bool, drop_last: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            drop_last=drop_last,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False, drop_last=True)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, shuffle=False, drop_last=False)
