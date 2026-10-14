"""Parquet embedding rows, pseudo-aggregated by gene and paired across modalities."""

import random
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if re.match(r"^d\d+$", column)]


class _AggregatedEmbeddingDataset(Dataset):
    """Dataset for loading pseud-aggregated embeddings from each modality.
    Each item is the mean of ``num_pairs`` rows that share ``pair_key``.

    Args:
        df: Table of embedding rows. Feature columns are named ``d`` plus digits.
        num_pairs: Number of rows that share ``pair_key`` to average into one embedding.
            Sampling is with replacement when a gene has fewer rows than ``num_pairs``.
        cols_to_keep: Metadata columns to retain with each item. ``"all"`` keeps every
            metadata column that has no missing values.
        perturbation_columns: Columns that identify the perturbation. The first entry is
            the key used to pair this modality with the other one.
    """

    pair_key: str

    def __init__(self, df: pd.DataFrame, num_pairs: int, cols_to_keep: list[str] | str):
        df = self._prepare(df.reset_index(drop=True))
        feature_columns = _feature_columns(df)
        metadata_columns = [
            column for column in df.columns if column not in feature_columns
        ]
        if cols_to_keep == "all":
            cols_to_keep = [
                column for column in metadata_columns if not df[column].isna().any()
            ]
        self.cols_to_keep = cols_to_keep
        self.X = torch.tensor(
            df[feature_columns].values, dtype=torch.float32
        ).share_memory_()
        self.md = df[self.cols_to_keep]
        self.num_pairs = num_pairs
        self.pair_index = (
            self.md.reset_index().groupby(self.pair_key)["index"].apply(list).to_dict()
        )
        max_index = max(max(indices) for indices in self.pair_index.values())
        if max_index != len(self.md) - 1:
            raise ValueError(
                f"Max index {max_index} is not equal to {len(self.md) - 1}"
            )

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def __len__(self):
        return len(self.md)

    def __getitem__(self, index):
        sample = self.md.iloc[index][self.cols_to_keep].to_dict()
        pair_indices = self.pair_index.get(sample[self.pair_key], [])
        if not pair_indices:
            # if sample has no replicates, use the sample itself as the embedding
            sample["embedding"] = self.X[index]
            return sample
        chosen = np.random.choice(
            pair_indices, self.num_pairs - 1, replace=self.num_pairs > len(pair_indices)
        )
        # add the sample itself to the chosen indices
        chosen = np.concatenate([chosen, [index]])
        try:
            sample["embedding"] = self.X[chosen].mean(axis=0)
        except (IndexError, ValueError):
            sample["embedding"] = self.X[index]
        return sample


class PerturbSeqEmbeddingDataset(_AggregatedEmbeddingDataset):
    """Perturb-seq embeddings. Rows are grouped by ``perturb_gene_level``."""

    pair_key = "perturb_gene_level"

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        if df[_feature_columns(df)].isna().any().any():
            raise ValueError("NaN values found in feature columns")
        return df


class CellPaintEmbeddingDataset(_AggregatedEmbeddingDataset):
    """Cell Painting embeddings. Rows are grouped by ``gene``."""

    pair_key = "gene"


class MultiModalDataset(Dataset):
    """Pair each transcriptomic row with a random Cell-Paint embedding of the same gene KO label.

    Args:
        tx_dataset: Perturb-seq embeddings grouped by ``perturb_gene_level``.
        ph_dataset: Cell Painting embeddings grouped by ``gene``.
    """

    def __init__(
        self,
        ps_dataset: PerturbSeqEmbeddingDataset,
        cp_dataset: CellPaintEmbeddingDataset,
    ):
        super().__init__()
        self.ps_dataset = ps_dataset
        self.cp_dataset = cp_dataset
        self.ps_pair_key = ps_dataset.pair_key
        self.cp_pair_key = cp_dataset.pair_key
        # filtering to common gene KO labels
        common_keys = set(ps_dataset.md[self.ps_pair_key]).intersection(
            set(cp_dataset.md[self.cp_pair_key])
        )
        ps_keep = ps_dataset.md[self.ps_pair_key].isin(common_keys)
        cp_keep = cp_dataset.md[self.cp_pair_key].isin(common_keys)
        ps_dataset.md = ps_dataset.md[ps_keep].reset_index(drop=True)
        ps_dataset.X = ps_dataset.X[ps_keep.values]
        cp_dataset.md = cp_dataset.md[cp_keep].reset_index(drop=True)
        cp_dataset.X = cp_dataset.X[cp_keep.values]
        # create a dictionary of Cell-Paint embedding indices for each gene KO label
        self.cp_pair_index = (
            cp_dataset.md.reset_index()
            .groupby(self.cp_pair_key)["index"]
            .apply(list)
            .to_dict()
        )

    def __len__(self):
        return len(self.ps_dataset)

    def __getitem__(self, index):
        ps_sample = self.ps_dataset[index]
        key = self.ps_dataset.md.iloc[index][self.ps_pair_key]
        cp_index = random.choice(self.cp_pair_index[key])
        return ps_sample, self.cp_dataset[cp_index]
