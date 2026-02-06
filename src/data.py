import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from datasets import DatasetDict, Value, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader, TensorDataset
from transformers import PreTrainedTokenizer

torch.manual_seed(0)

CACHE_DIR = ".cache/tokenized"


def get_cache_path(name: str, split: str, max_length: int) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{name}_{split}_{max_length}.pt")


def tokenize_texts(sentences, tokenizer, max_length):
    """
    Tokenize input sentences using the provided tokenizer and return the encoded outputs.
    """
    return tokenizer(
        sentences,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def filter_entailment(ds_split, label_key, entailment_value):
    """
    Filter the dataset split to select only the examples where the label corresponds to entailment.
    """
    filtered_indices = [
        i for i, label in enumerate(ds_split[label_key]) if label == entailment_value
    ]
    return ds_split.select(filtered_indices)


def create_dataloader(
    tokenizer,
    ds_split,
    batch_size,
    max_length,
    sentence1_key="sentence1",
    sentence2_key=None,
    label_key=None,
    return_tensors=False,
):
    sentences1 = list(ds_split[sentence1_key])
    encodings1 = tokenize_texts(sentences1, tokenizer, max_length)

    # Cast attention_mask to float32
    encodings1["attention_mask"] = encodings1["attention_mask"].float()

    tensors = [
        encodings1["input_ids"],
        encodings1["attention_mask"],
    ]

    if sentence2_key is not None:
        sentences2 = list(ds_split[sentence2_key])
        encodings2 = tokenize_texts(sentences2, tokenizer, max_length)

        # Cast attention_mask to float32
        encodings2["attention_mask"] = encodings2["attention_mask"].float()

        tensors.extend([encodings2["input_ids"], encodings2["attention_mask"]])

    if label_key is not None:
        labels = list(ds_split[label_key])
        tensors.append(torch.tensor(labels, dtype=torch.float32))

    dataset = TensorDataset(*tensors)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        pin_memory=True, num_workers=0  # num_workers=0 is fine for TensorDataset
    )

    if return_tensors:
        return dataloader, tensors
    return dataloader


@dataclass
class GlueDatasetLoader:
    dataset_names: List[str] = field(
        default_factory=lambda: [
            "mnli",
            "stsb",
            "mrpc",
            "rte",
            "qqp",
            "qnli",
            "sst2",
            "cola",
            "wnli",
            "ax",
            "paws",
            "mnli+qqp",
            "snli",
        ]
    )
    tokenizer: PreTrainedTokenizer = None
    max_length: int = None
    batch_size: int = None
    datasets: Dict[str, DatasetDict] = field(init=False, default_factory=dict)
    data_loaders: Dict[str, Dict[str, DataLoader]] = field(
        init=False, default_factory=dict
    )
    positive_label_values: Dict[str, int] = field(
        init=False,
        default_factory=lambda: {
            #'ax': 0,     # 'entailment'
            #'cola': 1,   # 'acceptable'
            # "mnli": 0,  # 'entailment'
            #'mrpc': 1,   # 'equivalent'
            #'qnli': 0,   # 'entailment'
            #'qqp': 1,    # 'duplicate'
            # 'rte': 0,    # 'entailment'
            #'sst2': 1,   # 'positive'
            #'wnli': 1,   # 'entailment'
            #'paws': 1,   # 'paraphrase'
            # "snli": 0,  # 'entailment'
            # Exclude 'stsb' as it's a regression task
        },
    )

    def __post_init__(self):
        self.load_datasets()
        if self.tokenizer is not None:
            self.create_all_dataloaders()

    def load_datasets(self):
        """
        Load all specified GLUE datasets, PAWS, and SNLI.
        """
        for name in self.dataset_names:
            if name == "paws":
                ds = load_dataset("paws", "labeled_final")
            elif name == "mnli+qqp":
                # Load MNLI and QQP datasets
                ds_mnli = load_dataset("glue", "mnli")
                ds_qqp = load_dataset("glue", "qqp")
                ds = {}
                for split in set(ds_mnli.keys()).intersection(ds_qqp.keys()):
                    # Rename columns to unify them
                    split_mnli = ds_mnli[split].rename_columns(
                        {"premise": "sentence1", "hypothesis": "sentence2"}
                    )
                    split_qqp = ds_qqp[split].rename_columns(
                        {"question1": "sentence1", "question2": "sentence2"}
                    )

                    # Adjust MNLI labels: 'entailment' (0) -> 1 (positive), others -> 0 (negative)
                    split_mnli = split_mnli.map(
                        lambda example: {"label": 1 if example["label"] == 0 else 0}
                    )
                    split_mnli = split_mnli.cast_column("label", Value("int64"))

                    # Cast 'label' column in QQP to Value('int64') to match MNLI
                    split_qqp = split_qqp.cast_column("label", Value("int64"))

                    # Concatenate datasets
                    combined_split = concatenate_datasets([split_mnli, split_qqp])
                    ds[split] = combined_split
                self.datasets[name] = DatasetDict(ds)
                print(f"Loaded {name} dataset with splits: {list(ds.keys())}")
            elif name == "snli":
                ds = load_dataset("snli")
                self.datasets[name] = ds
                print(f"Loaded {name} dataset with splits: {list(ds.keys())}")
            else:
                ds = load_dataset("glue", name)
                self.datasets[name] = ds
                print(f"Loaded {name} dataset with splits: {list(ds.keys())}")

    def create_all_dataloaders(self):
        """
        Create DataLoaders for all splits in all datasets.
        Uses caching to speed up subsequent runs.
        """
        for name, ds in self.datasets.items():
            self.data_loaders[name] = {}
            for split in ds.keys():
                cache_path = get_cache_path(name, split, self.max_length)

                # Try to load from cache
                if os.path.exists(cache_path):
                    tensors = torch.load(cache_path)
                    dataset = TensorDataset(*tensors)
                    dataloader = DataLoader(
                        dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
                    )
                    self.data_loaders[name][split] = dataloader
                    print(f"Loaded cached DataLoader for {name} [{split}] with {len(dataloader)} batches.")
                    continue

                ds_split = ds[split]

                sentence1_key, sentence2_key = self.get_sentence_keys(name)
                label_key = "label" if "label" in ds_split.column_names else None

                if name in self.positive_label_values and label_key is not None:
                    entailment_value = self.positive_label_values[name]
                    ds_split = filter_entailment(ds_split, label_key, entailment_value)

                    if len(ds_split) == 0:
                        print(
                            f"Skipping {name} [{split}] after filtering; no positive samples found."
                        )
                        continue  # Skip creating DataLoader for this split

                elif label_key is None:
                    print(
                        f"Labels not available for {name} [{split}]; skipping filtering."
                    )

                if len(ds_split) == 0:
                    print(
                        f"Skipping {name} [{split}]; dataset split is empty after processing."
                    )
                    continue  # Skip creating DataLoader for this split

                dataloader, tensors = create_dataloader(
                    tokenizer=self.tokenizer,
                    ds_split=ds_split,
                    batch_size=self.batch_size,
                    max_length=self.max_length,
                    sentence1_key=sentence1_key,
                    sentence2_key=sentence2_key,
                    label_key=label_key,
                    return_tensors=True,
                )

                # Save to cache
                torch.save(tensors, cache_path)

                self.data_loaders[name][split] = dataloader
                print(
                    f"Created DataLoader for {name} [{split}] with {len(dataloader)} batches."
                )

    @staticmethod
    def get_sentence_keys(name: str) -> (str, Optional[str]):
        """
        Determine the keys for sentences based on the dataset name.
        """
        if name in ["sst2", "cola"]:
            sentence1_key = "sentence"
            sentence2_key = None
        elif name == "stsb":
            sentence1_key = "sentence1"
            sentence2_key = "sentence2"
        elif name == "qqp":
            sentence1_key = "question1"
            sentence2_key = "question2"
        elif name == "qnli":
            sentence1_key = "question"
            sentence2_key = "sentence"
        elif name in ["mnli", "ax", "snli"]:
            sentence1_key = "premise"
            sentence2_key = "hypothesis"
        elif name == "paws":
            sentence1_key = "sentence1"
            sentence2_key = "sentence2"
        elif name == "mnli+qqp":
            sentence1_key = "sentence1"
            sentence2_key = "sentence2"
        else:  # mrpc, rte, wnli
            sentence1_key = "sentence1"
            sentence2_key = "sentence2"
        return sentence1_key, sentence2_key
