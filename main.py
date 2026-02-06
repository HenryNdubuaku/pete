import argparse
import os

import numpy as np
import torch
from torch.optim import AdamW
from transformers import BertTokenizer

torch.manual_seed(0)
np.random.seed(0)

from src.data import GlueDatasetLoader
from src.pete import PETE
from src.trainer import train
from src.transformer import Transformer
from src.utils import timer
from src.benchmark import GLUEWrapper


class Experiment:
    def __init__(
        self,
        args,
        vocab_size: int = 30552,
        d_model: int = 128,
        num_hidden_layers: int = 1,
        num_attention_heads: int = 1,
        dropout_prob: float = 0.2,
        max_seq_len: int = 128,
        num_epochs: int = 5,
        batch_size: int = 256,
        learning_rate: float = 2e-5,
        warmup_steps: int = 1000,
        train_datasets: list = ["snli", "mnli"],
        validation_datasets: list = ["stsb"],
        include_baseline=False,
        num_outputs=None,
        num_sentences=None,
    ):

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_dropout_prob = dropout_prob
        self.attention_probs_dropout_prob = dropout_prob
        self.max_seq_len = max_seq_len
        self.tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.train_datasets = train_datasets
        self.validation_datasets = validation_datasets
        self.include_baseline = include_baseline

        self.data = GlueDatasetLoader(
            tokenizer=self.tokenizer,
            max_length=self.max_seq_len,
            batch_size=self.batch_size,
            dataset_names=self.train_datasets + self.validation_datasets,
        )

        pete = PETE(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            max_seq_len=self.max_seq_len,
            permute_tokens=getattr(args, 'permute_tokens', False),
            random_embeddings=getattr(args, 'random_embeddings', False),
            index_mode=getattr(args, 'index_mode', 'raw'),
            index_scale=getattr(args, 'index_scale', 1.0),
            rff_sigma=getattr(args, 'rff_sigma', None),
        )

        if args.benchmark:
            Embedder = GLUEWrapper
        else:
            from src.embedder import Embedder

        if args.from_pretrained or args.benchmark:
            weight_path = os.path.join(
                "pretrained_weights", "pete_" + args.config + ".pt"
            )
            state_dict = torch.load(weight_path, map_location=torch.device("cuda"))
            pete.load_state_dict(state_dict)

        self.pete_embedder = Embedder(pete, num_outputs, num_sentences)
        self.pete_optimizer = AdamW(self.pete_embedder.parameters(), lr=self.learning_rate)
        print(
            f"\nNum of params PETE: {sum(p.numel() for p in pete.parameters() if p.requires_grad)}"
        )

        if self.include_baseline:
            transformer = Transformer(
                vocab_size=self.vocab_size,
                d_model=self.d_model,
                num_hidden_layers=self.num_hidden_layers,
                num_attention_heads=self.num_attention_heads,
                max_position_embeddings=self.max_seq_len,
                hidden_dropout_prob=self.hidden_dropout_prob,
                attention_probs_dropout_prob=self.attention_probs_dropout_prob,
                intermediate_size=0,
            )

            if args.from_pretrained:
                weight_path = os.path.join(
                    "pretrained_weights", "transformer_" + args.config + ".pt"
                )
                state_dict = torch.load(weight_path, map_location=torch.device("cuda"))
                transformer.load_state_dict(state_dict)

            self.transformer_embedder = Embedder(
                transformer, num_outputs, num_sentences
            )
            self.transformer_optimizer = AdamW(
                self.transformer_embedder.parameters(), lr=self.learning_rate
            )
            print(
                f"Num of params in Transformer: {sum(p.numel() for p in transformer.parameters() if p.requires_grad)}"
            )


def run(experiment, suffix=None, ablation=None):
    if suffix is None:
        suffix = f"{experiment.num_hidden_layers}_{experiment.d_model}"

    if experiment.include_baseline:
        print("\nTraining Transformer\n")
        name = f"transformer_{suffix}"
        with timer(f"Transformer training ({name})"):
            transformer_embedder = train(
                experiment.transformer_embedder,
                experiment.transformer_optimizer,
                experiment,
                name,
            )

    print("\nTraining PETE\n")
    name = f"pete_{suffix}"
    if ablation:
        name = f"{name}_{ablation}"
    with timer(f"PETE training ({name})"):
        pete_embedder = train(
            experiment.pete_embedder, experiment.pete_optimizer, experiment, name
        )


def run_benchmark(args):
    # Datasets:[num_outputs, num_sentences]
    benchmark_datasets = {
        "stsb": [0, 2],
        "rte": [2, 2],
        "cola": [2, 1],
        "qnli": [2, 2], 
        "wnli": [2, 2],
        "sst2": [2, 1],
        "mrpc": [2, 2], 
        "qqp":  [2, 2],
        "mnli": [3, 2],
        "ax": [3, 2],
    }

    configs = [
        # [1, 64],
        [1, 128],
        [1, 256],
        [1, 512],
        # [2, 64],
        [2, 128],
        [2, 256],
        [2, 512],
        ]

    for config in configs:
        for dataset, info in benchmark_datasets.items():
            num_outputs, num_sentences = info

            args.config = "_".join([str(item) for item in config])


            experiment = Experiment(
                args,
                batch_size=256,
                learning_rate=2e-5,
                warmup_steps=args.warmup_steps,
                train_datasets=[dataset],
                validation_datasets=[dataset],
                include_baseline=args.include_baseline,
                dropout_prob=args.dropout_prob,
                max_seq_len=args.max_seq_len,
                vocab_size=args.vocab_size,
                num_outputs=num_outputs,
                num_sentences=num_sentences,
                num_hidden_layers=config[0],
                num_attention_heads=config[0],
                d_model=config[1],
                num_epochs=30,
            )

            run(experiment, config)


def main():
    parser = argparse.ArgumentParser(
        description="Train models with various parameters."
    )

    parser.add_argument(
        "--num-hidden-layers",
        nargs="+",
        type=int,
        default=[1],
        help="List of num_hidden_layers to try.",
    )
    parser.add_argument(
        "--d-model",
        nargs="+",
        type=int,
        default=[128],
        help="List of d_model dimensions to try.",
    )
    parser.add_argument("--num-epochs", type=int, default=5, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-5, help="Learning rate."
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=1000, help="Number of warmup steps."
    )
    parser.add_argument(
        "--train-datasets",
        nargs="+",
        default=["snli", "mnli"],
        help="List of training datasets.",
    )
    parser.add_argument(
        "--validation-datasets",
        nargs="+",
        default=["stsb"],
        help="List of validation datasets.",
    )
    parser.add_argument(
        "--include-baseline",
        action="store_true",
        help="Whether to train the baseline Transformer.",
    )
    parser.add_argument(
        "--from-pretrained",
        action="store_true",
        help="Whether to load pretrained configuration",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="if to benchmark on glue",
    )
    parser.add_argument(
        "--dropout-prob", type=float, default=0.2, help="Dropout probability."
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=128, help="Maximum sequence length."
    )
    parser.add_argument(
        "--vocab-size", type=int, default=30552, help="Vocabulary size."
    )
    parser.add_argument(
        "--permute-tokens",
        action="store_true",
        help="Randomly permute token IDs before Fourier embedding (ablation).",
    )
    parser.add_argument(
        "--random-embeddings",
        action="store_true",
        help="Replace Fourier features with Random Fourier Features (ablation).",
    )
    parser.add_argument(
        "--index-mode",
        type=str,
        default="raw",
        choices=["raw", "normalized", "scaled"],
        help="Index mapping: 'raw' (x=p), 'normalized' (x=2*(p/(V-1))-1), 'scaled' (x=scale*p).",
    )
    parser.add_argument(
        "--index-scale",
        type=float,
        default=1.0,
        help="Scale factor when using --index-mode=scaled.",
    )
    parser.add_argument(
        "--rff-sigma",
        type=float,
        default=None,
        help="Frequency scale for Random Fourier Features (defaults to mean of inv_freq).",
    )

    args = parser.parse_args()

    if args.benchmark:
        from src.benchmark import GLUEWrapper as Embedder

        run_benchmark(args)
        return

    from src.embedder import Embedder

    for n in args.num_hidden_layers:
        for dim in args.d_model:
            experiment = Experiment(
                args,
                num_hidden_layers=n,
                num_attention_heads=n,
                d_model=dim,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                warmup_steps=args.warmup_steps,
                train_datasets=args.train_datasets,
                validation_datasets=args.validation_datasets,
                include_baseline=args.include_baseline,
                dropout_prob=args.dropout_prob,
                max_seq_len=args.max_seq_len,
                vocab_size=args.vocab_size,
            )

            # Build ablation suffix
            ablation_parts = []
            if args.permute_tokens:
                ablation_parts.append("permute")
            if args.random_embeddings:
                ablation_parts.append("rff")
            if args.index_mode != "raw":
                ablation_parts.append(args.index_mode)
            ablation = "_".join(ablation_parts) if ablation_parts else None
            run(experiment, ablation=ablation)
            return


if __name__ == "__main__":
    main()
