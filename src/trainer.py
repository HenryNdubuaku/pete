import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup

from src.benchmark import evaluate


def initialize_writer(name: str, is_master: bool) -> Optional[SummaryWriter]:
    if is_master:
        return SummaryWriter(log_dir=f"runs/{name}")
    return None


def save_best_model(embedder: torch.nn.Module, name: str):
    os.makedirs("weights", exist_ok=True)
    torch.save(embedder.state_dict(), f"weights/{name}.pt")
    print(f"New best model saved to weights/{name}.pt")


def setup_scheduler(optimizer, warmup_steps: int, total_steps: int):
    return get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )


def setup_scaler() -> GradScaler:
    return GradScaler(
        "cuda",
        init_scale=2.0**16,  # Initial scale (default: 2^16)
        growth_factor=2.0,  # Factor to increase the scale (default: 2.0)
        backoff_factor=0.5,  # Factor to decrease the scale (default: 0.5)
        growth_interval=2000,  # Steps before increasing the scale (default: 2000)
        enabled=True,  # Enable or disable the scaler (default: True)
    )


def log_metrics(
    writer: SummaryWriter,
    dataset_name: str,
    metric_name: str,
    value: float,
    global_step: int,
):
    writer.add_scalar(f"{dataset_name}/{metric_name}", value, global_step)


def train_loop(
    embedder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    experiment,
    name: str,
    writer: Optional[SummaryWriter],
    device: torch.device,
    is_ddp: bool = False,
    world_size: int = 1,
    rank: int = 0,
) -> torch.nn.Module:
    data = experiment.data
    dataset_names = experiment.train_datasets
    num_epochs = experiment.num_epochs
    warmup_steps = experiment.warmup_steps

    embedder.to(device)

    # Calculate total steps
    total_steps_per_epoch = sum(
        len(data.data_loaders[dataset_name]["train"])
        for dataset_name in dataset_names
        if "train" in data.data_loaders[dataset_name]
    )
    total_steps = total_steps_per_epoch * num_epochs

    scheduler = setup_scheduler(optimizer, warmup_steps, total_steps)
    scaler = setup_scaler()

    global_step = 0
    best_stsb_score = -float("inf")

    for epoch in range(num_epochs):
        if rank == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}]")

        for dataset_name in dataset_names:
            embedder.train()
            total_train_loss = 0.0
            train_loader = data.data_loaders[dataset_name]["train"]

            if is_ddp:
                train_loader.sampler.set_epoch(epoch)

            val_split_keys = [
                key
                for key in data.data_loaders[dataset_name].keys()
                if "validation" in key
            ]
            val_loader = None
            if val_split_keys:
                val_loader_key = val_split_keys[0]
                val_loader = data.data_loaders[dataset_name][val_loader_key]
                if is_ddp:
                    val_sampler = DistributedSampler(
                        val_loader.dataset,
                        num_replicas=world_size,
                        rank=rank,
                        shuffle=False,
                    )
                    val_loader = DataLoader(
                        val_loader.dataset,
                        batch_size=val_loader.batch_size,
                        sampler=val_sampler,
                        num_workers=val_loader.num_workers,
                        pin_memory=True,
                    )

            for batch in train_loader:
                batch = tuple(t.to(device) for t in batch)
                # optimizer.zero_grad()

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    train_loss = embedder(batch)

                scaler.scale(train_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                total_train_loss += train_loss.item()
                global_step += 1

                if writer and rank == 0:
                    log_metrics(
                        writer,
                        dataset_name,
                        "train_loss",
                        train_loss.item(),
                        global_step,
                    )

            avg_train_loss = total_train_loss / len(train_loader)

            if val_loader is not None:
                embedder.eval()
                total_val_loss = 0.0

                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        for batch in val_loader:
                            batch = tuple(t.to(device) for t in batch)
                            val_loss = embedder(batch)
                            total_val_loss += val_loss.item()

                avg_val_loss = total_val_loss / (len(val_loader) + 1e-6)

                if writer and rank == 0:
                    print(
                        f"{dataset_name} Train Loss: {avg_train_loss:.4f}, "
                        f"{dataset_name} Validation Loss: {avg_val_loss:.4f}"
                    )
                    log_metrics(
                        writer, dataset_name, "val_loss", avg_val_loss, global_step
                    )
            else:
                if writer and rank == 0:
                    print(
                        f"{dataset_name} Train Loss: {avg_train_loss:.4f}, "
                        f"No validation data for {dataset_name}."
                    )

            if writer and rank == 0:
                # For DDP, use the underlying model
                model_to_evaluate = embedder.module if is_ddp else embedder
                # Evaluate on validation datasets (e.g., stsb for contrastive training)
                eval_dataset = experiment.validation_datasets[0] if experiment.validation_datasets else dataset_name
                results = evaluate(
                    model_to_evaluate, data.data_loaders, device, eval_dataset, name
                )
                print(f"{eval_dataset} Validation: {results}")

                for metric, score in results.items():
                    log_metrics(writer, dataset_name, metric, score, global_step)

                values = list(results.values())
                current_stsb_score = sum(values) / len(values)

                if current_stsb_score > best_stsb_score:
                    best_stsb_score = current_stsb_score
                    if is_ddp:
                        save_best_model(embedder.module, name)
                    else:
                        save_best_model(embedder, name)

        if rank == 0:
            print("")

    if writer and rank == 0:
        # For DDP, use the underlying model
        model_to_evaluate = embedder.module if is_ddp else embedder
        evaluate(
            model_to_evaluate, data.data_loaders, device, dataset_name, name, test=True
        )

    return embedder


def train(
    embedder: torch.nn.Module, optimizer: torch.optim.Optimizer, experiment, name: str
):
    """
    Single-GPU training function.
    """
    device = experiment.device
    writer = initialize_writer(name, is_master=True)

    embedder = train_loop(
        embedder=embedder,
        optimizer=optimizer,
        experiment=experiment,
        name=name,
        writer=writer,
        device=device,
        is_ddp=False,
    )

    if writer:
        writer.close()

    return embedder


def train_worker(
    rank: int,
    world_size: int,
    embedder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    experiment,
    name: str,
    backend: str = "nccl",
):
    """
    Distributed training worker function.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "9999"  # Choose any free port

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    embedder.to(device)
    embedder = torch.nn.parallel.DistributedDataParallel(embedder, device_ids=[rank])

    is_master = rank == 0
    writer = initialize_writer(name, is_master)

    if is_master:
        os.makedirs("weights", exist_ok=True)

    data = experiment.data
    for dataset_name in experiment.train_datasets:
        train_loader = data.data_loaders[dataset_name]["train"]
        train_sampler = DistributedSampler(
            train_loader.dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        data.data_loaders[dataset_name]["train"] = DataLoader(
            train_loader.dataset,
            batch_size=train_loader.batch_size,
            sampler=train_sampler,
            num_workers=train_loader.num_workers,
            pin_memory=True,
        )

    embedder = train_loop(
        embedder=embedder,
        optimizer=optimizer,
        experiment=experiment,
        name=name,
        writer=writer,
        device=device,
        is_ddp=True,
        world_size=world_size,
        rank=rank,
    )

    if writer and is_master:
        writer.close()

    # Clean up
    dist.destroy_process_group()


def train_ddp(
    embedder: torch.nn.Module, optimizer: torch.optim.Optimizer, experiment, name: str
):
    """
    Main training function that sets up distributed training.
    """
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise ValueError("No GPUs available for training.")

    mp.spawn(
        train_worker,
        args=(world_size, embedder, optimizer, experiment, name),
        nprocs=world_size,
        join=True,
    )
