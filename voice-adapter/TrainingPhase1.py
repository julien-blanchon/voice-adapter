import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
import wandb
import argparse
import glob
import time
from SNACGeneratorModel import SNACGenerator, SNACConfig
import torch.distributed as dist
import psutil

_HABANA_AVAILABLE = os.getenv("HABANA", "0") == "1"

# Constants
SNAC_VOCAB_SIZE = 4096
DEVICE = torch.device("hpu")
PROCESSED_DIR = "/scratch-2/final_embeddings"


def log_system_info():
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    print(f"CPU Usage: {cpu_percent}%")
    print(
        f"Memory Usage: {memory.percent}% (Used: {memory.used / 1e9:.2f} GB, Available: {memory.available / 1e9:.2f} GB)"
    )
    print(
        f"Disk Usage: {disk.percent}% (Used: {disk.used / 1e9:.2f} GB, Free: {disk.free / 1e9:.2f} GB)"
    )


class ChunkLoader(Dataset):
    def __init__(self, chunk_file):
        self.chunk_file = chunk_file
        self.data = torch.load(chunk_file, mmap=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Ensure all components are tensors and of the correct type
        sample["embedding"] = sample["embedding"].to(
            torch.float32
        )  # From bfloat16 as needs to match the initial model weight type
        sample["attention_mask"] = sample["attention_mask"].to(torch.long)
        sample["snac_tokens"] = torch.tensor(
            [int(token) for token in sample["snac_tokens"]], dtype=torch.long
        )

        return sample


class LargeScaleDataset(Dataset):
    def __init__(self, chunk_files):
        self.chunks = []
        # Load just one chunk temporarily for debugging
        for file in tqdm(chunk_files, desc="Loading dataset chunks"):
            self.chunks.append(ChunkLoader(file))
            if file == chunk_files[0]:
                break

    def __len__(self):
        return sum(len(chunk) for chunk in self.chunks)

    def __getitem__(self, idx):
        for chunk in self.chunks:
            if idx < len(chunk):
                return chunk[idx]
            idx -= len(chunk)
        raise IndexError("Sample index out of range")


class ShardedDataLoader:
    def __init__(
        self, dataset, batch_size, num_workers, world_size, rank, shuffle=True
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.world_size = world_size
        self.rank = rank
        self.shuffle = shuffle
        self.epoch = 0

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        indices = indices[self.rank :: self.world_size]

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            yield self._collate_batch([self.dataset[idx] for idx in batch_indices])

    def _collate_batch(self, batch):
        embeddings = [item["embedding"] for item in batch]
        attention_masks = [item["attention_mask"] for item in batch]
        snac_tokens = [item["snac_tokens"] for item in batch]

        # Ensure the shapes are consistent
        max_seq_length = max([e.size(0) for e in embeddings])
        padded_embeddings = torch.zeros(
            len(batch), max_seq_length, embeddings[0].size(1)
        )
        padded_attention_masks = torch.zeros(len(batch), max_seq_length)
        max_snac_length = max([s.size(0) for s in snac_tokens])
        padded_snac_tokens = torch.full(
            (len(batch), max_snac_length), -100, dtype=torch.long
        )

        for i, (embedding, attention_mask, snac_token) in enumerate(
            zip(embeddings, attention_masks, snac_tokens)
        ):
            seq_length = embedding.size(0)
            padded_embeddings[i, :seq_length] = embedding
            padded_attention_masks[i, :seq_length] = attention_mask
            snac_length = snac_token.size(0)
            padded_snac_tokens[i, :snac_length] = snac_token

        return {
            "embedding": padded_embeddings,
            "attention_mask": padded_attention_masks,
            "snac_tokens": padded_snac_tokens,
        }

    def __len__(self):
        return (len(self.dataset) + self.batch_size * self.world_size - 1) // (
            self.batch_size * self.world_size
        )

    def set_epoch(self, epoch):
        self.epoch = epoch


class GradientAccumulator:
    def __init__(self, model, optimizer, accumulation_steps):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.current_step = 0

    def backward(self, loss):
        (loss / self.accumulation_steps).backward()
        self.current_step += 1

    def step(self):
        if self.current_step % self.accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            return True
        return False

    def zero_grad(self):
        self.optimizer.zero_grad()
        self.current_step = 0


class CustomLRScheduler:
    def __init__(
        self,
        optimizer,
        mode="min",
        factor=0.1,
        patience=10,
        verbose=False,
        min_lr=0,
        cooldown=0,
    ):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.min_lr = min_lr
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.best = None
        self.num_bad_epochs = 0
        self.last_epoch = -1

    def step(self, metrics, epoch=None):
        current = metrics
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown():
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # reset num_bad_epochs

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group["lr"])
            new_lr = max(old_lr * self.factor, self.min_lr)
            if old_lr - new_lr > 1e-8:
                param_group["lr"] = new_lr
                if self.verbose:
                    print(
                        f"Epoch {epoch}: reducing learning rate of group {i} to {new_lr:.4e}."
                    )

    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if best is None:
            return True
        return a < best if self.mode == "min" else a > best


def train_snac_generator(
    model,
    train_loader,
    test_loader,
    optimizer,
    scheduler,
    accumulator,
    num_epochs,
    x_tokens,
    rank,
):
    if _HABANA_AVAILABLE:
        from habana_frameworks.torch.hpex.experimental.transformer_engine import recipe  # type: ignore

        # Create an FP8 recipe
        fp8_recipe = recipe.DelayedScaling(fp8_format=recipe.Format.HYBRID)
        fp8_context = lambda: fp8_context()
    else:
        from torch.cuda.amp import autocast

        fp8_recipe = None
        fp8_context = lambda: autocast(enabled=True)

    print("Starting training")
    model.train()
    best_test_loss = float("inf")

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}/{num_epochs}")
        model.train()
        total_loss = 0
        total_samples = 0

        train_loader.set_epoch(epoch)
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=rank != 0
        )
        for batch_idx, batch in enumerate(progress_bar):
            if batch_idx % 100 == 0:
                log_system_info()

            optimizer.zero_grad()  # Clear gradients at the start of each batch

            llm_embeddings = batch["embedding"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            snac_tokens = batch["snac_tokens"].to(DEVICE)

            current_input = llm_embeddings
            total_loss_batch = 0

            for i in range(0, snac_tokens.size(1), x_tokens):
                position_ids = torch.arange(
                    current_input.size(1), dtype=torch.long, device=DEVICE
                )
                position_ids = position_ids.unsqueeze(0).expand(
                    current_input.size(0), -1
                )

                try:
                    with fp8_context():
                        with torch.autograd.set_detect_anomaly(True):
                            snac_logits = model(
                                current_input,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                            )
                except RuntimeError as e:
                    print(f"Error in model forward pass: {e}")
                    print(f"current_input shape: {current_input.shape}")
                    print(f"attention_mask shape: {attention_mask.shape}")
                    print(f"position_ids shape: {position_ids.shape}")
                    raise

                target_snac = snac_tokens[:, i : i + x_tokens]
                if target_snac.size(1) < x_tokens:
                    pad_length = x_tokens - target_snac.size(1)
                    stop_tokens = torch.zeros(
                        target_snac.size(0), pad_length, dtype=torch.long, device=DEVICE
                    )
                    target_snac = torch.cat([target_snac, stop_tokens], dim=1)

                loss = F.cross_entropy(
                    snac_logits[:, -x_tokens:].view(-1, SNAC_VOCAB_SIZE),
                    target_snac.view(-1),
                    ignore_index=-100,
                )
                total_loss_batch += loss

                with torch.no_grad():
                    predicted_snac = snac_logits[:, -x_tokens:].argmax(dim=-1)
                    snac_embeddings = model.snac_proj.weight[predicted_snac]
                    current_input = torch.cat([current_input, snac_embeddings], dim=1)
                    attention_mask = torch.cat(
                        [
                            attention_mask,
                            torch.ones(
                                (attention_mask.size(0), x_tokens), device=DEVICE
                            ),
                        ],
                        dim=1,
                    )

            total_loss_batch.backward()  # Backward pass on accumulated loss
            if _HABANA_AVAILABLE:
                # Execute after .backward()
                htcore.mark_step()  # type: ignore  # noqa: F821
            optimizer.step()

            total_loss += total_loss_batch.item()
            total_samples += llm_embeddings.size(0)
            progress_bar.set_postfix(
                {"loss": total_loss_batch.item() / llm_embeddings.size(0)}
            )

        avg_train_loss = total_loss / total_samples

        print("Starting validation")
        model.eval()
        test_loss = 0
        test_samples = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing", disable=rank != 0):
                llm_embeddings = batch["embedding"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                snac_tokens = batch["snac_tokens"].to(DEVICE)

                current_input = llm_embeddings

                for i in range(0, snac_tokens.size(1), x_tokens):
                    position_ids = torch.arange(
                        current_input.size(1), dtype=torch.long, device=DEVICE
                    )
                    position_ids = position_ids.unsqueeze(0).expand(
                        current_input.size(0), -1
                    )

                    with fp8_context():
                        snac_logits = model(
                            current_input,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                        )

                    target_snac = snac_tokens[:, i : i + x_tokens]
                    if target_snac.size(1) < x_tokens:
                        pad_length = x_tokens - target_snac.size(1)
                        stop_tokens = torch.zeros(
                            target_snac.size(0),
                            pad_length,
                            dtype=torch.long,
                            device=DEVICE,
                        )
                        target_snac = torch.cat([target_snac, stop_tokens], dim=1)

                    loss = F.cross_entropy(
                        snac_logits[:, -x_tokens:].view(-1, SNAC_VOCAB_SIZE),
                        target_snac.view(-1),
                        ignore_index=-100,
                    )
                    test_loss += loss.item() * llm_embeddings.size(0)
                    test_samples += llm_embeddings.size(0)

                    predicted_snac = snac_logits[:, -x_tokens:].argmax(dim=-1)
                    snac_embeddings = model.module.snac_proj.weight[predicted_snac]
                    current_input = torch.cat([current_input, snac_embeddings], dim=1)
                    attention_mask = torch.cat(
                        [
                            attention_mask,
                            torch.ones(
                                (attention_mask.size(0), x_tokens), device=DEVICE
                            ),
                        ],
                        dim=1,
                    )

        avg_test_loss = test_loss / test_samples

        scheduler.step(avg_test_loss)

        if rank == 0:
            print(
                f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}"
            )
            wandb.log({"train_loss": avg_train_loss, "test_loss": avg_test_loss})

            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                torch.save(
                    model.module.state_dict()
                    if hasattr(model, "module")
                    else model.state_dict(),
                    "best_snac_generator.pth",
                )
                print("New best model saved!")


def main():
    parser = argparse.ArgumentParser(description="Train SNAC Generator")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--x_tokens", type=int, default=30)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading",
    )
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps",
    )
    args = parser.parse_args()
    # torch.autograd.set_detect_anomaly(True)

    print("Starting script")
    log_system_info()

    print("Setting up HPU environment")
    if _HABANA_AVAILABLE:
        import habana_frameworks.torch.core as htcore  # type: ignore

        htcore.hpu_set_env()

    # Initialize the HPU distributed environment
    try:
        if _HABANA_AVAILABLE:
            import habana_frameworks.torch.distributed.hccl  # type: ignore

            world_size, rank, local_rank = (
                habana_frameworks.torch.distributed.hccl.initialize_distributed_hpu()
            )
            print(
                f"Distributed setup complete. World size: {world_size}, Rank: {rank}, Local rank: {local_rank}"
            )
        else:
            # raise NotImplementedError("Pytorch DDP not yet implemented for non-Habana")
            world_size, rank, local_rank = 1, 0, 0
    except RuntimeError:
        print("Running in single-process mode")
        world_size, rank, local_rank = 1, 0, 0

    if rank == 0:
        print("Initializing wandb")
        wandb.init(project="snac-generator", config=vars(args))

    print("Loading datasets")
    train_files = sorted(
        glob.glob(os.path.join(PROCESSED_DIR, "train_embeddings_chunk_*.pt"))
    )
    test_files = sorted(
        glob.glob(os.path.join(PROCESSED_DIR, "test_embeddings_chunk_*.pt"))
    )
    train_dataset = LargeScaleDataset(train_files)
    test_dataset = LargeScaleDataset(test_files)

    print("Creating data loaders")
    train_loader = ShardedDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        world_size=world_size,
        rank=rank,
        shuffle=True,
    )
    test_loader = ShardedDataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        world_size=world_size,
        rank=rank,
        shuffle=False,
    )

    print("Initializing model")
    config = SNACConfig()
    model = SNACGenerator(config).to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    if world_size > 1:
        print("Setting up DistributedDataParallel")
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank]
        )

    print("Initializing optimizer")
    if _HABANA_AVAILABLE:
        from habana_frameworks.torch.hpex.optimizers import FusedAdamW  # type: ignore

        optimizer = FusedAdamW(model.parameters(), lr=args.learning_rate)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    print("Setting up custom learning rate scheduler")
    scheduler = CustomLRScheduler(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        verbose=True,
        min_lr=1e-6,
        cooldown=0,
    )

    # print("Setting up FP8 recipe")
    # fp8_recipe = recipe.DelayedScaling(fp8_format=recipe.Format.HYBRID)

    print("Setting up gradient accumulator")
    accumulator = GradientAccumulator(model, optimizer, args.accumulation_steps)

    print("Starting training")
    train_snac_generator(
        model,
        train_loader,
        test_loader,
        optimizer,
        scheduler,
        accumulator,
        num_epochs=args.num_epochs,
        x_tokens=args.x_tokens,
        # fp8_recipe=fp8_recipe,
        rank=rank,
    )

    if rank == 0:
        print("Saving final model")
        torch.save(
            model.module.state_dict()
            if hasattr(model, "module")
            else model.state_dict(),
            "final_snac_generator.pth",
        )
        wandb.finish()

    print("Script completed")

    # Clean up the distributed environment if it was initialized
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    start_time = time.time()
    print(f"Script started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        main()
    except Exception as e:
        print(f"An error occurred during script execution: {str(e)}")
    finally:
        end_time = time.time()
        print(f"Script ended at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total execution time: {end_time - start_time:.2f} seconds")
