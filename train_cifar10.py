import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from conformer.encoder import ConformerEncoder


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float


class Logger:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text("")

    def log(self, message: str) -> None:
        print(message, flush=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")


class ConformerImageClassifier(nn.Module):
    """
    Treat an image as a time-feature map:
    - x axis (image width) -> time
    - y axis (image height * channels) -> features
    """

    def __init__(
            self,
            num_classes: int,
            image_height: int,
            image_width: int,
            channels: int = 3,
            encoder_dim: int = 128,
            num_encoder_layers: int = 4,
            num_attention_heads: int = 4,
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            input_dropout_p: float = 0.1,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 15,
            half_step_residual: bool = True,
    ) -> None:
        super().__init__()
        input_dim = image_height * channels
        self.image_width = image_width
        self.encoder = ConformerEncoder(
            input_dim=input_dim,
            encoder_dim=encoder_dim,
            num_layers=num_encoder_layers,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            input_dropout_p=input_dropout_p,
            feed_forward_dropout_p=feed_forward_dropout_p,
            attention_dropout_p=attention_dropout_p,
            conv_dropout_p=conv_dropout_p,
            conv_kernel_size=conv_kernel_size,
            half_step_residual=half_step_residual,
        )
        self.classifier = nn.Linear(encoder_dim, num_classes)

    def _images_to_sequence(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, channels, height, width = images.size()
        if width != self.image_width:
            raise ValueError(f"Expected image width {self.image_width}, but received {width}")

        sequence = images.permute(0, 3, 1, 2).contiguous().view(batch_size, width, channels * height)
        lengths = torch.full(
            size=(batch_size,),
            fill_value=width,
            dtype=torch.long,
            device=images.device,
        )
        return sequence, lengths

    @staticmethod
    def _masked_mean_pool(encoded: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        time_steps = encoded.size(1)
        mask = torch.arange(time_steps, device=encoded.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).type_as(encoded)
        pooled = (encoded * mask).sum(dim=1) / lengths.clamp_min(1).unsqueeze(1).type_as(encoded)
        return pooled

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        sequence, lengths = self._images_to_sequence(images)
        encoded, output_lengths = self.encoder(sequence, lengths)
        pooled = self._masked_mean_pool(encoded, output_lengths)
        return self.classifier(pooled)


def build_transforms(dataset_name: str, image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    if dataset_name == "cifar10":
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    else:
        normalize = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))

    train_transform = transforms.Compose([
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, eval_transform


def apply_subset(dataset, limit: int):
    if limit <= 0:
        return dataset
    return torch.utils.data.Subset(dataset, range(min(limit, len(dataset))))


def build_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], int, int]:
    train_transform, eval_transform = build_transforms(args.dataset, args.image_size)

    if args.dataset == "cifar10":
        train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=train_transform)
        val_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=eval_transform)
        test_dataset = val_dataset
        num_classes = 10
        image_size = 32
    elif args.dataset == "tiny-imagenet":
        train_dataset = datasets.ImageFolder(root=str(Path(args.data_dir) / "train"), transform=train_transform)
        val_dataset = datasets.ImageFolder(root=str(Path(args.data_dir) / "val"), transform=eval_transform)
        test_root = Path(args.data_dir) / "test"
        test_dataset = None
        if (test_root / "images").exists() and any(test_root.iterdir()):
            test_dataset = None
        num_classes = len(train_dataset.classes)
        image_size = args.image_size
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    train_dataset = apply_subset(train_dataset, args.train_subset)
    val_dataset = apply_subset(val_dataset, args.val_subset)
    if test_dataset is not None:
        test_dataset = apply_subset(test_dataset, args.test_subset)

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs) if test_dataset is not None else None
    return train_loader, val_loader, test_loader, num_classes, image_size


def run_epoch(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        logger: Logger,
        epoch: int,
        stage: str,
        optimizer: torch.optim.Optimizer = None,
) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        batch_size = images.size(0)
        batch_correct = (logits.argmax(dim=1) == labels).sum().item()
        total_loss += loss.item() * batch_size
        total_correct += batch_correct
        total_examples += batch_size

        logger.log(
            f"epoch={epoch} stage={stage} batch={batch_idx}/{len(loader)} "
            f"batch_loss={loss.item():.4f} batch_acc={batch_correct / max(batch_size, 1):.4%} "
            f"running_loss={total_loss / total_examples:.4f} running_acc={total_correct / total_examples:.4%}"
        )

    return EpochMetrics(
        loss=total_loss / max(total_examples, 1),
        accuracy=total_correct / max(total_examples, 1),
    )


def make_model(args: argparse.Namespace, num_classes: int, image_size: int) -> ConformerImageClassifier:
    return ConformerImageClassifier(
        num_classes=num_classes,
        image_height=image_size,
        image_width=image_size,
        encoder_dim=args.encoder_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_attention_heads=args.num_attention_heads,
        feed_forward_expansion_factor=args.feed_forward_expansion_factor,
        conv_expansion_factor=args.conv_expansion_factor,
        input_dropout_p=args.input_dropout_p,
        feed_forward_dropout_p=args.feed_forward_dropout_p,
        attention_dropout_p=args.attention_dropout_p,
        conv_dropout_p=args.conv_dropout_p,
        conv_kernel_size=args.conv_kernel_size,
    )


def save_checkpoint(
        path: Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        epoch: int,
        best_val_accuracy: float,
        args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_accuracy": best_val_accuracy,
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(
        checkpoint_path: Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        device: torch.device = torch.device("cpu"),
) -> Dict:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Conformer-based classifier on CIFAR-10 or tiny-ImageNet.")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "tiny-imagenet"])
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./runs/image_conformer")
    parser.add_argument("--log-file", type=str, default="train_log.txt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--encoder-dim", type=int, default=128)
    parser.add_argument("--num-encoder-layers", type=int, default=4)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--feed-forward-expansion-factor", type=int, default=4)
    parser.add_argument("--conv-expansion-factor", type=int, default=2)
    parser.add_argument("--input-dropout-p", type=float, default=0.1)
    parser.add_argument("--feed-forward-dropout-p", type=float, default=0.1)
    parser.add_argument("--attention-dropout-p", type=float, default=0.1)
    parser.add_argument("--conv-dropout-p", type=float, default=0.1)
    parser.add_argument("--conv-kernel-size", type=int, default=15)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-subset", type=int, default=0)
    parser.add_argument("--val-subset", type=int, default=0)
    parser.add_argument("--test-subset", type=int, default=0)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--eval-only", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    if args.eval_only and not args.resume:
        raise ValueError("--eval-only requires --resume to point to a checkpoint")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(output_dir / args.log_file)
    logger.log(f"args={json.dumps(vars(args), ensure_ascii=False)}")
    logger.log(f"device={device}")

    train_loader, val_loader, test_loader, num_classes, image_size = build_dataloaders(args)
    logger.log(
        f"dataset={args.dataset} num_classes={num_classes} image_size={image_size} "
        f"train_batches={len(train_loader)} val_batches={len(val_loader)} "
        f"test_batches={len(test_loader) if test_loader is not None else 0}"
    )

    model = make_model(args, num_classes, image_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 1
    best_val_accuracy = 0.0

    if args.resume:
        checkpoint = load_checkpoint(Path(args.resume), model, optimizer, scheduler, device)
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_accuracy = checkpoint.get("best_val_accuracy", 0.0)
        logger.log(
            f"resumed_from={args.resume} next_epoch={start_epoch} best_val_acc={best_val_accuracy:.4%}"
        )

    if args.eval_only:
        val_metrics = run_epoch(model, val_loader, criterion, device, logger, start_epoch, "val")
        logger.log(
            f"eval_only val_loss={val_metrics.loss:.4f} val_acc={val_metrics.accuracy:.4%}"
        )
        if test_loader is not None:
            test_metrics = run_epoch(model, test_loader, criterion, device, logger, start_epoch, "test")
            logger.log(
                f"eval_only test_loss={test_metrics.loss:.4f} test_acc={test_metrics.accuracy:.4%}"
            )
        return

    history = []
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, logger, epoch, "train", optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device, logger, epoch, "val")
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_record = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_metrics.loss,
            "train_acc": train_metrics.accuracy,
            "val_loss": val_metrics.loss,
            "val_acc": val_metrics.accuracy,
        }
        history.append(epoch_record)

        logger.log(
            f"epoch={epoch} lr={current_lr:.6f} "
            f"train_loss={train_metrics.loss:.4f} train_acc={train_metrics.accuracy:.4%} "
            f"val_loss={val_metrics.loss:.4f} val_acc={val_metrics.accuracy:.4%}"
        )

        save_checkpoint(output_dir / "last.pt", model, optimizer, scheduler, epoch, best_val_accuracy, args)
        if val_metrics.accuracy >= best_val_accuracy:
            best_val_accuracy = val_metrics.accuracy
            save_checkpoint(output_dir / "best.pt", model, optimizer, scheduler, epoch, best_val_accuracy, args)
            logger.log(f"saved_best_checkpoint val_acc={best_val_accuracy:.4%}")

        scheduler.step()

    (output_dir / "history.json").write_text(json.dumps(history, indent=2))

    best_checkpoint = load_checkpoint(output_dir / "best.pt", model, device=device)
    best_epoch = best_checkpoint.get("epoch", args.epochs)
    summary = {
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "args": vars(args),
    }

    if test_loader is not None:
        test_metrics = run_epoch(model, test_loader, criterion, device, logger, best_epoch, "test")
        logger.log(
            f"best_epoch={best_epoch} best_val_acc={best_val_accuracy:.4%} "
            f"test_loss={test_metrics.loss:.4f} test_acc={test_metrics.accuracy:.4%}"
        )
        summary["test_loss"] = test_metrics.loss
        summary["test_accuracy"] = test_metrics.accuracy
    else:
        logger.log(
            f"best_epoch={best_epoch} best_val_acc={best_val_accuracy:.4%} "
            f"test_metrics=unavailable (dataset has no labels in test split)"
        )

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
