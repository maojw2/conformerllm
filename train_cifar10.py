import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF

from conformer.encoder import ConformerEncoder


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    mean_iou: Optional[float] = None


class Logger:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text("")

    def log(self, message: str) -> None:
        print(message, flush=True)
        for attempt in range(20):
            try:
                with self.log_path.open("a", encoding="utf-8") as handle:
                    handle.write(message + "\n")
                return
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.05)


class DiceCELoss(nn.Module):
    def __init__(
            self,
            ignore_index: int = 255,
            ce_weight: float = 1.0,
            dice_weight: float = 1.0,
            smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.cross_entropy(logits, targets)
        valid_mask = targets != self.ignore_index
        if not valid_mask.any():
            return ce_loss

        probabilities = torch.softmax(logits, dim=1)
        safe_targets = targets.masked_fill(~valid_mask, 0)
        one_hot_targets = F.one_hot(safe_targets, num_classes=logits.size(1))
        one_hot_targets = one_hot_targets.permute(0, 3, 1, 2).type_as(probabilities)

        valid_mask = valid_mask.unsqueeze(1).type_as(probabilities)
        probabilities = probabilities * valid_mask
        one_hot_targets = one_hot_targets * valid_mask

        reduce_dims = (0, 2, 3)
        intersections = (probabilities * one_hot_targets).sum(dim=reduce_dims)
        cardinalities = probabilities.sum(dim=reduce_dims) + one_hot_targets.sum(dim=reduce_dims)
        target_pixels = one_hot_targets.sum(dim=reduce_dims)
        present_classes = target_pixels > 0
        if not present_classes.any():
            return ce_loss

        dice_scores = (2.0 * intersections + self.smooth) / (cardinalities + self.smooth)
        dice_loss = 1.0 - dice_scores[present_classes].mean()
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


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
            encoder_block_mode: str = "full",
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
            block_mode=encoder_block_mode,
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


class ConformerImageSegmenter(nn.Module):
    """
    Predict a dense foreground/background mask from images.

    The encoder still treats image columns as sequence steps. Each encoded
    column predicts a vertical strip of the mask, then the strip logits are
    upsampled back to the original image width.
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
            encoder_block_mode: str = "full",
    ) -> None:
        super().__init__()
        input_dim = image_height * channels
        self.image_height = image_height
        self.image_width = image_width
        self.num_classes = num_classes
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
            block_mode=encoder_block_mode,
        )
        self.segmentation_head = nn.Linear(encoder_dim, num_classes * image_height)

    def _images_to_sequence(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, channels, height, width = images.size()
        if height != self.image_height or width != self.image_width:
            raise ValueError(
                f"Expected image size {(self.image_height, self.image_width)}, "
                f"but received {(height, width)}"
            )

        sequence = images.permute(0, 3, 1, 2).contiguous().view(batch_size, width, channels * height)
        lengths = torch.full(
            size=(batch_size,),
            fill_value=width,
            dtype=torch.long,
            device=images.device,
        )
        return sequence, lengths

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        sequence, lengths = self._images_to_sequence(images)
        encoded, _ = self.encoder(sequence, lengths)
        strip_logits = self.segmentation_head(encoded)
        batch_size, low_width, _ = strip_logits.size()
        logits = strip_logits.view(batch_size, low_width, self.num_classes, self.image_height)
        logits = logits.permute(0, 2, 3, 1).contiguous()
        return F.interpolate(
            logits,
            size=(self.image_height, self.image_width),
            mode="bilinear",
            align_corners=False,
        )


def get_effective_image_size(dataset_name: str, requested_image_size: int) -> int:
    if dataset_name == "cifar10":
        return 32
    return requested_image_size


def build_transforms(dataset_name: str, image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    if dataset_name == "cifar10":
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    elif dataset_name == "imagenet-mini":
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    else:
        normalize = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))

    if dataset_name == "imagenet-mini":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        eval_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])
    else:
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


def tensor_quantile(values: torch.Tensor, quantile: float) -> torch.Tensor:
    flat_values = values.flatten()
    if flat_values.numel() == 0:
        raise ValueError("Cannot compute a quantile for an empty tensor")
    clamped_quantile = min(max(quantile, 0.0), 1.0)
    kth_index = int(clamped_quantile * (flat_values.numel() - 1)) + 1
    return flat_values.kthvalue(kth_index).values


def make_pseudo_foreground_mask(
        image: torch.Tensor,
        foreground_quantile: float,
        min_distance: float,
) -> torch.Tensor:
    """Build a CIFAR-10 foreground/background pseudo mask from RGB contrast."""
    _, height, width = image.size()
    border_pixels = torch.cat(
        [
            image[:, 0, :],
            image[:, -1, :],
            image[:, :, 0],
            image[:, :, -1],
        ],
        dim=1,
    )
    background_color = border_pixels.mean(dim=1).view(3, 1, 1)
    color_distance = torch.sqrt(torch.sum((image - background_color) ** 2, dim=0))

    y_coords = torch.linspace(-1.0, 1.0, height).view(height, 1)
    x_coords = torch.linspace(-1.0, 1.0, width).view(1, width)
    center_penalty = (x_coords ** 2 + y_coords ** 2).clamp(max=1.0)
    centered_score = color_distance * (1.0 - 0.35 * center_penalty)

    threshold = max(tensor_quantile(centered_score, foreground_quantile).item(), min_distance)
    mask = (centered_score > threshold).float().view(1, 1, height, width)
    mask = F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1)
    mask = (mask >= 0.35).float()
    mask = F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1)
    mask = mask.view(height, width) >= 0.5

    foreground_ratio = mask.float().mean().item()
    if foreground_ratio < 0.03 or foreground_ratio > 0.80:
        ellipse = ((x_coords / 0.75) ** 2 + (y_coords / 0.75) ** 2) <= 1.0
        mask = ellipse

    return mask.long()


class CIFAR10SegmentationDataset(Dataset):
    """CIFAR-10 images with deterministic foreground/background pseudo masks."""

    def __init__(
            self,
            root: str,
            train: bool,
            image_size: int,
            download: bool,
            foreground_quantile: float,
            min_distance: float,
    ) -> None:
        self.dataset = datasets.CIFAR10(root=root, train=train, download=download, transform=None)
        self.train = train
        self.image_size = image_size
        self.foreground_quantile = foreground_quantile
        self.min_distance = min_distance
        self.normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, _ = self.dataset[index]
        if self.train:
            image = TF.pad(image, padding=4)
            top, left, height, width = transforms.RandomCrop.get_params(
                image,
                output_size=(self.image_size, self.image_size),
            )
            image = TF.crop(image, top, left, height, width)
            if torch.rand(()) < 0.5:
                image = TF.hflip(image)
        elif image.size != (self.image_size, self.image_size):
            image = TF.resize(image, size=(self.image_size, self.image_size))

        image_tensor = TF.to_tensor(image)
        mask = make_pseudo_foreground_mask(
            image_tensor,
            foreground_quantile=self.foreground_quantile,
            min_distance=self.min_distance,
        )
        return self.normalize(image_tensor), mask


def oxford_pet_trimap_to_binary_mask(trimap, ignore_index: int) -> torch.Tensor:
    target = torch.as_tensor(np.array(trimap, dtype=np.uint8), dtype=torch.long)
    mask = torch.full_like(target, fill_value=ignore_index)
    mask[target == 2] = 0
    mask[target == 1] = 1
    return mask


class OxfordPetSegmentationDataset(Dataset):
    """Oxford-IIIT Pet images with real trimap segmentation labels."""

    def __init__(
            self,
            root: str,
            split: str,
            image_size: int,
            train: bool,
            download: bool,
            ignore_index: int,
    ) -> None:
        self.dataset = datasets.OxfordIIITPet(
            root=root,
            split=split,
            target_types="segmentation",
            download=download,
        )
        self.image_size = image_size
        self.train = train
        self.ignore_index = ignore_index
        self.normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, trimap = self.dataset[index]
        image = image.convert("RGB")
        image = TF.resize(
            image,
            size=(self.image_size, self.image_size),
            interpolation=transforms.InterpolationMode.BILINEAR,
        )
        trimap = TF.resize(
            trimap,
            size=(self.image_size, self.image_size),
            interpolation=transforms.InterpolationMode.NEAREST,
        )

        if self.train and torch.rand(()) < 0.5:
            image = TF.hflip(image)
            trimap = TF.hflip(trimap)

        image_tensor = self.normalize(TF.to_tensor(image))
        mask = oxford_pet_trimap_to_binary_mask(trimap, ignore_index=self.ignore_index)
        return image_tensor, mask


def apply_subset(dataset, limit: int):
    if limit <= 0:
        return dataset
    return torch.utils.data.Subset(dataset, range(min(limit, len(dataset))))


def has_imagefolder_classes(root: Path) -> bool:
    if not root.is_dir():
        return False
    return any(child.is_dir() for child in root.iterdir())


def resolve_split_dir(data_dir: Path, split: str, containers: Iterable[Path]) -> Path:
    candidates = [data_dir / split]
    candidates.extend(data_dir / container / split for container in containers)

    for candidate in candidates:
        if has_imagefolder_classes(candidate):
            return candidate

    formatted_candidates = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"Could not find an ImageFolder-style '{split}' split under {data_dir}.\n"
        f"Checked:\n{formatted_candidates}"
    )


def build_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], int, int]:
    image_size = get_effective_image_size(args.dataset, args.image_size)

    if args.task == "segmentation":
        if args.dataset == "cifar10":
            train_dataset = CIFAR10SegmentationDataset(
                root=args.data_dir,
                train=True,
                image_size=image_size,
                download=True,
                foreground_quantile=args.mask_foreground_quantile,
                min_distance=args.mask_min_distance,
            )
            val_dataset = CIFAR10SegmentationDataset(
                root=args.data_dir,
                train=False,
                image_size=image_size,
                download=True,
                foreground_quantile=args.mask_foreground_quantile,
                min_distance=args.mask_min_distance,
            )
            test_dataset = val_dataset
            num_classes = 2
        elif args.dataset == "oxford-pet":
            train_dataset = OxfordPetSegmentationDataset(
                root=args.data_dir,
                split="trainval",
                image_size=image_size,
                train=True,
                download=True,
                ignore_index=args.ignore_index,
            )
            val_dataset = OxfordPetSegmentationDataset(
                root=args.data_dir,
                split="test",
                image_size=image_size,
                train=False,
                download=True,
                ignore_index=args.ignore_index,
            )
            test_dataset = val_dataset
            num_classes = 2
        else:
            raise ValueError("Segmentation mode currently supports --dataset cifar10 or --dataset oxford-pet")
    else:
        train_transform, eval_transform = build_transforms(args.dataset, image_size)

        if args.dataset == "cifar10":
            train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=train_transform)
            val_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=eval_transform)
            test_dataset = val_dataset
            num_classes = 10
        elif args.dataset == "tiny-imagenet":
            train_dataset = datasets.ImageFolder(root=str(Path(args.data_dir) / "train"), transform=train_transform)
            val_dataset = datasets.ImageFolder(root=str(Path(args.data_dir) / "val"), transform=eval_transform)
            test_root = Path(args.data_dir) / "test"
            test_dataset = None
            if (test_root / "images").exists() and any(test_root.iterdir()):
                test_dataset = None
            num_classes = len(train_dataset.classes)
        elif args.dataset == "imagenet-mini":
            data_dir = Path(args.data_dir)
            containers = (
                Path("imagenet-mini"),
                Path("imagenetmini-1000") / "imagenet-mini",
            )
            train_root = resolve_split_dir(data_dir, "train", containers)
            val_root = resolve_split_dir(data_dir, "val", containers)
            train_dataset = datasets.ImageFolder(root=str(train_root), transform=train_transform)
            val_dataset = datasets.ImageFolder(root=str(val_root), transform=eval_transform)
            test_dataset = None
            num_classes = len(train_dataset.classes)
            if train_dataset.class_to_idx != val_dataset.class_to_idx:
                raise ValueError("ImageNet Mini train and val class folders do not match")
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


def segmentation_confusion_matrix(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        num_classes: int,
) -> torch.Tensor:
    predictions = predictions.view(-1).to(torch.long)
    targets = targets.view(-1).to(torch.long)
    valid_mask = (targets >= 0) & (targets < num_classes)
    if valid_mask.sum().item() == 0:
        return torch.zeros(num_classes, num_classes, device=targets.device, dtype=torch.float64)

    encoded = num_classes * targets[valid_mask] + predictions[valid_mask]
    return torch.bincount(encoded, minlength=num_classes * num_classes).view(num_classes, num_classes).double()


def metrics_from_confusion(confusion: torch.Tensor) -> Tuple[float, float]:
    total = confusion.sum().item()
    pixel_accuracy = confusion.diag().sum().item() / max(total, 1.0)

    intersections = confusion.diag()
    unions = confusion.sum(dim=1) + confusion.sum(dim=0) - intersections
    valid_classes = unions > 0
    if valid_classes.any().item():
        mean_iou = (intersections[valid_classes] / unions[valid_classes].clamp_min(1.0)).mean().item()
    else:
        mean_iou = 0.0

    return pixel_accuracy, mean_iou


def primary_metric(metrics: EpochMetrics, task: str) -> float:
    if task == "segmentation":
        return metrics.mean_iou or 0.0
    return metrics.accuracy


def primary_metric_name(task: str) -> str:
    if task == "segmentation":
        return "mean_iou"
    return "accuracy"


def format_metrics(metrics: EpochMetrics, prefix: str, task: str) -> str:
    if task == "segmentation":
        return (
            f"{prefix}_loss={metrics.loss:.4f} "
            f"{prefix}_pixel_acc={metrics.accuracy:.4%} "
            f"{prefix}_mean_iou={(metrics.mean_iou or 0.0):.4%}"
        )
    return f"{prefix}_loss={metrics.loss:.4f} {prefix}_acc={metrics.accuracy:.4%}"


def run_epoch(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        logger: Logger,
        epoch: int,
        stage: str,
        task: str,
        num_classes: int,
        optimizer: torch.optim.Optimizer = None,
) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.float64) if task == "segmentation" else None

    for batch_idx, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, targets)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

        if task == "segmentation":
            predictions = logits.argmax(dim=1)
            batch_confusion = segmentation_confusion_matrix(predictions, targets, num_classes)
            confusion += batch_confusion.cpu()
            batch_pixel_acc, batch_mean_iou = metrics_from_confusion(batch_confusion.cpu())
            running_pixel_acc, running_mean_iou = metrics_from_confusion(confusion)
            logger.log(
                f"epoch={epoch} stage={stage} batch={batch_idx}/{len(loader)} "
                f"batch_loss={loss.item():.4f} batch_pixel_acc={batch_pixel_acc:.4%} "
                f"batch_mean_iou={batch_mean_iou:.4%} "
                f"running_loss={total_loss / total_examples:.4f} "
                f"running_pixel_acc={running_pixel_acc:.4%} running_mean_iou={running_mean_iou:.4%}"
            )
        else:
            batch_correct = (logits.argmax(dim=1) == targets).sum().item()
            total_correct += batch_correct
            logger.log(
                f"epoch={epoch} stage={stage} batch={batch_idx}/{len(loader)} "
                f"batch_loss={loss.item():.4f} batch_acc={batch_correct / max(batch_size, 1):.4%} "
                f"running_loss={total_loss / total_examples:.4f} "
                f"running_acc={total_correct / total_examples:.4%}"
            )

    if task == "segmentation":
        pixel_accuracy, mean_iou = metrics_from_confusion(confusion)
        return EpochMetrics(
            loss=total_loss / max(total_examples, 1),
            accuracy=pixel_accuracy,
            mean_iou=mean_iou,
        )

    return EpochMetrics(
        loss=total_loss / max(total_examples, 1),
        accuracy=total_correct / max(total_examples, 1),
    )


def make_model(args: argparse.Namespace, num_classes: int, image_size: int) -> nn.Module:
    model_cls = ConformerImageSegmenter if args.task == "segmentation" else ConformerImageClassifier
    return model_cls(
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
        encoder_block_mode=getattr(args, "encoder_block_mode", "full"),
    )


def save_checkpoint(
        path: Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        epoch: int,
        best_val_score: float,
        args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_score": best_val_score,
        "best_val_metric": primary_metric_name(args.task),
        "args": vars(args),
    }
    if args.task == "segmentation":
        checkpoint["best_val_mean_iou"] = best_val_score
    else:
        checkpoint["best_val_accuracy"] = best_val_score

    torch.save(
        checkpoint,
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
    parser = argparse.ArgumentParser(
        description="Train a Conformer image model for Oxford Pet/CIFAR-10 segmentation or image classification."
    )
    parser.add_argument("--task", type=str, default="segmentation", choices=["segmentation", "classification"])
    parser.add_argument(
        "--dataset",
        type=str,
        default="oxford-pet",
        choices=["oxford-pet", "cifar10", "tiny-imagenet", "imagenet-mini"],
    )
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./runs/oxford_pet_segmentation")
    parser.add_argument("--log-file", type=str, default="train_log.txt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss", type=str, default="ce-dice", choices=["ce", "ce-dice"])
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
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
    parser.add_argument(
        "--encoder-block-mode",
        type=str,
        default="full",
        choices=["full", "attention-only", "convolution-only"],
        help="Use the full Conformer block, remove convolution, or remove attention for ablation.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-subset", type=int, default=0)
    parser.add_argument("--val-subset", type=int, default=0)
    parser.add_argument("--test-subset", type=int, default=0)
    parser.add_argument("--mask-foreground-quantile", type=float, default=0.65)
    parser.add_argument("--mask-min-distance", type=float, default=0.08)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--resume-model-only", action="store_true")
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
    if args.task == "segmentation" and args.dataset == "cifar10":
        if not 0.0 < args.mask_foreground_quantile < 1.0:
            raise ValueError("--mask-foreground-quantile must be between 0 and 1")
        if args.mask_min_distance < 0.0:
            raise ValueError("--mask-min-distance must be non-negative")
    if args.task == "segmentation" and args.ignore_index < 0:
        raise ValueError("--ignore-index must be non-negative")
    if args.ce_weight < 0.0 or args.dice_weight < 0.0:
        raise ValueError("--ce-weight and --dice-weight must be non-negative")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(output_dir / args.log_file)
    logger.log(f"args={json.dumps(vars(args), ensure_ascii=False)}")
    logger.log(f"device={device}")

    train_loader, val_loader, test_loader, num_classes, image_size = build_dataloaders(args)
    logger.log(
        f"task={args.task} dataset={args.dataset} num_classes={num_classes} image_size={image_size} "
        f"train_batches={len(train_loader)} val_batches={len(val_loader)} "
        f"test_batches={len(test_loader) if test_loader is not None else 0}"
    )

    model = make_model(args, num_classes, image_size).to(device)
    if args.task == "segmentation":
        if args.loss == "ce-dice":
            criterion = DiceCELoss(
                ignore_index=args.ignore_index,
                ce_weight=args.ce_weight,
                dice_weight=args.dice_weight,
            )
        else:
            criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_index)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 1
    best_val_score = 0.0
    best_metric_name = primary_metric_name(args.task)

    if args.resume:
        if args.resume_model_only:
            checkpoint = load_checkpoint(Path(args.resume), model, device=device)
            logger.log(
                f"loaded_model_only_from={args.resume} restart_epoch={start_epoch} lr={args.lr:.6f}"
            )
        else:
            checkpoint = load_checkpoint(Path(args.resume), model, optimizer, scheduler, device)
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_val_score = checkpoint.get(
                "best_val_score",
                checkpoint.get("best_val_mean_iou" if args.task == "segmentation" else "best_val_accuracy", 0.0),
            )
            logger.log(
                f"resumed_from={args.resume} next_epoch={start_epoch} "
                f"best_val_{best_metric_name}={best_val_score:.4%}"
            )

    if args.eval_only:
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            logger,
            start_epoch,
            "val",
            args.task,
            num_classes,
        )
        logger.log(
            f"eval_only {format_metrics(val_metrics, 'val', args.task)}"
        )
        if test_loader is not None:
            test_metrics = run_epoch(
                model,
                test_loader,
                criterion,
                device,
                logger,
                start_epoch,
                "test",
                args.task,
                num_classes,
            )
            logger.log(
                f"eval_only {format_metrics(test_metrics, 'test', args.task)}"
            )
        return

    history = []
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            logger,
            epoch,
            "train",
            args.task,
            num_classes,
            optimizer,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            logger,
            epoch,
            "val",
            args.task,
            num_classes,
        )
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_record = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_metrics.loss,
            "val_loss": val_metrics.loss,
        }
        if args.task == "segmentation":
            epoch_record.update(
                {
                    "train_pixel_accuracy": train_metrics.accuracy,
                    "train_mean_iou": train_metrics.mean_iou,
                    "val_pixel_accuracy": val_metrics.accuracy,
                    "val_mean_iou": val_metrics.mean_iou,
                }
            )
        else:
            epoch_record.update(
                {
                    "train_acc": train_metrics.accuracy,
                    "val_acc": val_metrics.accuracy,
                }
            )
        history.append(epoch_record)

        logger.log(
            f"epoch={epoch} lr={current_lr:.6f} "
            f"{format_metrics(train_metrics, 'train', args.task)} "
            f"{format_metrics(val_metrics, 'val', args.task)}"
        )

        val_score = primary_metric(val_metrics, args.task)
        if val_score >= best_val_score:
            best_val_score = val_score
            save_checkpoint(output_dir / "best.pt", model, optimizer, scheduler, epoch, best_val_score, args)
            logger.log(f"saved_best_checkpoint val_{best_metric_name}={best_val_score:.4%}")
        save_checkpoint(output_dir / "last.pt", model, optimizer, scheduler, epoch, best_val_score, args)

        scheduler.step()

    (output_dir / "history.json").write_text(json.dumps(history, indent=2))

    best_checkpoint = load_checkpoint(output_dir / "best.pt", model, device=device)
    best_epoch = best_checkpoint.get("epoch", args.epochs)
    summary = {
        "best_epoch": best_epoch,
        "best_val_metric": best_metric_name,
        "best_val_score": best_val_score,
        "args": vars(args),
    }
    if args.task == "segmentation":
        summary["best_val_mean_iou"] = best_val_score
    else:
        summary["best_val_accuracy"] = best_val_score

    if test_loader is not None:
        test_metrics = run_epoch(
            model,
            test_loader,
            criterion,
            device,
            logger,
            best_epoch,
            "test",
            args.task,
            num_classes,
        )
        logger.log(
            f"best_epoch={best_epoch} best_val_{best_metric_name}={best_val_score:.4%} "
            f"{format_metrics(test_metrics, 'test', args.task)}"
        )
        summary["test_loss"] = test_metrics.loss
        if args.task == "segmentation":
            summary["test_pixel_accuracy"] = test_metrics.accuracy
            summary["test_mean_iou"] = test_metrics.mean_iou
        else:
            summary["test_accuracy"] = test_metrics.accuracy
    else:
        logger.log(
            f"best_epoch={best_epoch} best_val_{best_metric_name}={best_val_score:.4%} "
            f"test_metrics=unavailable (dataset has no labels in test split)"
        )

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
