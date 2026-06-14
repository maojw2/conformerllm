import argparse
import json
import random
from argparse import Namespace
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF

from train_cifar10 import get_effective_image_size, make_model


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BACKGROUND_COLOR = (0, 0, 255)
FOREGROUND_COLOR = (255, 0, 0)
SEPARATOR_COLOR = (255, 255, 255)


def load_checkpoint(path: Path, device: torch.device) -> Dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")


def target_to_class_id(target) -> int:
    if isinstance(target, tuple):
        return int(target[0])
    return int(target)


def build_export_dataset(dataset_name: str, data_dir: str, split: str):
    if dataset_name == "cifar10":
        dataset = datasets.CIFAR10(
            root=data_dir,
            train=split == "train",
            download=True,
            transform=None,
        )
        normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        class_names = list(dataset.classes)
    elif dataset_name == "oxford-pet":
        dataset = datasets.OxfordIIITPet(
            root=data_dir,
            split="trainval" if split == "train" else "test",
            target_types=["category", "segmentation"],
            download=True,
        )
        normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        class_names = list(getattr(dataset, "classes", []))
    else:
        raise ValueError(f"Export only supports segmentation datasets, got: {dataset_name}")

    if not class_names:
        max_class_id = max(target_to_class_id(dataset[index][1]) for index in range(len(dataset)))
        class_names = [f"class_{class_id:02d}" for class_id in range(max_class_id + 1)]

    return dataset, class_names, normalize


def select_indices_by_class(dataset, class_names: List[str], per_class: int, seed: int) -> Dict[int, List[int]]:
    rng = random.Random(seed)
    by_class: Dict[int, List[int]] = {class_id: [] for class_id in range(len(class_names))}
    for index in range(len(dataset)):
        _, target = dataset[index]
        class_id = target_to_class_id(target)
        if class_id in by_class:
            by_class[class_id].append(index)

    selected: Dict[int, List[int]] = {}
    for class_id, indices in by_class.items():
        if len(indices) < per_class:
            raise ValueError(
                f"Class {class_names[class_id]} has only {len(indices)} images, "
                f"but --per-class requested {per_class}."
            )
        selected[class_id] = rng.sample(indices, per_class)
    return selected


def mask_to_color_image(mask: torch.Tensor, scale: int) -> Image.Image:
    height, width = mask.shape
    image = Image.new("RGB", (width, height), BACKGROUND_COLOR)
    pixels = image.load()
    mask_cpu = mask.cpu()
    for y in range(height):
        for x in range(width):
            if int(mask_cpu[y, x].item()) == 1:
                pixels[x, y] = FOREGROUND_COLOR
    if scale != 1:
        image = image.resize((width * scale, height * scale), Image.Resampling.NEAREST)
    return image


def make_comparison_image(original: Image.Image, mask_image: Image.Image, scale: int, gap: int) -> Image.Image:
    original = original.convert("RGB")
    if scale != 1:
        original = original.resize((original.width * scale, original.height * scale), Image.Resampling.NEAREST)

    height = max(original.height, mask_image.height)
    width = original.width + gap + mask_image.width
    comparison = Image.new("RGB", (width, height), SEPARATOR_COLOR)
    comparison.paste(original, (0, 0))
    comparison.paste(mask_image, (original.width + gap, 0))
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export segmentation examples as masks or original/mask comparisons."
    )
    parser.add_argument("--checkpoint", type=str, default="runs/oxford_pet_segmentation/best.pt")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="runs/oxford_pet_segmentation/val_blue_red_masks")
    parser.add_argument("--output-kind", type=str, default="mask", choices=["mask", "comparison"])
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", type=int, default=8)
    parser.add_argument("--gap", type=int, default=8)
    parser.add_argument("--split", type=str, default="val", choices=["val", "train"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(checkpoint_path, device)
    model_args = Namespace(**checkpoint["args"])
    model_args.task = "segmentation"
    image_size = get_effective_image_size(model_args.dataset, model_args.image_size)

    model = make_model(model_args, num_classes=2, image_size=image_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset, class_names, normalize = build_export_dataset(model_args.dataset, args.data_dir, args.split)
    selected = select_indices_by_class(dataset, class_names, args.per_class, args.seed)

    manifest = {
        "checkpoint": str(checkpoint_path),
        "dataset": model_args.dataset,
        "split": args.split,
        "per_class": args.per_class,
        "output_kind": args.output_kind,
        "background_color": BACKGROUND_COLOR,
        "foreground_color": FOREGROUND_COLOR,
        "separator_color": SEPARATOR_COLOR,
        "classes": {},
    }

    with torch.no_grad():
        for class_id, indices in selected.items():
            class_name = class_names[class_id]
            class_dir = output_dir / f"{class_id:02d}_{safe_name(class_name)}"
            class_dir.mkdir(parents=True, exist_ok=True)
            manifest["classes"][class_name] = []

            for sample_number, dataset_index in enumerate(indices, start=1):
                image, target = dataset[dataset_index]
                image = image.convert("RGB")
                model_image = TF.resize(
                    image,
                    size=(image_size, image_size),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                )
                image_tensor = normalize(TF.to_tensor(model_image)).unsqueeze(0).to(device)
                logits = model(image_tensor)
                prediction = logits.argmax(dim=1).squeeze(0)

                mask_image = mask_to_color_image(prediction, scale=args.scale)
                if args.output_kind == "comparison":
                    output_image = make_comparison_image(model_image, mask_image, scale=args.scale, gap=args.gap)
                    filename = f"{sample_number:02d}_idx{dataset_index:05d}_comparison.png"
                else:
                    output_image = mask_image
                    filename = f"{sample_number:02d}_idx{dataset_index:05d}_mask.png"

                output_path = class_dir / filename
                output_image.save(output_path)

                manifest["classes"][class_name].append(
                    {
                        "dataset_index": dataset_index,
                        "target": target_to_class_id(target),
                        "file": str(output_path),
                    }
                )

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"saved {len(class_names) * args.per_class} {args.output_kind} images to {output_dir}")


if __name__ == "__main__":
    main()
