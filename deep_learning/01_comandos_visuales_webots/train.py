"""Train and evaluate MobileNetV3-Small on the Webots command dataset."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from torchvision.models import MobileNet_V3_Small_Weights


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train MobileNetV3-Small to classify Webots visual commands."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=project_root / "data" / "synthetic_commands",
        help="Dataset root containing train, val, and test directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "runs" / "mobilenet_v3_small",
        help="Directory for checkpoints, metrics, and predictions.",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument(
        "--head-epochs",
        type=int,
        default=3,
        help="Initial epochs with the convolutional backbone frozen.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fine-tune-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=4012)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="auto uses CUDA when available and otherwise CPU.",
    )
    parser.add_argument(
        "--weights",
        choices=("imagenet", "none"),
        default="imagenet",
        help="Use ImageNet transfer learning or random initialization.",
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=0,
        help="Optional deterministic subset per class and split for smoke tests.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_transforms(image_size: int) -> dict[str, transforms.Compose]:
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    return {
        "train": transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomAffine(
                    degrees=6,
                    translate=(0.04, 0.04),
                    scale=(0.94, 1.06),
                ),
                transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        "eval": transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    }


def limited_subset(dataset: datasets.ImageFolder, limit: int, seed: int) -> Subset:
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, (_, target) in enumerate(dataset.samples):
        by_class[target].append(index)
    rng = random.Random(seed)
    selected: list[int] = []
    for target in sorted(by_class):
        indices = by_class[target][:]
        rng.shuffle(indices)
        selected.extend(indices[:limit])
    return Subset(dataset, sorted(selected))


def build_loaders(args: argparse.Namespace, device: torch.device):
    tfm = build_transforms(args.image_size)
    image_folders = {
        "train": datasets.ImageFolder(args.data / "train", transform=tfm["train"]),
        "val": datasets.ImageFolder(args.data / "val", transform=tfm["eval"]),
        "test": datasets.ImageFolder(args.data / "test", transform=tfm["eval"]),
    }
    classes = image_folders["train"].classes
    mappings = {tuple(folder.classes) for folder in image_folders.values()}
    if len(mappings) != 1:
        raise ValueError("Class directories differ among train, val, and test.")

    datasets_for_loaders = {}
    for offset, (split, folder) in enumerate(image_folders.items()):
        datasets_for_loaders[split] = (
            limited_subset(folder, args.limit_per_class, args.seed + offset)
            if args.limit_per_class > 0
            else folder
        )

    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
            generator=generator if split == "train" else None,
        )
        for split, dataset in datasets_for_loaders.items()
    }
    return loaders, image_folders, classes


def build_model(num_classes: int, weights_mode: str) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.DEFAULT if weights_mode == "imagenet" else None
    model = models.mobilenet_v3_small(weights=weights)
    input_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(input_features, num_classes)
    return model


def freeze_backbone(model: nn.Module) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def enable_fine_tuning(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True


def make_optimizer(model: nn.Module, lr: float, weight_decay: float):
    return torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=lr,
        weight_decay=weight_decay,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(images)
                loss = criterion(logits, targets)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def source_paths(dataset: datasets.ImageFolder, indices: list[int]) -> list[str]:
    return [str(dataset.samples[index][0]) for index in indices]


@torch.inference_mode()
def evaluate_test(
    model: nn.Module,
    loader: DataLoader,
    source_dataset: datasets.ImageFolder,
    data_root: Path,
    classes: list[str],
    device: torch.device,
    output_dir: Path,
) -> dict:
    model.eval()
    targets_all: list[int] = []
    predictions_all: list[int] = []
    confidences_all: list[float] = []
    probabilities_all: list[list[float]] = []

    for images, targets in loader:
        logits = model(images.to(device, non_blocking=True))
        probabilities = torch.softmax(logits, dim=1).cpu()
        confidence, prediction = probabilities.max(dim=1)
        targets_all.extend(targets.tolist())
        predictions_all.extend(prediction.tolist())
        confidences_all.extend(confidence.tolist())
        probabilities_all.extend(probabilities.tolist())

    subset = loader.dataset
    if isinstance(subset, Subset):
        paths = source_paths(source_dataset, list(subset.indices))
    else:
        paths = [sample[0] for sample in source_dataset.samples]
    paths = [Path(path).resolve().relative_to(data_root).as_posix() for path in paths]

    matrix = confusion_matrix(targets_all, predictions_all, labels=range(len(classes)))
    report = classification_report(
        targets_all,
        predictions_all,
        labels=range(len(classes)),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    accuracy = float(np.mean(np.array(targets_all) == np.array(predictions_all)))

    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["actual/predicted", *classes])
        for class_name, row in zip(classes, matrix.tolist()):
            writer.writerow([class_name, *row])

    prediction_fields = [
        "image_path",
        "actual_index",
        "actual_class",
        "predicted_index",
        "predicted_class",
        "confidence",
        "correct",
        *[f"prob_{class_name}" for class_name in classes],
    ]
    with (output_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=prediction_fields)
        writer.writeheader()
        for path, actual, predicted, confidence, probabilities in zip(
            paths, targets_all, predictions_all, confidences_all, probabilities_all
        ):
            row = {
                "image_path": path,
                "actual_index": actual,
                "actual_class": classes[actual],
                "predicted_index": predicted,
                "predicted_class": classes[predicted],
                "confidence": confidence,
                "correct": actual == predicted,
            }
            row.update(
                {f"prob_{class_name}": value for class_name, value in zip(classes, probabilities)}
            )
            writer.writerow(row)

    return {
        "test_accuracy": accuracy,
        "test_samples": len(targets_all),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if not 0 <= args.head_epochs <= args.epochs:
        raise ValueError("--head-epochs must be between 0 and --epochs.")
    if not args.data.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {args.data}")

    args.data = args.data.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = choose_device(args.device)
    loaders, image_folders, classes = build_loaders(args, device)
    model = build_model(len(classes), args.weights).to(device)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    if args.head_epochs > 0:
        freeze_backbone(model)
        phase = "classifier"
        optimizer = make_optimizer(model, args.lr, args.weight_decay)
    else:
        enable_fine_tuning(model)
        phase = "fine_tuning"
        optimizer = make_optimizer(model, args.fine_tune_lr, args.weight_decay)

    config = {
        **vars(args),
        "data": str(args.data),
        "output": str(args.output),
        "model": "mobilenet_v3_small",
        "classes": classes,
        "class_to_idx": image_folders["train"].class_to_idx,
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "split_sizes": {split: len(loader.dataset) for split, loader in loaders.items()},
        "normalization_mean": IMAGENET_MEAN,
        "normalization_std": IMAGENET_STD,
    }
    save_json(args.output / "config.json", config)

    print(f"device={device} classes={classes} sizes={config['split_sizes']}")
    history: list[dict] = []
    best_val_accuracy = -1.0
    best_epoch = -1
    training_start = time.perf_counter()

    for epoch in range(args.epochs):
        if epoch == args.head_epochs and args.head_epochs > 0:
            enable_fine_tuning(model)
            phase = "fine_tuning"
            optimizer = make_optimizer(model, args.fine_tune_lr, args.weight_decay)

        start = time.perf_counter()
        train_loss, train_accuracy = run_epoch(
            model, loaders["train"], criterion, device, optimizer, scaler
        )
        val_loss, val_accuracy = run_epoch(
            model, loaders["val"], criterion, device, None, scaler
        )
        row = {
            "epoch": epoch + 1,
            "phase": phase,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "seconds": time.perf_counter() - start,
        }
        history.append(row)
        print(
            f"epoch={epoch + 1:02d}/{args.epochs:02d} phase={phase} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.3f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": "mobilenet_v3_small",
                    "classes": classes,
                    "class_to_idx": image_folders["train"].class_to_idx,
                    "image_size": args.image_size,
                    "normalization_mean": IMAGENET_MEAN,
                    "normalization_std": IMAGENET_STD,
                    "best_epoch": best_epoch,
                    "best_val_accuracy": best_val_accuracy,
                },
                args.output / "best_model.pth",
            )

        with (args.output / "history.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)

    checkpoint = torch.load(args.output / "best_model.pth", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_test(
        model,
        loaders["test"],
        image_folders["test"],
        args.data,
        classes,
        device,
        args.output,
    )
    metrics = {
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "training_seconds": time.perf_counter() - training_start,
        **test_metrics,
    }
    save_json(args.output / "metrics.json", metrics)
    print(
        f"complete best_epoch={best_epoch} best_val_acc={best_val_accuracy:.3f} "
        f"test_acc={metrics['test_accuracy']:.3f} output={args.output}"
    )


if __name__ == "__main__":
    main()
