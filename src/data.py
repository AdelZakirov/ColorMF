"""Lazy palette dataset adapter matching the legacy project's LAB convention."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from .lab import rgb_to_lab


class IndexedManifest:
    """A parsed manifest that keeps lightweight path entries in memory."""

    def __init__(self, filename: str):
        self.filename = str(filename)
        manifest_dir = Path(filename).resolve().parent
        self.manifest_dir = manifest_dir
        candidate_image_dir = manifest_dir / "256"
        self.image_dir = (
            candidate_image_dir if candidate_image_dir.is_dir() else manifest_dir
        )
        self.items: list[Tuple[str, str]] = []
        with open(self.filename, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if "\t" in line:
                    image_id, path = line.split("\t", 1)
                else:
                    path = line
                    image_id = Path(path).stem
                self.items.append((image_id, path))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Tuple[str, str]:
        image_id, path = self.items[index]
        path_object = Path(path)
        if not path_object.is_absolute():
            path = str(self.image_dir / path_object)
        return image_id, path

    def fallback(self, index: int) -> Tuple[str, str]:
        image_id, path = self.items[index]
        path_object = Path(path)
        if not path_object.is_absolute():
            path = str(self.manifest_dir / path_object)
        return image_id, path


class PaletteDataset(Dataset):
    """Decode one RGB image lazily and return normalized ``ab`` and ``L``."""

    def __init__(
        self,
        *,
        paths: Optional[Sequence[str]] = None,
        manifest: Optional[str] = None,
        size: Tuple[int, int] = (256, 256),
        train: bool = False,
        horizontal_flip: bool = False,
        resize_strategy: str = "center_crop",
    ):
        if (paths is None) == (manifest is None):
            raise ValueError("provide exactly one of paths or manifest")
        self.items = IndexedManifest(manifest) if manifest else list(paths)
        self.size = tuple(size)
        self.train = train
        self.horizontal_flip = horizontal_flip
        self.resize_strategy = resize_strategy

    def __len__(self) -> int:
        return len(self.items)

    def _item(self, index: int) -> Tuple[str, str]:
        if isinstance(self.items, IndexedManifest):
            return self.items[index]
        path = str(self.items[index])
        return Path(path).stem, path

    def __getitem__(self, index: int) -> dict:
        image_id, path = self._item(index)
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None and isinstance(self.items, IndexedManifest):
            image_id, path = self.items.fallback(index)
            image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"could not decode image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target_height, target_width = self.size
        if self.resize_strategy == "center_crop":
            # Official pMF uses ADM-style aspect-preserving resize + center crop.
            scale = max(target_height / image.shape[0], target_width / image.shape[1])
            resized = cv2.resize(
                image,
                (round(image.shape[1] * scale), round(image.shape[0] * scale)),
                interpolation=cv2.INTER_CUBIC,
            )
            top = (resized.shape[0] - target_height) // 2
            left = (resized.shape[1] - target_width) // 2
            image = resized[top : top + target_height, left : left + target_width]
        elif self.resize_strategy == "stretch":
            image = cv2.resize(
                image, (target_width, target_height), interpolation=cv2.INTER_CUBIC
            )
        else:
            raise ValueError("resize_strategy must be 'center_crop' or explicit legacy 'stretch'")
        if self.train and self.horizontal_flip and random.random() < 0.5:
            image = image[:, ::-1].copy()
        L, ab = rgb_to_lab(image)
        return {"ab": ab, "L": L, "image_id": image_id}


def _paths_from_root(root: str) -> list:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG"}
    return [
        str(path)
        for path in Path(root).rglob("*")
        if path.is_file() and path.suffix in extensions
    ]


class PaletteDataModule(pl.LightningDataModule):
    def __init__(
        self,
        *,
        train_manifest: Optional[str] = None,
        val_manifest: Optional[str] = None,
        train_root: Optional[str] = None,
        val_root: Optional[str] = None,
        resolution: Tuple[int, int] = (256, 256),
        batch_size: int = 8,
        val_batch_size: int = 4,
        num_workers: int = 4,
        horizontal_flip: bool = False,
        resize_strategy: str = "center_crop",
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        if self.hparams.train_manifest:
            train_kwargs = {"manifest": self.hparams.train_manifest}
        elif self.hparams.train_root:
            train_kwargs = {"paths": _paths_from_root(self.hparams.train_root)}
        else:
            raise ValueError("a train manifest or train root is required")
        if self.hparams.val_manifest:
            val_kwargs = {"manifest": self.hparams.val_manifest}
        elif self.hparams.val_root:
            val_kwargs = {"paths": _paths_from_root(self.hparams.val_root)}
        else:
            raise ValueError("a validation manifest or validation root is required")
        common = {
            "size": tuple(self.hparams.resolution),
            "horizontal_flip": self.hparams.horizontal_flip,
            "resize_strategy": self.hparams.resize_strategy,
        }
        self.train_dataset = PaletteDataset(**train_kwargs, train=True, **common)
        self.val_dataset = PaletteDataset(**val_kwargs, train=False, **common)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.val_batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
        )
