from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

IMG_EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp"}
TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTENSIONS


def _read_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_wnids(root: Path) -> List[str]:
    return _read_lines(root / "wnids.txt")


def _read_val_annotations(val_dir: Path) -> Dict[str, str]:
    ann_path = val_dir / "val_annotations.txt"
    if not ann_path.exists():
        raise FileNotFoundError(f"val_annotations.txt not found at {ann_path}")
    mapping: Dict[str, str] = {}
    for line in ann_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        mapping[parts[0]] = parts[1]
    return mapping


def _sorted_class_names(names: Iterable[str]) -> List[str]:
    return sorted({name for name in names if name})


def _is_valid_root(root: Path) -> bool:
    return (root / "train").is_dir() and (root / "val").is_dir() and (root / "wnids.txt").exists()


def _find_wnids_file(root: Path) -> Optional[Path]:
    candidates = [root / "wnids.txt", root / "tiny-imagenet-200" / "wnids.txt"]
    for path in candidates:
        if path.exists():
            return path
    for path in root.rglob("wnids.txt"):
        return path
    return None


def resolve_root(root: Path) -> Path:
    if _is_valid_root(root):
        return root
    nested = root / "tiny-imagenet-200"
    if _is_valid_root(nested):
        return nested
    return root


def _find_parquet_files(root: Path) -> List[Path]:
    if root.is_file() and root.suffix.lower() == ".parquet":
        return [root]
    return sorted([p for p in root.rglob("*.parquet") if p.is_file()])


def _match_split(name: str, token: str) -> bool:
    pattern = rf"(?:^|[/_\-]){re.escape(token)}(?:[/_\-.]|$)"
    return re.search(pattern, name) is not None


def _group_parquet_files(files: List[Path]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {"train": [], "val": [], "test": [], "other": []}
    for path in files:
        name = str(path).lower()
        if _match_split(name, "train"):
            groups["train"].append(str(path))
        elif _match_split(name, "validation") or _match_split(name, "valid") or _match_split(name, "val"):
            groups["val"].append(str(path))
        elif _match_split(name, "test"):
            groups["test"].append(str(path))
        else:
            groups["other"].append(str(path))
    return groups


def _parquet_files_for_split(root: Path, split: str) -> Dict[str, List[str]]:
    parquet_files = _find_parquet_files(root)
    if not parquet_files:
        return {}
    groups = _group_parquet_files(parquet_files)
    data_files: Dict[str, List[str]] = {}
    if groups["train"]:
        data_files["train"] = groups["train"]
    if groups["val"]:
        data_files["val"] = groups["val"]
    if groups["test"]:
        data_files["test"] = groups["test"]
    if not data_files and groups["other"]:
        data_files["train"] = groups["other"]
    return data_files


def _load_parquet_dataset(root: Path, split: str):
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Parquet Tiny ImageNet requires the 'datasets' package. Install it with: pip install datasets"
        ) from e
    data_files = _parquet_files_for_split(root, split=split)
    if not data_files:
        raise FileNotFoundError(f"No parquet files found under {root}")
    split_key = "val" if str(split) == "val" else str(split)
    if split_key not in data_files:
        raise FileNotFoundError(f"Parquet split '{split_key}' not found under {root}. Available: {list(data_files.keys())}")
    return load_dataset("parquet", data_files=data_files, split=split_key)


def _infer_image_column(record: Dict[str, Any]) -> Optional[str]:
    for key in ("image", "img", "images", "pixel_values", "pixels"):
        if key in record:
            return key
    for key, value in record.items():
        if _is_image_value(value):
            return key
    return None


def _infer_label_column(record: Dict[str, Any]) -> Optional[str]:
    for key in ("label", "labels", "target", "targets", "class", "wnid"):
        if key in record:
            return key
    return None


def _is_image_value(value: Any) -> bool:
    if value is None:
        return False
    try:
        from PIL import Image as PILImage
    except ModuleNotFoundError:
        PILImage = None
    if PILImage is not None and isinstance(value, PILImage.Image):
        return True
    if isinstance(value, dict) and ("bytes" in value or "path" in value):
        return True
    return False


def _to_pil_image(value: Any):
    from PIL import Image as PILImage

    if isinstance(value, PILImage.Image):
        return value
    if isinstance(value, dict):
        if value.get("bytes"):
            return PILImage.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return PILImage.open(value["path"]).convert("RGB")
    try:
        import numpy as np
    except ModuleNotFoundError:
        np = None
    if np is not None and isinstance(value, np.ndarray):
        if value.dtype != np.uint8:
            value = value.astype(np.uint8)
        return PILImage.fromarray(value)
    try:
        import torch
    except ModuleNotFoundError:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
        return _to_pil_image(arr)
    raise TypeError("Unsupported image value type for parquet dataset")


def ensure_tiny_imagenet(root: str | Path, download: bool = False) -> Path:
    root_path = Path(root).expanduser()
    resolved = resolve_root(root_path)
    if _is_valid_root(resolved):
        return resolved
    if not download:
        return resolved
    try:
        from torchvision.datasets.utils import download_and_extract_archive
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError("Tiny ImageNet download requires torchvision. Install it with: pip install torchvision") from e

    root_path.mkdir(parents=True, exist_ok=True)
    download_and_extract_archive(TINY_IMAGENET_URL, download_root=str(root_path), filename="tiny-imagenet-200.zip")
    resolved = resolve_root(root_path)
    if not _is_valid_root(resolved):
        raise RuntimeError(f"Tiny ImageNet download completed but dataset not found under {root_path}")
    return resolved


class TinyImageNet(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root).expanduser()
        self.split = str(split)
        self.transform = transform
        self.loader = default_loader
        self._mode = "images"
        self._parquet_ds = None
        self._parquet_image_key: Optional[str] = None
        self._parquet_label_key: Optional[str] = None

        resolved_root = resolve_root(self.root)
        if _is_valid_root(resolved_root):
            self.root = resolved_root
        else:
            self._mode = "parquet"
            self._parquet_ds = _load_parquet_dataset(self.root, split=self.split)
            sample = self._parquet_ds[0]
            self._parquet_image_key = _infer_image_column(sample)
            self._parquet_label_key = _infer_label_column(sample)
            if self._parquet_image_key is None:
                raise KeyError("Parquet dataset missing image column (expected keys like 'image' or 'img').")
            if self._parquet_label_key is None:
                raise KeyError("Parquet dataset missing label column (expected keys like 'label' or 'wnid').")

        if class_to_idx is None:
            wnids_path = _find_wnids_file(self.root)
            wnids = _read_lines(wnids_path) if wnids_path is not None else []
            if wnids:
                class_to_idx = {wnid: i for i, wnid in enumerate(wnids)}
            elif self._mode == "images":
                if self.split == "train":
                    train_dir = self.root / "train"
                    if not train_dir.exists():
                        raise FileNotFoundError(f"train directory not found at {train_dir}")
                    class_to_idx = {d.name: i for i, d in enumerate(sorted(train_dir.iterdir())) if d.is_dir()}
                else:
                    val_mapping = _read_val_annotations(self.root / "val")
                    classes = _sorted_class_names(val_mapping.values())
                    class_to_idx = {wnid: i for i, wnid in enumerate(classes)}
            else:
                class_to_idx = {}

        self.class_to_idx = class_to_idx or {}
        if self.class_to_idx:
            self.classes = [name for name, _ in sorted(self.class_to_idx.items(), key=lambda kv: kv[1])]
        else:
            self.classes = []

        if self._mode == "images":
            if self.split == "train":
                self.samples = self._make_train_samples()
            elif self.split == "val":
                self.samples = self._make_val_samples()
            else:
                raise ValueError(f"Unknown split: {self.split!r} (expected 'train' or 'val')")
            self.targets = [s[1] for s in self.samples]
        else:
            self.samples = []
            self.targets = []
            if self._parquet_ds is not None:
                if not self.classes:
                    try:
                        unique_labels = self._parquet_ds.unique(self._parquet_label_key)
                        self.classes = [str(v) for v in sorted(unique_labels)]
                    except Exception:
                        pass
                if self.classes and not self.class_to_idx:
                    self.class_to_idx = {name: i for i, name in enumerate(self.classes)}

    def _make_train_samples(self) -> List[Tuple[str, int]]:
        train_dir = self.root / "train"
        if not train_dir.exists():
            raise FileNotFoundError(f"train directory not found at {train_dir}")
        samples: List[Tuple[str, int]] = []
        for wnid, idx in self.class_to_idx.items():
            class_dir = train_dir / wnid
            if not class_dir.exists():
                raise FileNotFoundError(f"class directory not found: {class_dir}")
            img_dir = class_dir / "images" if (class_dir / "images").is_dir() else class_dir
            if not img_dir.exists():
                raise FileNotFoundError(f"images directory not found: {img_dir}")
            for path in img_dir.iterdir():
                if _is_image_file(path):
                    samples.append((str(path), int(idx)))
        if not samples:
            raise RuntimeError(f"No training images found under {train_dir}")
        return samples

    def _make_val_samples(self) -> List[Tuple[str, int]]:
        val_dir = self.root / "val"
        mapping = _read_val_annotations(val_dir)
        img_dir = val_dir / "images"
        if not img_dir.exists():
            raise FileNotFoundError(f"val images directory not found at {img_dir}")
        samples: List[Tuple[str, int]] = []
        for image_name, wnid in mapping.items():
            if wnid not in self.class_to_idx:
                raise KeyError(f"Unknown class {wnid!r} from val annotations")
            img_path = img_dir / image_name
            if not img_path.exists():
                raise FileNotFoundError(f"val image not found: {img_path}")
            if _is_image_file(img_path):
                samples.append((str(img_path), int(self.class_to_idx[wnid])))
        if not samples:
            raise RuntimeError(f"No validation images found under {val_dir}")
        return samples

    def __len__(self) -> int:
        if self._mode == "parquet" and self._parquet_ds is not None:
            return len(self._parquet_ds)
        return len(self.samples)

    def __getitem__(self, index: int):
        if self._mode == "parquet":
            if self._parquet_ds is None or self._parquet_image_key is None or self._parquet_label_key is None:
                raise RuntimeError("Parquet dataset not initialized")
            record = self._parquet_ds[index]
            image = _to_pil_image(record[self._parquet_image_key])
            target = record[self._parquet_label_key]
            if isinstance(target, str):
                if target not in self.class_to_idx:
                    raise KeyError(f"Unknown label {target!r} in parquet dataset")
                target = self.class_to_idx[target]
            target = int(target)
        else:
            path, target = self.samples[index]
            image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target
