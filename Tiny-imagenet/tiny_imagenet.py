from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

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


def resolve_root(root: Path) -> Path:
    if _is_valid_root(root):
        return root
    nested = root / "tiny-imagenet-200"
    if _is_valid_root(nested):
        return nested
    return root


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

        if class_to_idx is None:
            wnids = _read_wnids(self.root)
            if wnids:
                class_to_idx = {wnid: i for i, wnid in enumerate(wnids)}
            elif self.split == "train":
                train_dir = self.root / "train"
                if not train_dir.exists():
                    raise FileNotFoundError(f"train directory not found at {train_dir}")
                class_to_idx = {d.name: i for i, d in enumerate(sorted(train_dir.iterdir())) if d.is_dir()}
            else:
                val_mapping = _read_val_annotations(self.root / "val")
                classes = _sorted_class_names(val_mapping.values())
                class_to_idx = {wnid: i for i, wnid in enumerate(classes)}
        self.class_to_idx = class_to_idx
        self.classes = [name for name, _ in sorted(self.class_to_idx.items(), key=lambda kv: kv[1])]

        if self.split == "train":
            self.samples = self._make_train_samples()
        elif self.split == "val":
            self.samples = self._make_val_samples()
        else:
            raise ValueError(f"Unknown split: {self.split!r} (expected 'train' or 'val')")
        self.targets = [s[1] for s in self.samples]

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
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target
