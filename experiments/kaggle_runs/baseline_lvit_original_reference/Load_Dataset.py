# -*- coding: utf-8 -*-
"""Dataset and transform helpers shared by training and evaluation."""
import os
import random
from typing import Any, Callable

import cv2
import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset

from text_encoder import CachedTextEncoder, load_report_features, save_report_features


def random_rot_flip(image: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply a random 90-degree rotation followed by a random flip."""
    k = np.random.randint(0, 4)
    image = np.rot90(image, k, axes=(0, 1)).copy()
    label = np.rot90(label, k, axes=(0, 1)).copy()
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply a small random rotation while preserving the canvas size."""
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, axes=(0, 1), order=3, reshape=False, mode="nearest")
    label = ndimage.rotate(label, angle, axes=(0, 1), order=0, reshape=False, mode="nearest")
    return image, label


def _ensure_channel_last(array: np.ndarray) -> np.ndarray:
    """Expand grayscale arrays so later code can assume HWC layout."""
    if array.ndim == 2:
        return np.expand_dims(array, axis=-1)
    return array


def _resize_image(image: np.ndarray, output_size: list[int] | tuple[int, int]) -> np.ndarray:
    """Resize an image with linear interpolation."""
    height, width = int(output_size[0]), int(output_size[1])
    if image.shape[0] == height and image.shape[1] == width:
        return image
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return _ensure_channel_last(resized)


def _resize_mask(mask: np.ndarray, output_size: list[int] | tuple[int, int]) -> np.ndarray:
    """Resize a mask with nearest-neighbor interpolation."""
    height, width = int(output_size[0]), int(output_size[1])
    if mask.shape[0] == height and mask.shape[1] == width:
        return mask
    resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return _ensure_channel_last(resized)


def to_long_tensor(mask: np.ndarray) -> torch.Tensor:
    """Convert a mask array into the integer tensor type expected by losses."""
    array = np.asarray(mask)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    return torch.from_numpy(np.ascontiguousarray(array.astype(np.int64)))


def _to_image_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert an HWC image array into a normalized CHW tensor."""
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        array = np.expand_dims(array, axis=-1)
    if array.max() > 1.0:
        array = array / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
    return tensor.float()


def _prepare_sample_tensor_dict(sample: dict[str, Any], output_size: list[int] | tuple[int, int]) -> dict[str, Any]:
    """Resize and convert one sample dict into tensors ready for the model."""
    image = sample.get("image")
    label = sample["label"]
    text = sample["text"]
    attributes = sample.get("attributes")

    if image is not None:
        image = _ensure_channel_last(np.asarray(image))
        image = _resize_image(image, output_size)
    label = _ensure_channel_last(np.asarray(label))
    label = _resize_mask(label, output_size)

    prepared: dict[str, Any] = {
        "label": to_long_tensor((label > 0).astype(np.uint8)),
        "text": torch.as_tensor(text, dtype=torch.float32),
    }
    if image is not None:
        prepared["image"] = _to_image_tensor(image)
    if attributes is not None:
        prepared["attributes"] = torch.as_tensor(attributes, dtype=torch.float32)
    return prepared


class RandomGenerator(object):
    """Implementation class used by the segmentation training pipeline."""
    def __init__(self, output_size):
        """Execute a documented step of the segmentation training pipeline."""
        self.output_size = output_size

    def __call__(self, sample):
        """Apply train-time augmentation and tensor conversion."""
        image = sample.get("image")
        label = sample["label"]
        if image is not None:
            image = _ensure_channel_last(np.asarray(image))
        label = _ensure_channel_last(np.asarray(label))

        if image is not None and random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif image is not None and random.random() > 0.5:
            image, label = random_rotate(image, label)

        transformed = dict(sample)
        if image is not None:
            transformed["image"] = image
        transformed["label"] = label
        return _prepare_sample_tensor_dict(transformed, self.output_size)


class ValGenerator(object):
    """Implementation class used by the segmentation training pipeline."""
    def __init__(self, output_size):
        """Execute a documented step of the segmentation training pipeline."""
        self.output_size = output_size

    def __call__(self, sample):
        """Apply deterministic preprocessing for validation or test data."""
        return _prepare_sample_tensor_dict(sample, self.output_size)


class _BaseTextDataset(Dataset):
    """Implementation class used by the segmentation training pipeline."""
    def __init__(
        self,
        task_name: str,
        row_text: dict[str, str],
        cache_dir: str | None = None,
        text_model_name: str | None = None,
        local_files_only: bool = False,
        cache_metadata: dict[str, Any] | None = None,
        max_text_units: int = 10,
    ) -> None:
        """Store shared text handling and optional caching logic."""
        self.task_name = task_name
        self.rowtext = row_text
        self.cache_dir = cache_dir
        self.text_model_name = text_model_name
        self.local_files_only = local_files_only
        self.cache_metadata = cache_metadata
        self.max_text_units = max_text_units
        self.text_encoder: CachedTextEncoder | None = None

        if cache_dir is None:
            self.text_encoder = self._build_text_encoder()

    def _build_text_encoder(self) -> CachedTextEncoder:
        """Create the text encoder only when the dataset needs it."""
        encoder = CachedTextEncoder(
            model_name=self.text_model_name or "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
            local_files_only=self.local_files_only,
        )
        if self.cache_metadata is None:
            self.cache_metadata = encoder.cache_metadata(self.max_text_units)
        return encoder

    def _get_text_encoder(self) -> CachedTextEncoder:
        """Reuse the same encoder instance across multiple samples."""
        if self.text_encoder is None:
            self.text_encoder = self._build_text_encoder()
        return self.text_encoder

    def _load_cached_features(self, *sample_ids: str):
        """Try to load cached text features for one of the candidate sample ids."""
        if not self.cache_dir:
            return None
        for sample_id in sample_ids:
            cached = load_report_features(
                self.cache_dir,
                sample_id,
                expected_metadata=self.cache_metadata,
            )
            if cached is not None:
                return cached
        return None

    def _get_text_features(self, report: str, *sample_ids: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return encoded report tokens and structured attributes for one sample."""
        cached = self._load_cached_features(*sample_ids)
        if cached is not None:
            text, attributes, _ = cached
            return text, attributes

        encoder = self._get_text_encoder()
        text, attributes = encoder.encode_report(report, max_lines=self.max_text_units)
        if self.cache_dir:
            metadata = self.cache_metadata or encoder.cache_metadata(self.max_text_units)
            for sample_id in sample_ids:
                save_report_features(self.cache_dir, sample_id, text, attributes, metadata=metadata)
        return text, attributes


class LV2D(_BaseTextDataset):
    """Implementation class used by the segmentation training pipeline."""
    def __init__(
        self,
        dataset_path: str,
        task_name: str,
        row_text: dict[str, str],
        joint_transform: Callable = None,
        one_hot_mask: int = False,
        image_size: int = 224,
        cache_dir: str | None = None,
        text_model_name: str | None = None,
        local_files_only: bool = False,
        cache_metadata: dict[str, Any] | None = None,
        max_text_units: int = 10,
    ) -> None:
        """Load mask-only samples used by the semi-supervised branch."""
        super().__init__(
            task_name=task_name,
            row_text=row_text,
            cache_dir=cache_dir,
            text_model_name=text_model_name,
            local_files_only=local_files_only,
            cache_metadata=cache_metadata,
            max_text_units=max_text_units,
        )
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.output_path = os.path.join(dataset_path)
        self.mask_list = sorted(os.listdir(self.output_path))
        self.one_hot_mask = one_hot_mask
        self.joint_transform = joint_transform or ValGenerator(output_size=[image_size, image_size])

    def __len__(self):
        """Execute a documented step of the segmentation training pipeline."""
        return len(self.mask_list)

    def _resolve_report(self, mask_filename: str) -> str:
        """Match a mask file name with its text annotation."""
        candidates = [mask_filename, os.path.splitext(mask_filename)[0]]
        for candidate in candidates:
            if candidate in self.rowtext:
                return self.rowtext[candidate]
        raise KeyError(f"Could not resolve text annotation for mask={mask_filename}")

    def __getitem__(self, idx):
        """Return one mask sample together with its encoded text features."""
        mask_filename = self.mask_list[idx]
        mask = cv2.imread(os.path.join(self.output_path, mask_filename), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Unable to read mask {mask_filename} from {self.output_path}")
        mask = (mask > 0).astype(np.uint8)

        report = self._resolve_report(mask_filename)
        text, attributes = self._get_text_features(report, mask_filename, os.path.splitext(mask_filename)[0])
        sample = {"label": mask, "text": text, "attributes": attributes}

        if self.joint_transform:
            sample = self.joint_transform(sample)

        if self.one_hot_mask:
            assert self.one_hot_mask > 0, "one_hot_mask must be nonnegative"
            label = sample["label"]
            sample["label"] = torch.zeros(
                (self.one_hot_mask, label.shape[0], label.shape[1]),
                dtype=torch.float32,
            ).scatter_(0, label.unsqueeze(0), 1.0)

        return sample, mask_filename


class ImageToImage2D(_BaseTextDataset):
    """Implementation class used by the segmentation training pipeline."""
    def __init__(
        self,
        dataset_path: str,
        task_name: str,
        row_text: dict[str, str],
        joint_transform: Callable = None,
        one_hot_mask: int = False,
        image_size: int = 224,
        cache_dir: str | None = None,
        text_model_name: str | None = None,
        local_files_only: bool = False,
        cache_metadata: dict[str, Any] | None = None,
        max_text_units: int = 10,
    ) -> None:
        """Execute a documented step of the segmentation training pipeline."""
        super().__init__(
            task_name=task_name,
            row_text=row_text,
            cache_dir=cache_dir,
            text_model_name=text_model_name,
            local_files_only=local_files_only,
            cache_metadata=cache_metadata,
            max_text_units=max_text_units,
        )
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.input_path = os.path.join(dataset_path, "img")
        self.output_path = os.path.join(dataset_path, "labelcol")
        self.images_list = sorted(os.listdir(self.input_path))
        self.mask_list = sorted(os.listdir(self.output_path))
        self.one_hot_mask = one_hot_mask
        self.joint_transform = joint_transform or ValGenerator(output_size=[image_size, image_size])

    def __len__(self):
        """Execute a documented step of the segmentation training pipeline."""
        return len(self.images_list)

    def _resolve_mask_filename(self, image_filename: str) -> str:
        """Execute a documented step of the segmentation training pipeline."""
        image_stem = os.path.splitext(image_filename)[0]
        candidates = [
            f"{image_stem}.png",
            f"{image_stem}.jpg",
            f"{image_stem}.jpeg",
            image_filename,
            image_filename.replace("mask_", ""),
            image_filename.replace(".tif", ".png"),
            image_filename.replace(".tiff", ".png"),
        ]
        for candidate in candidates:
            if candidate in self.mask_list:
                return candidate
        raise FileNotFoundError(f"Could not resolve mask for image {image_filename} in {self.output_path}")

    def _resolve_report(self, image_filename: str, mask_filename: str) -> str:
        """Execute a documented step of the segmentation training pipeline."""
        candidates = [
            mask_filename,
            image_filename,
            os.path.splitext(mask_filename)[0],
            os.path.splitext(image_filename)[0],
        ]
        for candidate in candidates:
            if candidate in self.rowtext:
                return self.rowtext[candidate]
        raise KeyError(f"Could not resolve text annotation for image={image_filename}, mask={mask_filename}")

    def __getitem__(self, idx):
        """Execute a documented step of the segmentation training pipeline."""
        image_filename = self.images_list[idx]
        mask_filename = self._resolve_mask_filename(image_filename)

        image = cv2.imread(os.path.join(self.input_path, image_filename), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to read image {image_filename} from {self.input_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(os.path.join(self.output_path, mask_filename), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Unable to read mask {mask_filename} from {self.output_path}")
        mask = (mask > 0).astype(np.uint8)

        report = self._resolve_report(image_filename, mask_filename)
        text, attributes = self._get_text_features(
            report,
            mask_filename,
            image_filename,
            os.path.splitext(mask_filename)[0],
            os.path.splitext(image_filename)[0],
        )
        sample = {"image": image, "label": mask, "text": text, "attributes": attributes}

        if self.joint_transform:
            sample = self.joint_transform(sample)

        if self.one_hot_mask:
            assert self.one_hot_mask > 0, "one_hot_mask must be nonnegative"
            label = sample["label"]
            sample["label"] = torch.zeros(
                (self.one_hot_mask, label.shape[0], label.shape[1]),
                dtype=torch.float32,
            ).scatter_(0, label.unsqueeze(0), 1.0)

        return sample, image_filename
