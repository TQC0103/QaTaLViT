import re
import threading
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import torch


FEATURE_CACHE_FORMAT_VERSION = 3
REPORT_UNIT_SPLIT_VERSION = 2

ATTRIBUTE_GROUPS = {
    "laterality": ["unknown", "left", "right", "bilateral", "diffuse"],
    "vertical": ["unknown", "upper", "middle", "lower", "basal"],
    "count": ["unknown", "single", "multiple", "diffuse"],
    "extent": ["unknown", "focal", "multifocal", "diffuse"],
    "severity": ["unknown", "mild", "moderate", "severe"],
}


def attribute_vector_size() -> int:
    return sum(len(options) for options in ATTRIBUTE_GROUPS.values())


def build_cache_metadata(
    model_name: str,
    max_units: int,
    parser_name: str = "structured-report-parser-v1",
) -> dict[str, Any]:
    return {
        "format_version": FEATURE_CACHE_FORMAT_VERSION,
        "model_name": model_name,
        "max_units": int(max_units),
        "parser_name": parser_name,
        "report_unit_split_version": REPORT_UNIT_SPLIT_VERSION,
    }


def split_report_into_units(report: str) -> List[str]:
    text = (report or "").strip()
    if not text:
        return ["[NO_TEXT]"]

    raw_units = re.split(r"[\n.;]+|,\s+(?=[A-Za-z])", text)
    units = [unit.strip() for unit in raw_units if unit and unit.strip()]
    if not units:
        return ["[NO_TEXT]"]
    return units


def feature_cache_path(cache_dir: str | Path, sample_id: str) -> Path:
    stem = Path(sample_id).stem
    return Path(cache_dir) / f"{stem}.pt"


def save_report_features(
    cache_dir: str | Path,
    sample_id: str,
    text: torch.Tensor,
    attributes: torch.Tensor,
    metadata: dict[str, Any] | None = None,
) -> Path:
    cache_path = feature_cache_path(cache_dir, sample_id)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "text": text.cpu(),
        "attributes": attributes.cpu(),
        "metadata": metadata or {"format_version": FEATURE_CACHE_FORMAT_VERSION},
    }
    torch.save(payload, cache_path)
    return cache_path


def _metadata_matches(
    cached_metadata: dict[str, Any] | None,
    expected_metadata: dict[str, Any] | None,
) -> bool:
    if expected_metadata is None:
        return True
    if cached_metadata is None:
        return False
    for key, expected_value in expected_metadata.items():
        if cached_metadata.get(key) != expected_value:
            return False
    return True


def load_report_features(
    cache_dir: str | Path,
    sample_id: str,
    expected_metadata: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]] | None:
    cache_path = feature_cache_path(cache_dir, sample_id)
    if not cache_path.exists():
        return None
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata", {})
    if not _metadata_matches(metadata, expected_metadata):
        return None
    return payload["text"], payload["attributes"], metadata


class StructuredReportParser:
    """Heuristic parser that extracts coarse structured report attributes."""

    def __init__(self) -> None:
        self.parser_name = "structured-report-parser-v1"
        self.patterns = {
            "laterality": {
                "bilateral": re.compile(r"\bbilateral\b|\bboth lungs?\b", re.IGNORECASE),
                "left": re.compile(r"\bleft\b", re.IGNORECASE),
                "right": re.compile(r"\bright\b", re.IGNORECASE),
                "diffuse": re.compile(r"\bdiffuse\b|\bscattered\b", re.IGNORECASE),
            },
            "vertical": {
                "upper": re.compile(r"\bupper\b|\bapical\b", re.IGNORECASE),
                "middle": re.compile(r"\bmiddle\b|\bmid\b", re.IGNORECASE),
                "lower": re.compile(r"\blower\b|\bbasal\b|\bbase\b", re.IGNORECASE),
                "basal": re.compile(r"\bbasal\b|\bbase\b", re.IGNORECASE),
            },
            "count": {
                "single": re.compile(r"\bsingle\b|\bone lesion\b", re.IGNORECASE),
                "multiple": re.compile(r"\bmultiple\b|\bseveral\b|\bmultifocal\b", re.IGNORECASE),
                "diffuse": re.compile(r"\bdiffuse\b|\bwidespread\b", re.IGNORECASE),
            },
            "extent": {
                "focal": re.compile(r"\bfocal\b|\blocalized\b", re.IGNORECASE),
                "multifocal": re.compile(r"\bmultifocal\b|\bpatchy\b", re.IGNORECASE),
                "diffuse": re.compile(r"\bdiffuse\b|\bextensive\b|\bconfluent\b", re.IGNORECASE),
            },
            "severity": {
                "mild": re.compile(r"\bmild\b", re.IGNORECASE),
                "moderate": re.compile(r"\bmoderate\b", re.IGNORECASE),
                "severe": re.compile(r"\bsevere\b|\bcritical\b|\bextensive\b", re.IGNORECASE),
            },
        }

    def parse(self, report: str) -> dict[str, str]:
        text = report or ""
        parsed: dict[str, str] = {}
        for group, options in ATTRIBUTE_GROUPS.items():
            parsed[group] = "unknown"
            for option in options:
                if option == "unknown":
                    continue
                pattern = self.patterns[group].get(option)
                if pattern is not None and pattern.search(text):
                    parsed[group] = option
                    break
        return parsed

    def vectorize(self, report: str) -> torch.Tensor:
        parsed = self.parse(report)
        vector = torch.zeros(attribute_vector_size(), dtype=torch.float32)
        offset = 0
        for group, options in ATTRIBUTE_GROUPS.items():
            value = parsed[group]
            index = options.index(value)
            vector[offset + index] = 1.0
            offset += len(options)
        return vector


class CachedDomainTextEncoder:
    """Cached domain-aware text encoder for medical report text."""

    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        device: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.local_files_only = local_files_only
        self._tokenizer = None
        self._model = None
        self._cache: dict[str, torch.Tensor] = {}
        self._lock = threading.Lock()
        self.report_parser = StructuredReportParser()

    def cache_metadata(self, max_units: int) -> dict[str, Any]:
        return build_cache_metadata(
            model_name=self.model_name,
            max_units=max_units,
            parser_name=self.report_parser.parser_name,
        )

    def _ensure_model(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        with self._lock:
            if self._model is None or self._tokenizer is None:
                from transformers import AutoModel, AutoTokenizer

                try:
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name,
                        local_files_only=self.local_files_only,
                        use_fast=False,
                    )
                    self._model = AutoModel.from_pretrained(
                        self.model_name,
                        local_files_only=self.local_files_only,
                        use_safetensors=True,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "Failed to initialize the Hugging Face text encoder. "
                        "Make sure the requested model is available, the environment has "
                        "compatible tokenizer dependencies, and offline mode has a cached copy. "
                        "Recommended setup: transformers<5, protobuf, and sentencepiece installed."
                    ) from exc
                self._model.eval()
                self._model.to(self.device)

    def encode_lines(self, lines: Iterable[str], max_lines: int | None = None) -> torch.Tensor:
        cleaned_lines: List[str] = [line.strip() for line in lines if line and line.strip()]
        if max_lines is not None:
            cleaned_lines = cleaned_lines[:max_lines]
        if not cleaned_lines:
            cleaned_lines = ["[NO_TEXT]"]
        outputs = [self._encode_single(line) for line in cleaned_lines]
        if max_lines is not None and len(outputs) < max_lines:
            pad_token = outputs[0].new_zeros(outputs[0].shape)
            outputs.extend([pad_token.clone() for _ in range(max_lines - len(outputs))])
        return torch.stack(outputs, dim=0)

    def encode_report(self, report: str, max_lines: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        lines = split_report_into_units(report)
        tokens = self.encode_lines(lines, max_lines=max_lines)
        attrs = self.report_parser.vectorize(report)
        return tokens, attrs

    def batch_attribute_vectors(self, reports: Sequence[str]) -> torch.Tensor:
        return torch.stack([self.report_parser.vectorize(report) for report in reports], dim=0)

    def _encode_single(self, text: str) -> torch.Tensor:
        cached = self._cache.get(text)
        if cached is not None:
            return cached.clone()

        self._ensure_model()
        assert self._tokenizer is not None
        assert self._model is not None
        with torch.inference_mode():
            encoded = self._tokenizer(
                text,
                truncation=True,
                max_length=64,
                padding="max_length",
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            outputs = self._model(**encoded)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).detach().cpu()

        self._cache[text] = embedding
        return embedding.clone()


# Backward-compatible alias for earlier local changes.
CachedTextEncoder = CachedDomainTextEncoder
