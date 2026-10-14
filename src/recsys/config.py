"""
src/recsys/config.py
Quản lý và kiểm tra tính hợp lệ của cấu hình hệ thống (Configuration & Validation).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from recsys.exceptions import ConfigError


SUPPORTED_MODELS = {"popularity", "matrix_factorization", "bpr"}


@dataclass(frozen=True)
class DataConfig:
    """Cấu hình cho tầng dữ liệu (Data Layer)."""
    data_path: str = "data/ml-100k/u.data"
    movies_path: str = "data/ml-100k/u.item"
    test_ratio: float = 0.2
    random_seed: int = 42
    min_rating: float = 1.0
    max_rating: float = 5.0

    def __post_init__(self) -> None:
        if not self.data_path or not str(self.data_path).strip():
            raise ConfigError("Đường dẫn dữ liệu (data_path) không được để trống.")
        if not (0.0 < self.test_ratio < 1.0):
            raise ConfigError(
                "Tỷ lệ tập test (test_ratio) phải nằm trong khoảng (0.0, 1.0).",
                details={"test_ratio": self.test_ratio},
            )
        if self.min_rating >= self.max_rating:
            raise ConfigError(
                "min_rating phải nhỏ hơn nghiêm ngặt so với max_rating.",
                details={"min_rating": self.min_rating, "max_rating": self.max_rating},
            )


@dataclass(frozen=True)
class ModelConfig:
    """Cấu hình cho thuật toán gợi ý (Recommendation Model)."""
    model_type: str = "matrix_factorization"
    n_factors: int = 20
    learning_rate: float = 0.01
    reg: float = 0.02
    n_epochs: int = 15
    random_seed: int = 42

    def __post_init__(self) -> None:
        normalized_type = self.model_type.strip().lower()
        if normalized_type not in SUPPORTED_MODELS:
            raise ConfigError(
                f"Loại mô hình '{self.model_type}' không được hỗ trợ. "
                f"Các mô hình hợp lệ: {sorted(SUPPORTED_MODELS)}",
                details={"model_type": self.model_type},
            )
        # Chuẩn hóa tên model về chữ thường
        object.__setattr__(self, "model_type", normalized_type)

        if self.n_factors <= 0:
            raise ConfigError(
                "Số chiều ẩn (n_factors) phải là số nguyên dương.",
                details={"n_factors": self.n_factors},
            )
        if self.learning_rate <= 0.0:
            raise ConfigError(
                "Tốc độ học (learning_rate) phải lớn hơn 0.",
                details={"learning_rate": self.learning_rate},
            )
        if self.reg < 0.0:
            raise ConfigError(
                "Hệ số điều chuẩn (reg) không được âm.",
                details={"reg": self.reg},
            )
        if self.n_epochs <= 0:
            raise ConfigError(
                "Số vòng lặp huấn luyện (n_epochs) phải là số nguyên dương.",
                details={"n_epochs": self.n_epochs},
            )


@dataclass(frozen=True)
class EvalConfig:
    """Cấu hình cho tầng đánh giá (Evaluation Layer)."""
    top_k: int = 10
    metrics: tuple[str, ...] = ("ndcg", "recall", "precision", "auc")
    rating_threshold: float = 4.0

    def __post_init__(self) -> None:
        if not isinstance(self.top_k, int) or self.top_k <= 0:
            raise ConfigError(
                "Tham số đánh giá top_k phải là số nguyên dương.",
                details={"top_k": self.top_k},
            )
        if not self.metrics:
            raise ConfigError("Danh sách độ đo (metrics) không được để trống.")


@dataclass(frozen=True)
class SystemConfig:
    """
    Cấu hình tổng thể của hệ thống (Áp dụng Composition: kết hợp các cấu hình thành phần).
    """
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    default_top_k: int = 10

    def __post_init__(self) -> None:
        if not isinstance(self.default_top_k, int) or self.default_top_k <= 0:
            raise ConfigError(
                "default_top_k phải là số nguyên dương.",
                details={"default_top_k": self.default_top_k},
            )

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> SystemConfig:
        """Khởi tạo SystemConfig an toàn từ dictionary (ví dụ đọc từ JSON/YAML)."""
        if not isinstance(config_dict, dict):
            raise ConfigError("Dữ liệu cấu hình phải là một dictionary.")
        try:
            data_cfg = DataConfig(**config_dict.get("data", {}))
            model_cfg = ModelConfig(**config_dict.get("model", {}))
            eval_raw = dict(config_dict.get("evaluation", {}))
            if "metrics" in eval_raw and isinstance(eval_raw["metrics"], list):
                eval_raw["metrics"] = tuple(eval_raw["metrics"])
            eval_cfg = EvalConfig(**eval_raw)
            default_k = config_dict.get("default_top_k", 10)
            return cls(
                data=data_cfg,
                model=model_cfg,
                evaluation=eval_cfg,
                default_top_k=default_k,
            )
        except TypeError as exc:
            raise ConfigError(f"Tham số cấu hình không khớp định dạng: {exc}") from exc