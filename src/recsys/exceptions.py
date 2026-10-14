"""
src/recsys/exceptions.py
Hệ thống phân cấp ngoại lệ (Exception Hierarchy) cho toàn bộ Recommendation System.
"""
from __future__ import annotations
from typing import Any


class RecSysError(Exception):
    """
    Lớp ngoại lệ cơ sở (Base Exception) cho toàn bộ hệ thống gợi ý.
    Mọi ngoại lệ nghiệp vụ trong hệ thống đều kế thừa từ lớp này,
    giúp tầng Service/CLI dễ dàng bắt lỗi đa hình (Polymorphic Exception Handling).
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if not self.details:
            return f"[{self.__class__.__name__}] {self.message}"
        detail_str = ", ".join(f"{k}={v!r}" for k, v in self.details.items())
        return f"[{self.__class__.__name__}] {self.message} ({detail_str})"


class DataError(RecSysError):
    """Ném ra khi dữ liệu đầu vào bị rỗng, thiếu cột, hỏng định dạng hoặc không hợp lệ."""


class ConfigError(RecSysError):
    """Ném ra khi cấu hình hệ thống hoặc siêu tham số (hyperparameters) không hợp lệ."""


class UnknownUserError(RecSysError):
    """Ném ra khi truy vấn một user_id không tồn tại trong tập dữ liệu huấn luyện."""

    def __init__(self, user_id: Any) -> None:
        super().__init__(
            message=f"User ID '{user_id}' không tồn tại trong hệ thống.",
            details={"user_id": user_id},
        )
        self.user_id = user_id


class UnknownMovieError(RecSysError):
    """Ném ra khi truy vấn một movie_id không tồn tại trong tập dữ liệu huấn luyện."""

    def __init__(self, movie_id: Any) -> None:
        super().__init__(
            message=f"Movie ID '{movie_id}' không tồn tại trong hệ thống.",
            details={"movie_id": movie_id},
        )
        self.movie_id = movie_id


class InvalidParameterError(RecSysError):
    """Ném ra khi tham số truyền vào hàm (ví dụ: top_k <= 0) vi phạm ràng buộc."""

    def __init__(self, param_name: str, param_value: Any, reason: str) -> None:
        super().__init__(
            message=f"Tham số '{param_name}' không hợp lệ: {reason}",
            details={"param_name": param_name, "param_value": param_value},
        )
        self.param_name = param_name
        self.param_value = param_value


class ModelNotTrainedError(RecSysError):
    """Ném ra khi gọi dự đoán/gợi ý trên một mô hình chưa được huấn luyện (fit)."""

    def __init__(self, model_name: str = "Recommender") -> None:
        super().__init__(
            message=f"Mô hình '{model_name}' chưa được huấn luyện. Vui lòng gọi train()/fit() trước.",
            details={"model_name": model_name},
        )
        self.model_name = model_name