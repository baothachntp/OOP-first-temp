"""
src/recsys/service/recommendation_service.py
Tầng dịch vụ gợi ý (Recommendation Service Layer) cung cấp giao diện đơn giản,
trừu tượng hóa toàn bộ chi tiết cài đặt của Data Layer và Model Layer.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

from recsys.config import ModelConfig
from recsys.exceptions import (
    DataError,
    InvalidParameterError,
    ModelNotTrainedError,
    UnknownMovieError,
    UnknownUserError,
)
from recsys.service.model_factory import ModelFactory, RecommenderProtocol


@runtime_checkable
class MovieEntityProtocol(Protocol):
    """Giao diện đại diện cho thực thể Movie từ tầng Data của Vân Anh."""
    movie_id: Any
    title: str


@runtime_checkable
class DatasetProtocol(Protocol):
    """Giao diện đại diện cho lớp Dataset từ tầng Data của Vân Anh."""
    @property
    def num_users(self) -> int: ...

    @property
    def num_movies(self) -> int: ...

    @property
    def num_ratings(self) -> int: ...

    def has_user(self, user_id: Any) -> bool: ...

    def has_movie(self, movie_id: Any) -> bool: ...

    def get_user_index(self, user_id: Any) -> int: ...

    def get_movie_index(self, movie_id: Any) -> int: ...

    def get_movie_by_index(self, movie_idx: int) -> MovieEntityProtocol: ...

    def get_user_seen_movie_indices(self, user_idx: int) -> set[int]: ...


@dataclass(frozen=True)
class Recommendation:
    """
    Đối tượng giá trị (Value Object) chứa kết quả gợi ý cho người dùng.
    Sử dụng frozen=True để đảm bảo tính đóng gói và bất biến (Encapsulation).
    """
    movie_id: Any
    title: str
    score: float
    rank: int


class RecommendationService:
    """
    Lớp Facade cung cấp giao diện cấp cao để huấn luyện và sinh gợi ý Top-K.
    Áp dụng:
    - Composition: Chứa Dataset và Recommender.
    - Encapsulation: Ẩn toàn bộ ánh xạ ID <-> Index và ma trận điểm số bên dưới.
    - Abstraction: Cung cấp hàm `recommend()` trực quan cho tầng giao diện (CLI).
    """

    def __init__(
        self,
        dataset: DatasetProtocol,
        model: RecommenderProtocol,
        default_top_k: int = 10,
    ) -> None:
        if dataset is None:
            raise DataError("Dataset truyền vào RecommendationService không được là None.")
        if model is None:
            raise InvalidParameterError("model", None, "Mô hình Recommender không được là None.")
        if not isinstance(default_top_k, int) or isinstance(default_top_k, bool) or default_top_k <= 0:
            raise InvalidParameterError(
                "default_top_k", default_top_k, "Phải là một số nguyên dương (> 0)."
            )

        self._validate_dataset_not_empty(dataset)

        self._dataset: DatasetProtocol = dataset
        self._model: RecommenderProtocol = model
        self._default_top_k: int = default_top_k
        self._is_trained: bool = bool(getattr(model, "is_trained", False))

    @classmethod
    def from_config(
        cls,
        dataset: DatasetProtocol,
        model_config: ModelConfig,
        default_top_k: int = 10,
    ) -> RecommendationService:
        """
        Factory method tiện lợi: Tự động dùng ModelFactory để tạo model từ cấu hình
        và khởi tạo RecommendationService.
        """
        model = ModelFactory.create(model_config)
        return cls(dataset=dataset, model=model, default_top_k=default_top_k)

    @property
    def is_trained(self) -> bool:
        """Kiểm tra xem mô hình bên dưới đã được huấn luyện hay chưa."""
        return self._is_trained or bool(getattr(self._model, "is_trained", False))

    @property
    def model_name(self) -> str:
        """Trả về tên lớp của mô hình đang sử dụng."""
        return self._model.__class__.__name__

    def train(self) -> None:
        """Huấn luyện mô hình gợi ý trên tập dữ liệu hiện tại."""
        self._validate_dataset_not_empty(self._dataset)
        self._model.fit(self._dataset)
        self._is_trained = True

    def recommend(
        self,
        user_id: Any,
        top_k: int | None = None,
        exclude_seen: bool = True,
    ) -> list[Recommendation]:
        """
        Sinh danh sách Top-K bộ phim gợi ý cho một người dùng cụ thể.

        Args:
            user_id: Định danh bên ngoài của người dùng (external user ID).
            top_k: Số lượng phim muốn gợi ý. Nếu None, dùng default_top_k.
            exclude_seen: Nếu True, loại bỏ các phim người dùng đã đánh giá trong tập train.

        Returns:
            Danh sách các đối tượng `Recommendation` đã sắp xếp giảm dần theo điểm.
        """
        self._ensure_trained()

        k = self._default_top_k if top_k is None else top_k
        self._validate_top_k(k)
        self._validate_user_exists(user_id)

        user_idx = self._dataset.get_user_index(user_id)
        raw_scores = self._model.predict_for_user(user_idx)

        if len(raw_scores) != self._dataset.num_movies:
            raise DataError(
                "Số lượng điểm dự đoán từ mô hình không khớp với tổng số phim trong Dataset.",
                details={
                    "expected": self._dataset.num_movies,
                    "actual": len(raw_scores),
                },
            )

        seen_indices: set[int] = (
            self._dataset.get_user_seen_movie_indices(user_idx) if exclude_seen else set()
        )

        # Lọc các phim chưa xem và ghép cặp (movie_idx, score)
        candidate_scores: list[tuple[int, float]] = [
            (movie_idx, float(score))
            for movie_idx, score in enumerate(raw_scores)
            if movie_idx not in seen_indices
        ]

        # Sắp xếp giảm dần theo điểm dự đoán; nếu bằng điểm thì ưu tiên movie_idx nhỏ hơn để tất định (deterministic)
        candidate_scores.sort(key=lambda item: (-item[1], item[0]))

        selected = candidate_scores[:k]
        recommendations: list[Recommendation] = []
        for rank_idx, (movie_idx, score) in enumerate(selected, start=1):
            movie = self._dataset.get_movie_by_index(movie_idx)
            recommendations.append(
                Recommendation(
                    movie_id=movie.movie_id,
                    title=movie.title,
                    score=round(score, 4),
                    rank=rank_idx,
                )
            )

        return recommendations

    def predict_score(self, user_id: Any, movie_id: Any) -> float:
        """
        Dự đoán điểm số tương tác của một người dùng cho một bộ phim cụ thể.
        """
        self._ensure_trained()
        self._validate_user_exists(user_id)
        self._validate_movie_exists(movie_id)

        user_idx = self._dataset.get_user_index(user_id)
        movie_idx = self._dataset.get_movie_index(movie_id)

        scores = self._model.predict_for_user(user_idx)
        return float(scores[movie_idx])

    def get_system_summary(self) -> dict[str, Any]:
        """Trả về thông tin tóm tắt trạng thái của dịch vụ."""
        return {
            "model_name": self.model_name,
            "is_trained": self.is_trained,
            "num_users": self._dataset.num_users,
            "num_movies": self._dataset.num_movies,
            "num_ratings": self._dataset.num_ratings,
            "default_top_k": self._default_top_k,
        }

    # -------------------------------------------------------------------------
    # Private Validation Helpers (Đóng gói logic kiểm tra tính hợp lệ nội bộ)
    # -------------------------------------------------------------------------

    def _ensure_trained(self) -> None:
        if not self.is_trained:
            raise ModelNotTrainedError(self.model_name)

    def _validate_user_exists(self, user_id: Any) -> None:
        if user_id is None or (isinstance(user_id, str) and not user_id.strip()):
            raise InvalidParameterError("user_id", user_id, "user_id không được rỗng hoặc None.")
        if not self._dataset.has_user(user_id):
            raise UnknownUserError(user_id)

    def _validate_movie_exists(self, movie_id: Any) -> None:
        if movie_id is None or (isinstance(movie_id, str) and not movie_id.strip()):
            raise InvalidParameterError("movie_id", movie_id, "movie_id không được rỗng hoặc None.")
        if not self._dataset.has_movie(movie_id):
            raise UnknownMovieError(movie_id)

    @staticmethod
    def _validate_top_k(top_k: Any) -> None:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise InvalidParameterError(
                param_name="top_k",
                param_value=top_k,
                reason="top_k phải là một số nguyên dương (> 0).",
            )

    @staticmethod
    def _validate_dataset_not_empty(dataset: DatasetProtocol) -> None:
        try:
            n_users = dataset.num_users
            n_movies = dataset.num_movies
            n_ratings = dataset.num_ratings
        except AttributeError as exc:
            raise DataError("Đối tượng Dataset thiếu các thuộc tính thống kê bắt buộc.") from exc

        if n_users <= 0 or n_movies <= 0 or n_ratings <= 0:
            raise DataError(
                "Tập dữ liệu rỗng hoặc không đầy đủ (số lượng user, movie và rating phải lớn hơn 0).",
                details={
                    "num_users": n_users,
                    "num_movies": n_movies,
                    "num_ratings": n_ratings,
                },
            )