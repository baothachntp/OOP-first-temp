"""
src/recsys/service/model_factory.py
Factory Pattern để khởi tạo các mô hình gợi ý từ ModelConfig.
"""
from __future__ import annotations
from typing import Callable, Protocol, Sequence, runtime_checkable

from recsys.config import ModelConfig
from recsys.exceptions import ConfigError


@runtime_checkable
class RecommenderProtocol(Protocol):
    """
    Giao diện trừu tượng (Protocol) đại diện cho lớp Recommender của Minh Hương.
    Giúp kiểm tra kiểu (type-checking) và giảm sự phụ thuộc cứng (loose coupling).
    """
    @property
    def is_trained(self) -> bool: ...

    def fit(self, dataset: object) -> None: ...

    def predict_for_user(self, user_idx: int) -> Sequence[float]: ...


ModelBuilder = Callable[[ModelConfig], RecommenderProtocol]


class ModelFactory:
    """
    Nhà máy khởi tạo đối tượng Recommender dựa trên cấu hình (Factory & Registry Pattern).
    Che giấu chi tiết khởi tạo (constructor arguments) của từng thuật toán cụ thể.
    """

    _registry: dict[str, ModelBuilder] = {}

    @classmethod
    def register(cls, model_type: str, builder: ModelBuilder) -> None:
        """Đăng ký một hàm khởi tạo mô hình mới vào Factory."""
        key = model_type.strip().lower()
        if not key:
            raise ConfigError("Tên loại mô hình đăng ký không được để trống.")
        cls._registry[key] = builder

    @classmethod
    def create(cls, config: ModelConfig) -> RecommenderProtocol:
        """
        Khởi tạo và trả về một thể hiện (instance) của Recommender tương ứng với ModelConfig.
        """
        if not isinstance(config, ModelConfig):
            raise ConfigError(
                "Tham số truyền vào ModelFactory.create phải là một ModelConfig.",
                details={"received_type": type(config).__name__},
            )

        # Tự động nạp các model mặc định từ package recsys.models nếu chưa được đăng ký
        cls._ensure_default_models_registered()

        model_type = config.model_type.lower()
        builder = cls._registry.get(model_type)
        if builder is None:
            raise ConfigError(
                f"Không tìm thấy trình khởi tạo cho mô hình '{model_type}'. "
                f"Các mô hình đã đăng ký: {sorted(cls._registry.keys())}",
                details={"model_type": model_type},
            )

        return builder(config)

    @classmethod
    def available_models(cls) -> list[str]:
        """Trả về danh sách các loại mô hình hiện có trong Factory."""
        cls._ensure_default_models_registered()
        return sorted(cls._registry.keys())

    @classmethod
    def _ensure_default_models_registered(cls) -> None:
        """
        Lazy-loading các mô hình từ module `recsys.models` của Minh Hương.
        Sử dụng cơ chế import an toàn để tầng Service vẫn có thể test độc lập
        với Mock models ngay cả khi module `recsys.models` chưa hoàn thiện.
        """
        if "popularity" not in cls._registry:
            try:
                from recsys.models.popularity import PopularityRecommender
                cls.register(
                    "popularity",
                    lambda cfg: PopularityRecommender(),
                )
            except ImportError:
                pass

        if "matrix_factorization" not in cls._registry:
            try:
                from recsys.models.matrix_factorization import MatrixFactorizationRecommender
                cls.register(
                    "matrix_factorization",
                    lambda cfg: MatrixFactorizationRecommender(
                        n_factors=cfg.n_factors,
                        learning_rate=cfg.learning_rate,
                        reg=cfg.reg,
                        n_epochs=cfg.n_epochs,
                        random_seed=cfg.random_seed,
                    ),
                )
            except ImportError:
                pass

        if "bpr" not in cls._registry:
            try:
                from recsys.models.bpr import BPRRecommender
                cls.register(
                    "bpr",
                    lambda cfg: BPRRecommender(
                        n_factors=cfg.n_factors,
                        learning_rate=cfg.learning_rate,
                        reg=cfg.reg,
                        n_epochs=cfg.n_epochs,
                        random_seed=cfg.random_seed,
                    ),
                )
            except ImportError:
                pass