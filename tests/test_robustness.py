"""
tests/test_robustness.py
Kiểm thử tính bền vững và khả năng chịu lỗi (Robustness & Edge-Case Testing)
theo đúng 5 yêu cầu trong đề bài:
1. Unknown users or items
2. Invalid recommendation parameters
3. Empty or incomplete data
4. Invalid configurations
5. Requests made before model training
"""
from __future__ import annotations
import pytest

from recsys.config import DataConfig, EvalConfig, ModelConfig, SystemConfig
from recsys.exceptions import (
    ConfigError,
    DataError,
    InvalidParameterError,
    ModelNotTrainedError,
    UnknownMovieError,
    UnknownUserError,
)
from recsys.service.recommendation_service import RecommendationService
from tests.service.test_service import StubDataset, StubRecommender


# =====================================================================
# 1. KIỂM THỬ UNKNOWN USERS OR ITEMS (Người dùng hoặc phim không tồn tại)
# =====================================================================

def test_unknown_user_raises_unknown_user_error() -> None:
    service = RecommendationService(dataset=StubDataset(), model=StubRecommender())
    service.train()

    with pytest.raises(UnknownUserError) as exc_info:
        service.recommend(user_id=999999, top_k=5)

    assert exc_info.value.user_id == 999999


def test_unknown_movie_in_predict_score_raises_unknown_movie_error() -> None:
    service = RecommendationService(dataset=StubDataset(), model=StubRecommender())
    service.train()

    with pytest.raises(UnknownMovieError) as exc_info:
        service.predict_score(user_id=101, movie_id=888888)

    assert exc_info.value.movie_id == 888888


@pytest.mark.parametrize("bad_id", [None, "", "   "])
def test_blank_or_none_user_id_raises_invalid_parameter_error(bad_id: object) -> None:
    service = RecommendationService(dataset=StubDataset(), model=StubRecommender())
    service.train()

    with pytest.raises(InvalidParameterError):
        service.recommend(user_id=bad_id, top_k=5)


# =====================================================================
# 2. KIỂM THỬ INVALID RECOMMENDATION PARAMETERS (Tham số gợi ý sai)
# =====================================================================

@pytest.mark.parametrize("invalid_k", [0, -1, -100, 3.5, "5", True, False])
def test_invalid_top_k_raises_invalid_parameter_error(invalid_k: object) -> None:
    service = RecommendationService(dataset=StubDataset(), model=StubRecommender())
    service.train()

    with pytest.raises(InvalidParameterError) as exc_info:
        service.recommend(user_id=101, top_k=invalid_k)  # type: ignore[arg-type]

    assert exc_info.value.param_name == "top_k"


def test_invalid_default_top_k_in_constructor_raises_error() -> None:
    with pytest.raises(InvalidParameterError):
        RecommendationService(dataset=StubDataset(), model=StubRecommender(), default_top_k=0)


# =====================================================================
# 3. KIỂM THỬ EMPTY OR INCOMPLETE DATA (Dữ liệu rỗng hoặc không khớp)
# =====================================================================

def test_empty_dataset_rejected_on_service_init() -> None:
    empty_dataset = StubDataset(empty=True)
    with pytest.raises(DataError):
        RecommendationService(dataset=empty_dataset, model=StubRecommender())


def test_none_dataset_rejected_on_service_init() -> None:
    with pytest.raises(DataError):
        RecommendationService(dataset=None, model=StubRecommender())  # type: ignore[arg-type]


def test_corrupted_model_score_length_raises_data_error() -> None:
    dataset = StubDataset()  # Có 4 bộ phim
    bad_model = StubRecommender(scores=[4.0, 5.0])  # Chỉ trả về 2 điểm dự đoán
    service = RecommendationService(dataset=dataset, model=bad_model)
    service.train()

    with pytest.raises(DataError):
        service.recommend(user_id=101, top_k=2)


# =====================================================================
# 4. KIỂM THỬ INVALID CONFIGURATIONS (Cấu hình hệ thống sai)
# =====================================================================

def test_invalid_model_type_raises_config_error() -> None:
    with pytest.raises(ConfigError):
        ModelConfig(model_type="deep_neural_transformer_xyz")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_factors": 0},
        {"n_factors": -10},
        {"learning_rate": 0.0},
        {"learning_rate": -0.01},
        {"reg": -0.5},
        {"n_epochs": 0},
    ],
)
def test_invalid_model_hyperparameters_raise_config_error(kwargs: dict) -> None:
    with pytest.raises(ConfigError):
        ModelConfig(**kwargs)


@pytest.mark.parametrize("bad_ratio", [0.0, 1.0, -0.2, 1.5])
def test_invalid_data_test_ratio_raises_config_error(bad_ratio: float) -> None:
    with pytest.raises(ConfigError):
        DataConfig(test_ratio=bad_ratio)


def test_invalid_rating_bounds_raise_config_error() -> None:
    with pytest.raises(ConfigError):
        DataConfig(min_rating=5.0, max_rating=1.0)


def test_invalid_eval_config_raises_config_error() -> None:
    with pytest.raises(ConfigError):
        EvalConfig(top_k=-5)
    with pytest.raises(ConfigError):
        EvalConfig(metrics=())


def test_system_config_is_immutable() -> None:
    cfg = SystemConfig()
    with pytest.raises(Exception):
        cfg.default_top_k = 99  # type: ignore[misc]


# =====================================================================
# 5. KIỂM THỬ REQUESTS MADE BEFORE MODEL TRAINING (Chưa train đã gọi)
# =====================================================================

def test_recommend_before_training_raises_model_not_trained_error() -> None:
    untrained_model = StubRecommender()
    service = RecommendationService(dataset=StubDataset(), model=untrained_model)

    with pytest.raises(ModelNotTrainedError):
        service.recommend(user_id=101, top_k=3)


def test_predict_score_before_training_raises_model_not_trained_error() -> None:
    untrained_model = StubRecommender()
    service = RecommendationService(dataset=StubDataset(), model=untrained_model)

    with pytest.raises(ModelNotTrainedError):
        service.predict_score(user_id=101, movie_id=1)