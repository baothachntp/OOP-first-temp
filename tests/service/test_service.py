"""
tests/service/test_service.py
Kiểm thử đơn vị (Unit Tests) cho ModelFactory, RecommendationService và RecSysCLI.
"""
from __future__ import annotations
from dataclasses import dataclass
import pytest

from recsys.config import ModelConfig, SystemConfig
from recsys.service.cli import RecSysCLI
from recsys.service.recommendation_service import RecommendationService
from recsys.service.model_factory import ModelFactory


@dataclass(frozen=True)
class DummyMovie:
    movie_id: int
    title: str


class StubDataset:
    """Dataset giả lập (Test Double) phục vụ kiểm thử độc lập tầng Service."""

    def __init__(self, empty: bool = False) -> None:
        if empty:
            self._users: dict[int, int] = {}
            self._movies: list[DummyMovie] = []
            self._seen: dict[int, set[int]] = {}
            self._num_ratings = 0
        else:
            self._users = {101: 0, 102: 1}
            self._movies = [
                DummyMovie(movie_id=1, title="The Matrix (1999)"),
                DummyMovie(movie_id=2, title="Inception (2010)"),
                DummyMovie(movie_id=3, title="Interstellar (2014)"),
                DummyMovie(movie_id=4, title="Parasite (2019)"),
            ]
            # User 101 (index 0) đã xem phim index 0 (The Matrix)
            self._seen = {0: {0}, 1: set()}
            self._num_ratings = 5

        self._movie_id_to_idx = {m.movie_id: idx for idx, m in enumerate(self._movies)}

    @property
    def num_users(self) -> int:
        return len(self._users)

    @property
    def num_movies(self) -> int:
        return len(self._movies)

    @property
    def num_ratings(self) -> int:
        return self._num_ratings

    def has_user(self, user_id: int) -> bool:
        return user_id in self._users

    def has_movie(self, movie_id: int) -> bool:
        return movie_id in self._movie_id_to_idx

    def get_user_index(self, user_id: int) -> int:
        return self._users[user_id]

    def get_movie_index(self, movie_id: int) -> int:
        return self._movie_id_to_idx[movie_id]

    def get_movie_by_index(self, movie_idx: int) -> DummyMovie:
        return self._movies[movie_idx]

    def get_user_seen_movie_indices(self, user_idx: int) -> set[int]:
        return set(self._seen.get(user_idx, set()))


class StubRecommender:
    """Mô hình gợi ý giả lập trả về điểm số cố định để kiểm thử tính đúng đắn của xếp hạng."""

    def __init__(self, scores: list[float] | None = None) -> None:
        self._is_trained = False
        self._scores = scores or [4.9, 3.2, 4.5, 4.1]

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def fit(self, dataset: object) -> None:
        self._is_trained = True

    def predict_for_user(self, user_idx: int) -> list[float]:
        return list(self._scores)


def test_recommend_top_k_excludes_seen_and_sorts_correctly() -> None:
    dataset = StubDataset()
    # Điểm cho 4 phim: Phim 0 (4.9 - đã xem), Phim 1 (3.2), Phim 2 (4.5), Phim 3 (4.1)
    model = StubRecommender(scores=[4.9, 3.2, 4.5, 4.1])
    service = RecommendationService(dataset=dataset, model=model, default_top_k=2)

    service.train()
    recs = service.recommend(user_id=101, top_k=2, exclude_seen=True)

    assert len(recs) == 2
    # Phim 0 có điểm 4.9 cao nhất nhưng User 101 đã xem nên phải bị loại.
    # Top 1 phải là Phim 2 (id=3, Interstellar, 4.5) và Top 2 là Phim 3 (id=4, Parasite, 4.1)
    assert recs[0].movie_id == 3
    assert recs[0].title == "Interstellar (2014)"
    assert recs[0].score == pytest.approx(4.5)
    assert recs[0].rank == 1

    assert recs[1].movie_id == 4
    assert recs[1].rank == 2


def test_recommend_include_seen_when_flag_is_false() -> None:
    dataset = StubDataset()
    model = StubRecommender(scores=[4.9, 3.2, 4.5, 4.1])
    service = RecommendationService(dataset=dataset, model=model)
    service.train()

    recs = service.recommend(user_id=101, top_k=1, exclude_seen=False)
    assert len(recs) == 1
    assert recs[0].movie_id == 1  # The Matrix được giữ lại vì exclude_seen=False


def test_predict_score_returns_exact_value() -> None:
    dataset = StubDataset()
    model = StubRecommender(scores=[4.9, 3.2, 4.5, 4.1])
    service = RecommendationService(dataset=dataset, model=model)
    service.train()

    score = service.predict_score(user_id=101, movie_id=3)
    assert score == pytest.approx(4.5)


def test_model_factory_registers_and_creates_custom_model() -> None:
    ModelFactory.register("popularity", lambda cfg: StubRecommender())
    cfg = ModelConfig(model_type="popularity")
    dataset = StubDataset()

    service = RecommendationService.from_config(dataset=dataset, model_config=cfg)
    assert service.model_name == "StubRecommender"


def test_cli_commands_work_end_to_end() -> None:
    dataset = StubDataset()
    model = StubRecommender()
    service = RecommendationService(dataset=dataset, model=model)
    cli = RecSysCLI(service)

    # Gọi recommend trước khi train trong CLI -> Trả về thông báo lỗi nghiệp vụ lịch sự, không crash
    untrained_out = cli.execute_command("recommend 101 2")
    assert "ModelNotTrainedError" in untrained_out

    # Gọi train qua CLI
    train_out = cli.execute_command("train")
    assert "[OK]" in train_out

    # Gọi recommend sau khi train
    rec_out = cli.execute_command("recommend 101 2")
    assert "Interstellar (2014)" in rec_out
    assert "Parasite (2019)" in rec_out


def test_system_config_from_dict() -> None:
    raw = {
        "data": {"test_ratio": 0.25},
        "model": {"model_type": "matrix_factorization", "n_factors": 32},
        "evaluation": {"top_k": 5, "metrics": ["ndcg", "recall"]},
        "default_top_k": 5,
    }
    sys_cfg = SystemConfig.from_dict(raw)
    assert sys_cfg.data.test_ratio == 0.25
    assert sys_cfg.model.n_factors == 32
    assert sys_cfg.evaluation.metrics == ("ndcg", "recall")