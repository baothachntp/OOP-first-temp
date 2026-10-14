"""Object-oriented recommendation system package."""

from .config import RecommendationConfig
from .exceptions import (
    ConfigError,
    DataError,
    InvalidParameterError,
    ModelNotTrainedError,
    RecSysError,
    UnknownMovieError,
    UnknownUserError,
)

__all__ = [
    "ConfigError",
    "DataError",
    "InvalidParameterError",
    "ModelNotTrainedError",
    "RecommendationConfig",
    "RecSysError",
    "UnknownMovieError",
    "UnknownUserError",
]