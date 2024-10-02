from enum import Enum

from pydantic import BaseModel, Field


class GroupCategoryEnum(Enum):
    all = "all"
    topic = "topic"
    tv_show = "tv_show"


class EvalMetricsModel(BaseModel):
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1_score: float = Field(ge=0.0, le=1.0)


class GroupStatisticsModel(BaseModel):
    category: GroupCategoryEnum
    name: str
    eval_metrics: EvalMetricsModel
    avg_cosine_intra_similarity: float = Field(ge=0.0, le=1.0)
