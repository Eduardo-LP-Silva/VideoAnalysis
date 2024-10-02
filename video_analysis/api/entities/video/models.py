from pydantic import BaseModel


class VideoModelOut(BaseModel):
    content_id: str
    topic: str
    predicted_topic: str
    pca_x: float
    pca_y: float
    tv_show: str


class VideoModel(VideoModelOut):
    feature_vector: list[float]
