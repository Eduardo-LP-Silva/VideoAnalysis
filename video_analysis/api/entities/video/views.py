from typing import Optional

from fastapi import APIRouter, Request

from video_analysis.api.entities.video.models import VideoModelOut
from video_analysis.api.entities.video.service import read

router = APIRouter()


@router.get("/")
async def get_videos(
    request: Request,
    content_id: Optional[str] = None,
    topic: Optional[str] = None,
    predicted_topic: Optional[str] = None,
    tv_show: Optional[str] = None,
) -> list[VideoModelOut]:
    """Returns the matching videos.

    Args:
        request (Request): Request details.
        content_id (Optional[str], optional): Video ID to filter by.. Defaults to None.
        topic (Optional[str], optional): Topic to filter by.. Defaults to None.
        predicted_topic (Optional[str], optional): Predicted topic to filter by. Defaults to None.
        tv_show (Optional[str], optional): Show to filter by. Defaults to None.

    Returns:
        list[VideoModelOut]: Video list (without feature vectors).
    """
    return await read(request.app.state.db, content_id, topic, predicted_topic, tv_show)
