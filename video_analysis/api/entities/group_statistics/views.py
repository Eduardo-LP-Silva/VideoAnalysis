from fastapi import APIRouter, Request
from typing import Optional
from video_analysis.api.entities.group_statistics.models import GroupStatisticsModel
from video_analysis.api.entities.group_statistics.service import read

router = APIRouter()

@router.get("/")
async def get_group_stats(
    request: Request,
    category: Optional[str] = None, name: Optional[str] = None
) -> list[GroupStatisticsModel]:
    """Returns group statistics.

    Args:
        request (Request): Request details.
        category (Optional[str], optional): Category to filter by. Defaults to None.
        name (Optional[str], optional): Name to filter by. Defaults to None.

    Returns:
        list[GroupStatisticsModel]: List with group statistics.
    """
    return await read(request.app.state.db, category, name)