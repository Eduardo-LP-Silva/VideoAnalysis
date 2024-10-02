from typing import Optional

import motor.motor_asyncio as motor_asyncio

from .models import VideoModel, VideoModelOut


async def create_many(db: motor_asyncio.AsyncIOMotorDatabase, videos: list[VideoModel]):
    """Inserts videos into the database.

    Args:
        db (motor_asyncio.AsyncIOMotorDatabase): The database connection.
        videos (list[VideoModel]): Video list.
    """
    await db.get_collection("videos").insert_many(
        [g_stats.model_dump(mode="json") for g_stats in videos]
    )


async def read(
    db: motor_asyncio.AsyncIOMotorDatabase,
    content_id: Optional[str] = None,
    topic: Optional[str] = None,
    predicted_topic: Optional[str] = None,
    tv_show: Optional[str] = None,
) -> list[VideoModelOut]:
    """Fetches videos from the database.

    Args:
        db (motor_asyncio.AsyncIOMotorDatabase): The database connection.
        content_id (Optional[str], optional): Video ID to filter by. Defaults to None.
        topic (Optional[str], optional): Topic to filter by. Defaults to None.
        predicted_topic (Optional[str], optional): Predicted topic to filter by. Defaults to None.
        tv_show (Optional[str], optional): Show to filter by. Defaults to None.

    Returns:
        list[VideoModelOut]: Video list (without feature vectors).
    """
    args = locals()
    filter = {
        arg: value for arg, value in args.items() if arg != "db" and value is not None
    }

    return (
        await db.get_collection("videos")
        .find(filter, {"_id": False, "feature_vector": False})
        .to_list()
    )
