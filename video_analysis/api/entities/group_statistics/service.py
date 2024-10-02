from typing import Optional

import motor.motor_asyncio as motor_asyncio

from .models import GroupStatisticsModel


async def create_many(
    db: motor_asyncio.AsyncIOMotorDatabase, groups_stats: list[GroupStatisticsModel]
):
    """Inserts group statistics into the database.

    Args:
        db (motor_asyncio.AsyncIOMotorDatabase): The database connection.
        groups_stats (list[GroupStatisticsModel]): List with group statistics.
    """
    await db.get_collection("group_statistics").insert_many(
        [g_stats.model_dump(mode="json") for g_stats in groups_stats]
    )


async def read(
    db: motor_asyncio.AsyncIOMotorDatabase,
    category: Optional[str] = None,
    name: Optional[str] = None,
) -> list[GroupStatisticsModel]:
    """Fetches group statistics from the database.

    Args:
        db (motor_asyncio.AsyncIOMotorDatabase): The database connection.
        category (Optional[str], optional): Category to filter by. Defaults to None.
        name (Optional[str], optional): Name to filter by. Defaults to None.

    Returns:
        list[GroupStatisticsModel]: List with group statistics.
    """
    args = locals()
    filter = {
        arg: value for arg, value in args.items() if arg != "db" and value is not None
    }

    return (
        await db.get_collection("group_statistics")
        .find(filter, {"_id": False})
        .to_list()
    )
