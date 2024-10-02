import os

import motor.motor_asyncio as motor_asyncio
import pandas as pd

from video_analysis.api.data_processing import process_data
from video_analysis.api.entities.group_statistics.service import (
    create_many as add_group_stats,
)
from video_analysis.api.entities.video.service import create_many as add_videos


def connect() -> motor_asyncio.AsyncIOMotorClient:
    """Connects to a running MongoDB instance.

    Returns:
        motor_asyncio.AsyncIOMotorClient: The database client connection.
    """
    # Connection details should be in a .env file but for simplicity purposes they are directly declared here
    return motor_asyncio.AsyncIOMotorClient("mongodb://root:password@mongodb:27017/")


def disconnect(db_client: motor_asyncio.AsyncIOMotorClient):
    """Disconnects from a MongoDB instance.

    Args:
        db_client (motor_asyncio.AsyncIOMotorClient): The database client.
    """
    db_client.close()


async def populate(db: motor_asyncio.AsyncIOMotorDatabase, data_file_path: os.PathLike):
    """Populates the database by parsing data from a local file.

    Args:
        db (motor_asyncio.AsyncIOMotorDatabase): The database connection.
        data_file_path (os.PathLike): The path to the local file to process.
    """
    # Deletes previous data
    # Only done to ensure a clean state everytime in this scenario
    await db.get_collection("group_statistics").delete_many({})
    await db.get_collection("videos").delete_many({})

    video_df = pd.read_csv(str(data_file_path), sep=";", index_col=0)

    processed_data = process_data(video_df)

    await add_group_stats(db, processed_data["group_stats"])
    await add_videos(db, processed_data["videos"])
