from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from video_analysis.api.entities.group_statistics.views import (
    router as group_statistics_router,
)
from video_analysis.api.entities.video.views import router as video_router

from .db.mongo import connect, disconnect, populate


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan controller for the app, connects to the database instance upon initialization and disconnects when shutting down.

    Args:
        app (FastAPI): The API instance.
    """
    app.state.db_client = connect()
    app.state.db = app.state.db_client.get_database("video_analysis")
    await populate(app.state.db, Path("data/Interview_data(in).csv").resolve())
    yield
    disconnect(app.state.db_client)


app = FastAPI(lifespan=lifespan)

# Routers
app.include_router(
    group_statistics_router, tags=["Group Statistics"], prefix="/groupstats"
)
app.include_router(video_router, tags=["Videos"], prefix="/videos")
