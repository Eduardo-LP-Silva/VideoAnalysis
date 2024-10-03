from typing import Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

from video_analysis.api.entities.group_statistics.models import (
    EvalMetricsModel,
    GroupCategoryEnum,
    GroupStatisticsModel,
)
from video_analysis.api.entities.video.models import VideoModel


def process_data(
    video_df: pd.DataFrame,
) -> dict[str, Union[list[GroupStatisticsModel], list[VideoModel]]]:
    """Processes a dataframe containing video information, calculating group statistics and video information.

    Args:
        video_df (pd.DataFrame): The dataframe to process.

    Returns:
        dict[str, Union[list[GroupStatisticsModel], list[VideoModel]]]: Dictionary with two entries: "group_stats", representing a list of statistics for
        the DataFrame groups, and "videos", a list with the video information present in the original DataFrame, modified to include numerical feature vectors and
        addicional PCA components.
    """
    processed_df = video_df.copy()
    # Convert feature vector string column to actual float arrays
    processed_df["feature_vector"] = list(
        np.array(processed_df["feature_vector"].str.split(", ").to_list(), dtype=float)
    )

    group_stats = get_video_groups_stats(processed_df)
    videos = get_videos(processed_df)

    return {"group_stats": group_stats, "videos": videos}


def get_videos(video_df: pd.DataFrame) -> list[VideoModel]:
    """Obtains a list with the video information present in the given DataFrame, modified to include numerical feature vectors and addicional PCA components

    Args:
        video_df (pd.DataFrame): The base DataFrame.

    Returns:
        list[VideoModel]: The list with the video information.
    """
    processed_df = video_df.copy()
    processed_df.rename(
        {
            "actual_label": "topic",
            "predicted_label": "predicted_topic",
            "tvshow": "tv_show",
        },
        axis=1,
        inplace=True,
    )
    video_records = processed_df.to_dict(orient="records")
    pca_res = PCA(n_components=2).fit_transform(
        np.array(processed_df["feature_vector"].to_list())
    )

    return [
        VideoModel(**video, pca_x=pca_res[idx, 0], pca_y=pca_res[idx, 1])
        for idx, video in enumerate(video_records)
    ]


def get_video_groups_stats(video_df: pd.DataFrame) -> list[GroupStatisticsModel]:
    """Obtains a list of statistics for certain groups present in the given DataFrame, including evaluation metrics such as precision, recall and f-1 score, as well
    as the average intra group cosine similarity.

    Args:
        video_df (pd.DataFrame): The DataFrame containing video information.

    Returns:
        list[GroupStatisticsModel]: List with the groups' statistical information.
    """
    groups: list[GroupStatisticsModel] = []
    video_topic_list = video_df["actual_label"].unique()

    report = classification_report(
        video_df["actual_label"],
        video_df["predicted_label"],
        zero_division=np.nan,
        output_dict=True,
    )

    # Evaluation metrics and similarity for the whole set
    general_eval_metrics = EvalMetricsModel(
        precision=report["macro avg"]["precision"],
        recall=report["macro avg"]["recall"],
        f1_score=report["macro avg"]["f1-score"],
    )
    general_avg_cosine_distance = pdist(
        video_df["feature_vector"].to_list(), metric="cosine"
    ).mean()

    groups.append(
        GroupStatisticsModel(
            name="all",
            category=GroupCategoryEnum.all,
            eval_metrics=general_eval_metrics,
            avg_cosine_intra_similarity=general_avg_cosine_distance,
        )
    )

    # Evaluation metrics and similarity for video type groups
    type_groups = video_df.groupby("actual_label")["feature_vector"].aggregate(
        lambda group: pdist(group.to_list(), metric="cosine").mean()
    )

    for video_topic in video_topic_list:
        eval_metrics = EvalMetricsModel(
            precision=report[video_topic]["precision"],
            recall=report[video_topic]["recall"],
            f1_score=report[video_topic]["f1-score"],
        )
        groups.append(
            GroupStatisticsModel(
                category=GroupCategoryEnum.topic,
                name=video_topic,
                eval_metrics=eval_metrics,
                avg_cosine_intra_similarity=type_groups[video_topic],
            )
        )

    # Evaluation metrics and similarity for tv show groups
    tvshow_groups = video_df.groupby("tvshow")["feature_vector"].aggregate(
        lambda group: pdist(group.to_list(), metric="cosine").mean()
    )

    # Add report metrics to dataframe
    report_df = video_df.copy()
    report_metrics = ["precision", "recall", "f1-score"]

    for metric in report_metrics:
        metric_map = {
            label: report[label][metric] for label in report_df["actual_label"].unique()
        }
        report_df[metric] = report_df["actual_label"].map(metric_map)
        report_df[f"tvshow_avg_{metric}"] = report_df.groupby("tvshow")[
            metric
        ].transform("mean")

    for tv_show in report_df["tvshow"].unique():
        example_record = report_df.loc[report_df["tvshow"] == tv_show].iloc[0]
        eval_metrics = EvalMetricsModel(
            precision=example_record["tvshow_avg_precision"],
            recall=example_record["tvshow_avg_recall"],
            f1_score=example_record["tvshow_avg_f1-score"],
        )
        groups.append(
            GroupStatisticsModel(
                category=GroupCategoryEnum.tv_show,
                name=tv_show,
                eval_metrics=eval_metrics,
                avg_cosine_intra_similarity=tvshow_groups[tv_show],
            )
        )

    return groups
