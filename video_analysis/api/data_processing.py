import pandas as pd
from sklearn.decomposition import PCA
from video_analysis.api.entities.group_statistics.models import GroupStatisticsModel, EvalMetricsModel, GroupCategoryEnum
from video_analysis.api.entities.video.models import VideoModel
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from scipy.spatial.distance import pdist
from typing import Union

def process_data(video_df: pd.DataFrame) -> dict[str, Union[list[GroupStatisticsModel], list[VideoModel]]]:
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
    processed_df["feature_vector"] = list(np.array(processed_df["feature_vector"].str.split(", ").to_list(), dtype=float))

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
    processed_df.rename({"actual_label": "topic", "predicted_label": "predicted_topic", "tvshow": "tv_show"}, axis=1, inplace=True)
    video_records = processed_df.to_dict(orient="records")
    pca_res = PCA(n_components=2).fit_transform(np.array(processed_df["feature_vector"].to_list()))

    return [VideoModel(**video, pca_x=pca_res[idx, 0], pca_y=pca_res[idx, 1]) for idx, video in enumerate(video_records)]


def get_video_groups_stats(video_df: pd.DataFrame) -> list[GroupStatisticsModel]:
    """Obtains a list of statistics for certain groups present in the given DataFrame, including evaluation metrics such as precision, recall and f-1 score, as well
    as the average intra group cosine similarity.

    Args:
        video_df (pd.DataFrame): The DataFrame containing video information.

    Returns:
        list[GroupStatisticsModel]: List with the groups' statistical information.
    """
    groups: list[GroupStatisticsModel] = []
    video_type_list = video_df["actual_label"].unique()

    # Calculate metrics and similarity for entire set
    general_eval_metrics = calc_eval_metrics(video_df, labels=video_type_list)
    general_avg_cosine_distance = pdist(video_df["feature_vector"].to_list(), metric="cosine").mean()
    
    groups.append(GroupStatisticsModel(name="all", category=GroupCategoryEnum.all, eval_metrics=general_eval_metrics, avg_cosine_intra_similarity=general_avg_cosine_distance))

    # Calculate metrics and similarity for video type groups
    type_groups = video_df.groupby("actual_label")["feature_vector"].aggregate(lambda group: pdist(group.to_list(), metric="cosine").mean())

    for video_type in video_type_list:
        eval_metrics = calc_eval_metrics(video_df.loc[video_df["actual_label"] == video_type], labels=video_type_list)
        groups.append(GroupStatisticsModel(category=GroupCategoryEnum.topic, name=video_type, eval_metrics=eval_metrics, avg_cosine_intra_similarity=type_groups[video_type]))

    # Calculate metrics and similarity for tv show groups
    tvshow_groups = video_df.groupby("tvshow")["feature_vector"].aggregate(lambda group: pdist(group.to_list(), metric="cosine").mean())

    for tv_show in video_df["tvshow"].unique():
        eval_metrics = calc_eval_metrics(video_df.loc[video_df["tvshow"] == tv_show], labels=video_type_list)
        groups.append(GroupStatisticsModel(category=GroupCategoryEnum.tv_show, name=tv_show, eval_metrics=eval_metrics, avg_cosine_intra_similarity=tvshow_groups[tv_show]))

    return groups
    
def calc_eval_metrics(video_df: pd.DataFrame, labels: list[str]) -> EvalMetricsModel:
    """Calculates evaluation metrics for a given DataFrame.

    Args:
        video_df (pd.DataFrame): The video DataFrame.
        labels (list[str]): The labels to consider.

    Returns:
        EvalMetricsModel: The precision, recall and f-1 score, macro-averaged.
    """
    precision = precision_score(video_df["actual_label"].values, video_df["predicted_label"].values, labels=labels, average="macro", zero_division=1.0)
    recall = recall_score(video_df["actual_label"].values, video_df["predicted_label"].values, labels=labels, average="macro", zero_division=1.0)
    f1 = f1_score(video_df["actual_label"].values, video_df["predicted_label"].values, labels=labels, average="macro", zero_division=1.0)

    return EvalMetricsModel(precision=precision, recall=recall, f1_score=f1)
