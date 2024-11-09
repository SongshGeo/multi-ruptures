#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

from __future__ import annotations

from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd


def generate_testing_numpy_data(
    n_points: int = 300,
    segments: Sequence[int] = (100, 200),
    means: Sequence[float] = (10, 15, 8),
    seed: int | None = 0,
) -> np.ndarray:
    """Generate a synthetic time series with arbitrary segments.

    Args:
        n_points: Total number of points in the series.
        segments: Breakpoint positions, must be in ascending order.
        means: Mean values for each segment. Length should be len(segments) + 1.
        scale: Standard deviation for all segments.
        seed: Random seed for reproducibility.

    Returns:
        pd.Series: Time series with specified segments and properties.
    """
    if len(means) != len(segments) + 1:
        raise ValueError("Number of means should be equal to number of segments + 1")
    if not all(x < y for x, y in zip(segments[:-1], segments[1:])):
        raise ValueError("Segments must be in ascending order")
    if any(x >= n_points for x in segments):
        raise ValueError("Segment positions cannot exceed n_points")

    np.random.seed(seed=seed)
    scale = np.random.randint(1, (max(means) - min(means)) / len(segments))

    # 计算每段的大小
    segment_sizes = []
    prev_pos = 0
    for pos in segments:
        segment_sizes.append(pos - prev_pos)
        prev_pos = pos
    segment_sizes.append(n_points - prev_pos)

    # 生成每段数据
    data_segments = [
        np.random.normal(loc=mean, scale=scale, size=size)
        for mean, size in zip(means, segment_sizes)
    ]

    # 合并数据
    return np.concatenate(data_segments)


def testing_series_data(
    data: np.ndarray,
    index_type: Literal["date", "int"] = "date",
    start: Optional[str | int] = None,
) -> pd.Series:
    """Convert numpy data to a pandas Series with a specified index type.

    Args:
        data: Numpy data to convert.
        index_type: Type of index, either "date" or "int".
        start: Start of the index, either a string or an integer.

    Returns:
        pd.Series: Converted pandas Series.
    """
    if start is None:
        start = "2024-01-01" if index_type == "date" else 0
    if index_type == "date":
        assert isinstance(start, str), f"start must be a string, got {type(start)}."
        index = pd.date_range(start=start, periods=len(data))
    elif index_type == "int":
        assert isinstance(start, int), f"start must be an integer, got {type(start)}."
        index = range(start, start + len(data))
    else:
        raise ValueError(f"Invalid index type: {index_type}")
    return pd.Series(data, index=index)
