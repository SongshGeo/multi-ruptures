#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

import numpy as np
import pytest

from tests.helper import generate_testing_numpy_data


@pytest.fixture(name="three_segments")
def three_segment_data() -> tuple[np.ndarray, list[int]]:
    """A synthetic time series with three segments."""
    segments = [100, 200]
    means = [10, 15, 8]
    return (
        generate_testing_numpy_data(n_points=300, segments=segments, means=means),
        segments,
    )


@pytest.fixture(name="two_segments")
def two_segment_data() -> tuple[np.ndarray, list[int]]:
    """A synthetic time series with two segments."""
    segments = [100]
    means = [10, 15]
    return (
        generate_testing_numpy_data(n_points=300, segments=segments, means=means),
        segments,
    )
