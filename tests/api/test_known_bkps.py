#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

"""
Test known breakpoints.
"""

import numpy as np
from numpy.testing import assert_allclose

from src.api.pettitt import iterative_pettitt


def test_three_segments(three_segments: tuple[np.ndarray, tuple]) -> None:
    """Test three segments."""
    # Unpack data and segments
    data, segments = three_segments
    # Test
    assert iterative_pettitt(data) == segments


def test_two_segments(two_segments: tuple[np.ndarray, tuple]) -> None:
    """Test two segments."""
    data, segments = two_segments
    # Test
    result = np.array(iterative_pettitt(data))
    # Allow for 5% relative tolerance
    assert_allclose(result, np.array(segments), rtol=5)
