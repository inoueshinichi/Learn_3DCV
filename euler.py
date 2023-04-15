"""オイラー角から各回転表現への変換
"""
import os
import sys
import math

import numpy as np

from type_hint import *

from euler_state import EulerState

def euler_to_rot(ax_deg: float, 
                 ay_deg: float, 
                 az_deg: float, 
                 euler_state: EulerState) -> np.ndarray:

    return euler_state.to_rot(ax_deg, ay_deg, az_deg)

def euler_from_rot(rot: np.ndarray,
                   euler_state: EulerState) -> Tuple[float, float, float]:

    return euler_state.from_rot(rot)