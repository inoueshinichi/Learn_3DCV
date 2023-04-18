"""オイラー角から各回転表現への変換
"""
import os
import sys
import math

import numpy as np

from type_hint import *

from euler_state import EulerState

def euler_to_rot(theta1_deg: float, 
                 theta2_deg: float, 
                 theta3_deg: float, 
                 euler_state: EulerState) -> np.ndarray:

    return euler_state.to_rot(theta1_deg, theta2_deg, theta3_deg)

