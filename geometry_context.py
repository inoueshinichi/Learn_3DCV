"""3Dジオメトリのコンテキスト

Default:
    右手座標系 Zup ・・・ OpenCV/Blender系統

    z  
    |
    /---y
   x

    オイラー角:
    内因性 ZXY
"""

import os
import sys

import numpy as np

from type_hint import *

from coordinate_state import (
    CoordinateState,
    CoorRightYupState,
    CoorRightZupState,
    CoorLeftYupState,
    CoorLeftZupState,
)

from euler_state import (
    EulerState,
    EulerInnerXYXState,
    EulerInnerXZXState,
    EulerInnerYXYState,
    EulerInnerYZYState,
    EulerInnerZXZState,
    EulerInnerZYZState,
    EulerInnerXYZState,
    EulerInnerXZYState,
    EulerInnerYXZState,
    EulerInnerYZXState,
    EulerInnerZXYState,
    EulerInnerZYXState,
    EulerOuterXYXState,
    EulerOuterXZXState,
    EulerOuterYXYState,
    EulerOuterYZYState,
    EulerOuterZXZState,
    EulerOuterZYZState,
    EulerOuterXYZState,
    EulerOuterXZYState,
    EulerOuterYXZState,
    EulerOuterYZXState,
    EulerOuterZXYState,
    EulerOuterZYXState,
)

class GeometryContext:

    def __init__(self):
        self.coor_state: CoordinateState = CoorRightZupState()
        self.euler_state: EulerState = EulerInnerZXYState()

    def look_at(self,
                target_pos: np.ndarray, 
                camera_pos: np.ndarray, 
                up_axis: Tuple[float, float, float]) -> np.ndarray:
        return self.coor_state.look_at(target_pos, camera_pos, up_axis)
