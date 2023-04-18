"""3Dジオメトリのコンテキスト

Default:
    右手座標系
    Yup

    Y  
    |
    /---x
   z

    オイラー角:
    内因性 ZXY
"""

import os
import sys

import numpy as np

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
        self.coordinate_state: CoordinateState = CoorRightYupState()
        self.euler_state: EulerState = EulerInnerZXYState()

