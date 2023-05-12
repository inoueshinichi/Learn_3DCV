"""3Dジオメトリのコンテキスト

Default:
    (1) 座標系定義:
        右手座標系 Zup ・・・ OpenCV/Blender系統

         z  
         |
         /---y
        x

    (2) オイラー角:
        内因性 ZXY
"""

import os
import sys

module_parent_dir = '/'.join([os.path.dirname(__file__), '..'])
sys.path.append(module_parent_dir)

import numpy as np

from type_hint import *

from BasicModule.coordinate_state import (
    CoordinateState,
    CoorRightYupXforwardState, # OpenGL系
    CoorRightZupYforwardState, # OpenCV系
    CoorLeftYupZforwardState,  # DirectX系
    CoorLeftZupXforwardState,  # UnrealEngine系
)

from BasicModule.euler_state import (
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
        self.coor_state: CoordinateState = CoorRightZupYforwardState()
        self.euler_state: EulerState = EulerInnerZXYState()

    def look_at(self,
                target_pos: np.ndarray, 
                camera_pos: np.ndarray, 
                up_axis: Tuple[float, float, float]) -> np.ndarray:
        return self.coor_state.look_at(target_pos, camera_pos, up_axis)
    
    def forward_axis(self, rot: np.ndarray) -> np.ndarray:
        return self.coor_state.forward_axis(rot)
    
    def right_axis(self, rot: np.ndarray) -> np.ndarray:
        return self.coor_state.right_axis(rot)
    
    def up_axis(self, rot: np.ndarray) -> np.ndarray:
        return self.coor_state.up_axis(rot)





   
    
    
