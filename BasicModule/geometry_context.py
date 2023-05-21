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
    CoorRH_PZup_PYforward_PXright_State, # OpenCV系(右手系)
    CoorRH_PYup_PZforward_NXright_State, # OpenGL系(Default: 右手系)
    CoorLH_PYup_PZforward_PXright_State, # DirectX系(Default: 左手系)
    CoorLH_PZup_NYforward_PXright_State, # UnrealEngine系(左手系)
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
        self.coor_state: CoordinateState = CoorRH_PZup_PYforward_PXright_State()
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
    
    def change_rvec(self, rvec: np.ndarray, coor_state: CoordinateState) -> np.ndarray:
        """回転ベクトルを座標定義に従って変換

        Args:
            rvec (np.ndarray): 回転ベクトル[3x1] (vx,vy,vz)
            coor_state (CoordinateState): 座標系定義

        Returns:
            np.ndarray: 指定座標系定義における回転ベクトル[3x1]
        """
        if type(self.coor_state) is type(coor_state):
            return rvec
    

        # source
        if type(self.coor_state) is CoorRH_PZup_PYforward_PXright_State: # OpenCV系

            # destination
            if type(coor_state) is CoorLH_PYup_PZforward_PXright_State: # DirectX系
                # Y軸鏡映
                new_rvec = rvec.copy()
                new_rvec[0] *= -1 # x
                new_rvec[2] *= -1 # z
                return new_rvec
            
            elif type(coor_state) is CoorLH_PZup_NYforward_PXright_State: # UE系
                # Y軸鏡映
                new_rvec = rvec.copy()
                new_rvec[0] *= -1 # x
                new_rvec[2] *= -1 # z
                return new_rvec
            
            else:
                raise TypeError(f"Invalid type(coor_state). Given is {type(coor_state)}")
            
        # source
        elif type(self.coor_state) is CoorRH_PYup_PZforward_NXright_State: # OpenGL系
            # destination
            if type(coor_state) is CoorLH_PYup_PZforward_PXright_State: # DirectX系
                # X軸鏡映
                new_rvec = rvec.copy()
                new_rvec[1] *= -1 # y
                new_rvec[2] *= -1 # z
                return new_rvec
            
            elif type(coor_state) is CoorLH_PZup_NYforward_PXright_State: # UE系
                # X軸鏡映
                new_rvec = rvec.copy()
                new_rvec[1] *= -1 # y
                new_rvec[2] *= -1 # z
                return new_rvec
            
            else:
                raise TypeError(f"Invalid type(coor_state). Given is {type(coor_state)}")
            
        # source
        elif type(self.coor_state) is CoorLH_PYup_PZforward_PXright_State: # DirectX系
            # destination
            if type(coor_state) is CoorRH_PZup_PYforward_PXright_State: # OpenCV系
                # Y軸鏡映
                new_rvec = rvec.copy()
                new_rvec[0] *= -1 # x
                new_rvec[2] *= -1 # z
                return new_rvec
            
            elif type(coor_state) is CoorRH_PYup_PZforward_NXright_State: # OpenGL系
                # X軸鏡映
                new_rvec = rvec.copy()
                new_rvec[1] *= -1 # y
                new_rvec[2] *= -1 # z
                return new_rvec
            
            else:
                raise TypeError(f"Invalid type(coor_state). Given is {type(coor_state)}")
            
        # source 
        elif type(self.coor_state) is CoorLH_PZup_NYforward_PXright_State: # UE系
            # destination
            if type(coor_state) is CoorRH_PZup_PYforward_PXright_State: # OpenCV系
                # Y軸鏡映
                new_rvec = rvec.copy()
                new_rvec[0] *= -1 # x
                new_rvec[2] *= -1 # z
                return new_rvec
            
            elif type(coor_state) is CoorRH_PZup_PYforward_PXright_State: # OpenGL系
                # X軸鏡映
                new_rvec = rvec.copy()
                new_rvec[1] *= -1 # y
                new_rvec[2] *= -1 # z
                return new_rvec
            
            else:
                raise TypeError(f"Invalid type(coor_state). Given is {type(coor_state)}")
            
        else:
            raise TypeError(f"Invalid type(self.coor_state). Given is {type(self.coor_state)}")
        

    def change_quat(self, quat: np.ndarray, coor_state: CoordinateState) -> np.ndarray:
        """クォータニオンを座標系定義に従って変換

        Args:
            quat (np.ndarray): クォータニオン[4x1]
            coor_state (CoordinateState): 座標系定義

        Returns:
            np.ndarray: 指定座標系定義におけるクォータニオン[4x1]
        """
        if type(self.coor_state) is type(coor_state):
            return quat
    

        # source
        if type(self.coor_state) is CoorRH_PZup_PYforward_PXright_State: # OpenCV系

            # destination
            if type(coor_state) is CoorLH_PYup_PZforward_PXright_State: # DirectX系
                # Y軸鏡映
                new_quat = quat.copy()
                new_quat[0] *= -1 # x
                new_quat[2] *= -1 # z
                return new_quat
            
            elif type(coor_state) is CoorLH_PZup_NYforward_PXright_State: # UE系
                # Y軸鏡映
                new_quat = quat.copy()
                new_quat[0] *= -1 # x
                new_quat[2] *= -1 # z
                return new_quat
            
            else:
                raise TypeError(f"Invalid type(coor_state). Given is {type(coor_state)}")
            
        # source
        elif type(self.coor_state) is CoorRH_PYup_PZforward_NXright_State: # OpenGL系

            # destination
            if type(coor_state) is CoorLH_PYup_PZforward_PXright_State: # DirectX系
                # X軸鏡映
                new_quat = quat.copy()
                new_quat[1] *= -1 # y
                new_quat[2] *= -1 # z
                return new_quat
            
            elif type(coor_state) is CoorLH_PZup_NYforward_PXright_State: # UE系
                # X軸鏡映
                new_quat = quat.copy()
                new_quat[1] *= -1 # y
                new_quat[2] *= -1 # z
                return new_quat
            
            else:
                raise TypeError(f"Invalid type(coor_state). Given is {type(coor_state)}")
            
        # source
        elif type(self.coor_state) is CoorLH_PYup_PZforward_PXright_State: # DirectX系

            # destination
            if type(coor_state) is CoorRH_PZup_PYforward_PXright_State: # OpenCV系
                # X軸鏡映
                new_quat = quat.copy()
                new_quat[1] *= -1 # y
                new_quat[2] *= -1 # z
                return new_quat
            
            elif type(coor_state) is CoorRH_PYup_PZforward_NXright_State: # OpenGL系
                # X軸鏡映
                new_quat = quat.copy()
                new_quat[1] *= -1 # y
                new_quat[2] *= -1 # z
                return new_quat
            
            else:
                raise TypeError(f"Invalid type(coor_state). Given is {type(coor_state)}")
            
        # source 
        elif type(self.coor_state) is CoorLH_PZup_NYforward_PXright_State: # UE系

            # destination
            if type(coor_state) is CoorRH_PZup_PYforward_PXright_State: # OpenCV系
                # X軸鏡映
                new_quat = quat.copy()
                new_quat[1] *= -1 # y
                new_quat[2] *= -1 # z
                return new_quat
            
            elif type(coor_state) is CoorRH_PZup_PYforward_PXright_State: # OpenGL系
                # X軸鏡映
                new_quat = quat.copy()
                new_quat[1] *= -1 # y
                new_quat[2] *= -1 # z
                return new_quat
            
            else:
                raise TypeError(f"Invalid type(coor_state). Given is {type(coor_state)}")
            
        else:
            raise TypeError(f"Invalid type(self.coor_state). Given is {type(self.coor_state)}")
        




   
    
    
