"""座標系設定の状態

  クロス積(外積)は, 右手系と左手系で定義が異なるので注意.
  https://yaju3d.hatenablog.jp/entry/2013/05/26/215841
"""

import os
import sys
import abc
import inspect

import numpy as np

from type_hint import *

class CoordinateState(abc.ABCMeta):

    # 派生クラスへのインターフェースAPIの強制
    @staticmethod
    def overrides(klass):
        def check_super(method) -> Any:
            method_name = method.__name__
            msg = f"`{method_name}()` is not defined in `{klass.__name__}`."
            assert method_name in dir(klass), msg

        def wrapper(method) -> Any:
            check_super(method)
            return method

        return wrapper

    def __init__(self, coor_style: str, up_axis: str):
        self.coor_style: str = coor_style
        self.up_axis: str = up_axis

    @abc.abstractclassmethod
    def camera_right(self, target_dir: np.ndarray, up_dir: np.ndarray) -> np.ndarray:
        """クロス積(外積)
        右手系・・・右手フレミング 
        左手系・・・左手フレミング
        ※クロス積は右手系と左手系で定義が異なるので要注意.

        Args:
            target_dir (np.ndarray): 始点ベクトル[3x1]
            up_dir (np.ndarray): 終点ベクトル[3x1]

        Returns:
            np.ndarray: 外積ベクトル[3x1]
        """
        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__
        raise NotImplementedError(f"No implement {func_name} on {class_name}")

    @abc.abstractclassmethod
    def look_at(self, 
               target_pos: np.ndarray, 
               camera_pos: np.ndarray, 
               up_axis: Tuple[float, float, float]) -> np.ndarray:
        
        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__
        raise NotImplementedError(f"No implement {func_name} on {class_name}")


# 右手座標系 Yup
class CoorRightYupState(CoordinateState):

    def __init__(self):
        super(CoorRightYupState, self).__init__("right", "yup")

    @CoordinateState.overrides(CoordinateState)
    def camera_right(self, target_dir: np.ndarray, up_dir: np.ndarray) -> np.ndarray:
        other_dir = np.cross(target_dir, up_dir)
        return other_dir / np.linalg.norm(other_dir)

    @CoordinateState.overrides(CoordinateState)
    def look_at(self, 
               target_pos: np.ndarray, 
               camera_pos: np.ndarray, 
               up_axis: Tuple[float, float, float] = [0,1,0]) -> np.ndarray:
        """カメラのView行列[4x4]を求める
        視線方向: Z軸の負の向き
        右方向: X軸の正の向き
        上方向: Y軸の正の向き
         y 視線方向
         |/
         /---x
        z
        Args:
            target_pos (np.ndarray): ターゲット位置ベクトル[3x1]
            camera_pos (np.ndarray): カメラ中心位置ベクトル[3x1]
            up_axis (Tuple[float, float, float]): カメラの上向きベクトル

        Returns:
            np.ndarray: カメラのView行列[4x4]
        """
        # カメラのZ軸負向きベクトル(単位ベクトル) : 視線ベクトル
        ct_vec = target_pos - camera_pos
        cam_forward = ct_vec / np.linalg.norm(ct_vec)

        # カメラ上向き
        up = np.array(up_axis, dtype=np.float32)

        # カメラのX軸正向きベクトル(単位ベクトル) : 右向きベクトル
        cam_right = self.camera_right(cam_forward, up)

        # カメラY軸正向きベクトル(単位ベクトル) : 上向きベクトル
        cam_up = np.cross(cam_right, cam_forward)

        # カメラ位置
        tx = -1.0 * np.dot(cam_right, camera_pos)
        ty = -1.0 * np.dot(cam_up, camera_pos)
        tz = -1.0 * np.dot(cam_forward, camera_pos)

        # 4x4行列(ΣWの座標をΣCの座標に変換)
        M = np.array([
            []
        ])

        
# 右手座標系 Zup
class CoorRightZupState(CoordinateState):

    def __init__(self):
        super(CoorRightZupState, self).__init__("right", "zup")

    @CoordinateState.overrides(CoordinateState)
    def camera_right(self, target_dir: np.ndarray, up_dir: np.ndarray) -> np.ndarray:
        other_dir = np.cross(target_dir, up_dir)
        return other_dir / np.linalg.norm(other_dir)

    @CoordinateState.overrides(CoordinateState)
    def look_at(self, 
               target_pos: np.ndarray, 
               camera_pos: np.ndarray, 
               up_axis: Tuple[float, float, float] = [0,0,1]) -> np.ndarray:
        """カメラのView行列[4x4]を求める
        視線方向: Z軸の正の向き
        右方向: X軸の正の向き
        上方向: Y軸の負の向き
          z : 視線方向
         /
        /---x
        |
        y
        Args:
            target_pos (np.ndarray): ターゲット位置ベクトル[3x1]
            camera_pos (np.ndarray): カメラ中心位置ベクトル[3x1]
            up_axis (Tuple[float, float, float]): カメラの上向きベクトル

        Returns:
            np.ndarray: カメラのView行列[4x4]
        """
        # 視線ベクトル(単位ベクトル) = Z軸正の向き
        ct_vec = target_pos - camera_pos
        forward_vec = ct_vec / np.linalg.norm(ct_vec)

        # 右方向ベクトル = X軸正の向き
        cam
        right_vec = np.cross()


# 左手座標系 Yup
class CoorLeftYupState(CoordinateState):

    def __init__(self):
        super(CoorLeftYupState, self).__init__("left", "yup")

    @CoordinateState.overrides(CoordinateState)
    def camera_right(self, target_dir: np.ndarray, up_dir: np.ndarray) -> np.ndarray:
        other_dir = np.cross(target_dir, up_dir)
        return -1.0 * other_dir / np.linalg.norm(other_dir)

    @CoordinateState.overrides(CoordinateState)
    def look_at(self, 
               target_pos: np.ndarray, 
               camera_pos: np.ndarray, 
               up_axis: Tuple[float, float, float] = [0,1,0]) -> np.ndarray:
        """カメラのView行列[4x4]を求める
        視線方向: Z軸の正の向き
        右方向: X軸の正の向き
        上方向: Y軸の正の向き
         y
         |
         /---z : 視線方向
        x
        Args:
            target_pos (np.ndarray): ターゲット位置ベクトル[3x1]
            camera_pos (np.ndarray): カメラ中心位置ベクトル[3x1]
            up_axis (Tuple[float, float, float]): カメラの上向きベクトル

        Returns:
            np.ndarray: カメラのView行列[4x4]
        """
        pass

# 左手座標系 Zup
class CoorLeftZupState(CoordinateState):

    def __init__(self):
        super(CoorLeftZupState, self).__init__("left", "zup")

    @CoordinateState.overrides(CoordinateState)
    def camera_right(self, target_dir: np.ndarray, up_dir: np.ndarray) -> np.ndarray:
        other_dir = np.cross(target_dir, up_dir)
        return -1.0 * other_dir / np.linalg.norm(other_dir)

    @CoordinateState.overrides(CoordinateState)
    def look_at(self, 
               target_pos: np.ndarray, 
               camera_pos: np.ndarray, 
               up_axis: Tuple[float, float, float] = [0,0,1]) -> np.ndarray:
        """カメラのView行列[4x4]を求める
        視線方向: Z軸の負の向き
        右方向: X軸の正の向き
        上方向: Y軸の負の向き
           視線方向
           /
          /---x
         z|
          y
        Args:
            target_pos (np.ndarray): ターゲット位置ベクトル[3x1]
            camera_pos (np.ndarray): カメラ中心位置ベクトル[3x1]
            up_axis (Tuple[float, float, float]): カメラの上向きベクトル

        Returns:
            np.ndarray: カメラのView行列[4x4]
        """
        pass

