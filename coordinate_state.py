"""座標系
"""
import os
import sys
import abc
import inspect

import numpy as np

from rotation import axis_x, axis_y, axis_z

from type_hint import *

def make_nip(img_pts: np.ndarray, K: np.ndarray) -> np.ndarray:
    """カメラ内部パラメータ行列Kを用いて画像平面上の点を正規化画像座標系に変換
    正規化画像座標系(Normalized Image Plane Coordinate)
    同次座標系(x,y,w=1)

    Args:
        img_pts (np.ndarray): 画像平面上の点群[3xN]
        K (np.ndarray): カメラの内部パラメータ行列[3x3]

    Returns:
        np.ndarray: 正規化座標の点群[3xN]
    """
    if img_pts.shape[0] != 3: 
        raise ValueError(f"Not match shape (3,N). Given is {img_pts.shape}")

    npts = np.linalg.inv(K) @ img_pts # [3xN]
    npts /= npts[-1,:] # w=1
    return npts


"""同次座標系 (Homogeneous[Perspective] Coordinate)
@note 同次座標系
同次座標の点はスケールと一緒に定義されているので, 
M=[x,y,z,w]=[sx,sy,sz,sw]=[x/w,y/w,z/w,1]は, いずれも同じ3D点を表す.
m=[x,y,w]=[sx,sy,sw]=[x/w,y/w,1]は, いずれも同じ2D点を表す.
最終的にユークリッド座標系の(x,y,z)と一致させるために点をw=1で正規化することになる.

@note Perspective Division
3D空間を2D空間に投影する方法. M=[x,y,z,1] -> m = [x/z,y/z,1]
"""
def make_homo(v: np.ndarray) -> np.ndarray:
    """同次座標を作成

    Args:
        v (np.ndarray): 2Dor3Dの座標[DxN]
        D: 座標系の次元
        N: 座標点の個数

    Returns:
        np.ndarray: 同次座標[(D+1)xN]
    """
    # 2D or 3D
    if v.shape[0] != 2 or v.shape[0] != 3:
        raise ValueError(f"Not match shape (2or3, N). Given is {v.shape}")
    
    N: int = v.shape[1]
    return np.vstack((v, np.ones((1, N))))  


def normalize_homo(homo_v: np.ndarray) -> np.ndarray:
    """2Dor3Dの同次座標の正規化

    Args:
        homo_v (np.ndarray): 2Dor3Dの同次座標[DxN]
        D: 座標系の次元
        N: 座標点の個数

    Returns:
        np.ndarray: 正規化した同次座標[DxN]
    """
    # 2D or 3D homo
    if homo_v.shape[0] != 2 or homo_v.shape[0] != 3:
        raise ValueError(f"Not match shape (2or3, N). Given is {homo_v.shape}")
    
    N: int = homo_v.shape[1]
    return homo_v / homo_v[-1,:].reshape(-1, N)  


"""座標系設定の状態

  1. 右手系 or 左手系
  2. Yup or Zup
  3. Xforward, Yforward or Zforward

  クロス積(外積)は, 右手系と左手系で定義が異なるので注意.
  https://yaju3d.hatenablog.jp/entry/2013/05/26/215841
"""
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

    def __init__(self, coor_style: str, up_axis: str, forward_axis: str):
        self.coor_style: str = coor_style
        self.up_axis: str = up_axis
        self.forward_axis: str = forward_axis

    @abc.abstractclassmethod
    def look_at(self, 
               target_pos: np.ndarray, 
               camera_pos: np.ndarray, 
               up_axis: Tuple[float, float, float]) -> np.ndarray:
        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__
        raise NotImplementedError(f"No implement {func_name} on {class_name}")
    
    # @abc.abstractclassmethod
    # def decomp_camera(self, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    #     func_name = inspect.currentframe().f_code.co_name
    #     class_name = self.__class__.__name__
    #     raise NotImplementedError(f"No implement {func_name} on {class_name}")
    
    @abc.abstractclassmethod
    def forward_axis(self, rot: np.ndarray) -> np.ndarray:
        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__
        raise NotImplementedError(f"No implement {func_name} on {class_name}")
    
    @abc.abstractclassmethod
    def right_axis(self, rot: np.ndarray) -> np.ndarray:
        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__
        raise NotImplementedError(f"No implement {func_name} on {class_name}")
    
    @abc.abstractclassmethod
    def up_axis(self, rot: np.ndarray) -> np.ndarray:
        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__
        raise NotImplementedError(f"No implement {func_name} on {class_name}")


# 右手座標系 Yup-Xforward (OpenGL, AutoDesk Maya, SolidWorks 系統)
class CoorRightYupXforwardState(CoordinateState):

    def __init__(self):
        super(CoorRightYupXforwardState, self).__init__("right", "yup", "xforward")

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
        cam_eye = ct_vec / np.linalg.norm(ct_vec)

        # カメラ上向き
        cam_up = np.array(up_axis, dtype=np.float32)

        # カメラX軸正向きベクトル(単位ベクトル)
        cam_x = np.cross(cam_eye, cam_up)
        cam_x /= np.linalg.norm(cam_x)

        # カメラY軸正向きベクトル(単位ベクトル)
        cam_y = np.cross(cam_x, cam_eye)
        cam_y /= np.linalg.norm(cam_y)

        # カメラZ軸正向きベクトル(単位ベクトル)
        cam_z = np.cross(cam_x, cam_y)
        cam_z /= np.linalg.norm(cam_z)

        # カメラ位置
        tx = -1.0 * np.dot(cam_x, camera_pos)
        ty = -1.0 * np.dot(cam_y, camera_pos)
        tz = -1.0 * np.dot(cam_z, camera_pos)

        # 4x4行列(ΣWの座標をΣCの座標に変換)
        V = np.array([
            [cam_x[0], cam_y[0], cam_z[0], tx],
            [cam_x[1], cam_y[1], cam_z[1], ty],
            [cam_x[2], cam_y[2], cam_z[2], tz],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32) # 列優先表現

        return V
    
    @CoordinateState.overrides(CoordinateState)
    def forward_axis(self, rot: np.ndarray) -> np.ndarray:
        """座標系の前方ベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: 前方ベクトル[3x1]
        """
        # 行列は列優先表現
        return axis_x(rot) # forward : X軸方向ベクトル
    
    @CoordinateState.overrides(CoordinateState)
    def right_axis(self, rot: np.ndarray) -> np.ndarray:
        """座標系の右方ベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: 右方ベクトル[3x1]
        """
        # 行列は列優先表現
        return axis_z(rot) # right : Z軸方向ベクトル
    
    @CoordinateState.overrides(CoordinateState)
    def up_axis(self, rot: np.ndarray) -> np.ndarray:
        """座標系の上方ベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: 上方ベクトル[3x1]
        """
        # 行列は列優先表現
        return axis_y(rot) # up : Y軸方向ベクトル

        
# 右手座標系 Zup-Yforward (OpenCV, Blender, AutoCAD 系統)
class CoorRightZupYforwardState(CoordinateState):

    def __init__(self):
        super(CoorRightZupYforwardState, self).__init__("right", "zup", "yforward")

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
        # カメラのZ軸正向きベクトル(単位ベクトル) : 視線ベクトル
        ct_vec = target_pos - camera_pos
        cam_eye = ct_vec / np.linalg.norm(ct_vec)

        # カメラ上向き
        cam_up = np.array(up_axis, dtype=np.float32)

        # カメラX軸正向きベクトル(単位ベクトル)
        cam_x = np.cross(cam_eye, cam_up)
        cam_x /= np.linalg.norm(cam_x)

        # カメラY軸正向きベクトル(単位ベクトル)
        cam_y = -1.0 * np.cross(cam_x, cam_eye)
        cam_y /= np.linalg.norm(cam_y)

        # カメラZ軸正向きベクトル(単位ベクトル)
        cam_z = cam_eye

        # カメラ位置
        tx = -1.0 * np.dot(cam_x, camera_pos)
        ty = -1.0 * np.dot(cam_y, camera_pos)
        tz = -1.0 * np.dot(cam_z, camera_pos)

        # 4x4行列(ΣWの座標をΣCの座標に変換)
        V = np.array([
            [cam_x[0], cam_y[0], cam_z[0], tx],
            [cam_x[1], cam_y[1], cam_z[1], ty],
            [cam_x[2], cam_y[2], cam_z[2], tz],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32) # 列優先表現

        return V
    
    @CoordinateState.overrides(CoordinateState)
    def forward_axis(self, rot: np.ndarray) -> np.ndarray:
        """座標系の前方ベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: 前方ベクトル[3x1]
        """
        # 行列は列優先表現
        return axis_y(rot) # forward : Y軸方向ベクトル
    
    @CoordinateState.overrides(CoordinateState)
    def right_axis(self, rot: np.ndarray) -> np.ndarray:
        """座標系の右方ベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: 右方ベクトル[3x1]
        """
        # 行列は列優先表現
        return axis_x(rot) # right : X軸方向ベクトル
    
    @CoordinateState.overrides(CoordinateState)
    def up_axis(self, rot: np.ndarray) -> np.ndarray:
        """座標系の上方ベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: 上方ベクトル[3x1]
        """
        # 行列は列優先表現
        return axis_z(rot) # up : Z軸方向ベクトル


# 左手座標系 Yup-Zforward (Direct3D, Metal, Unity 系統)
class CoorLeftYupZforwardState(CoordinateState):

    def __init__(self):
        super(CoorLeftYupZforwardState, self).__init__("left", "yup", "zforward")

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
        # カメラのZ軸正向きベクトル(単位ベクトル) : 視線ベクトル
        ct_vec = target_pos - camera_pos
        cam_eye = ct_vec / np.linalg.norm(ct_vec)

        # カメラ上向き
        cam_up = np.array(up_axis, dtype=np.float32)

        # カメラX軸正向きベクトル(単位ベクトル)
        cam_x = -1.0 * np.cross(cam_eye, cam_up)
        cam_x /= np.linalg.norm(cam_x)

        # カメラY軸正向きベクトル(単位ベクトル)
        cam_y = -1.0 * np.cross(cam_x, cam_eye)
        cam_y /= np.linalg.norm(cam_y)

        # カメラZ軸正向きベクトル(単位ベクトル)
        cam_z = cam_eye

        # カメラ位置
        tx = -1.0 * np.dot(cam_x, camera_pos)
        ty = -1.0 * np.dot(cam_y, camera_pos)
        tz = -1.0 * np.dot(cam_z, camera_pos)

        # 4x4行列(ΣWの座標をΣCの座標に変換)
        V = np.array([
            [cam_x[0], cam_y[0], cam_z[0], tx],
            [cam_x[1], cam_y[1], cam_z[1], ty],
            [cam_x[2], cam_y[2], cam_z[2], tz],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32) # 列優先表現

        return V
    
    @CoordinateState.overrides(CoordinateState)
    def forward_axis(self, rot: np.ndarray) -> np.ndarray:
        """座標系の前方ベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: 前方ベクトル[3x1]
        """
        # 行列は列優先表現
        return axis_z(rot) # forward : Z軸方向ベクトル
    
    @CoordinateState.overrides(CoordinateState)
    def right_axis(self, rot: np.ndarray) -> np.ndarray:
        """座標系の右方ベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: 右方ベクトル[3x1]
        """
        # 行列は列優先表現
        return axis_x(rot) # right : X軸方向ベクトル
    
    @CoordinateState.overrides(CoordinateState)
    def up_axis(self, rot: np.ndarray) -> np.ndarray:
        """座標系の上方ベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: 上方ベクトル[3x1]
        """
        # 行列は列優先表現
        return axis_y(rot) # up : Y軸方向ベクトル


# 左手座標系 Zup-Xforward (Unreal Engine 系統)
class CoorLeftZupXforwardState(CoordinateState):

    def __init__(self):
        super(CoorLeftZupXforwardState, self).__init__("left", "zup", "xforward")

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
        # カメラのZ軸負向きベクトル(単位ベクトル) : 視線ベクトル
        ct_vec = target_pos - camera_pos
        cam_eye = ct_vec / np.linalg.norm(ct_vec)

        # カメラ上向き
        cam_up = np.array(up_axis, dtype=np.float32)

        # カメラX軸正向きベクトル(単位ベクトル)
        cam_x = -1.0 * np.cross(cam_eye, cam_up)
        cam_x /= np.linalg.norm(cam_x)

        # カメラY軸正向きベクトル(単位ベクトル)
        cam_y = np.cross(cam_x, cam_eye)
        cam_y /= np.linalg.norm(cam_y)

        # カメラZ軸正向きベクトル(単位ベクトル)
        cam_z = np.cross(cam_x, cam_y)
        cam_z /= np.linalg.norm(cam_z)
        
        # カメラ位置
        tx = -1.0 * np.dot(cam_x, camera_pos)
        ty = -1.0 * np.dot(cam_y, camera_pos)
        tz = -1.0 * np.dot(cam_z, camera_pos)

        # 4x4行列(ΣWの座標をΣCの座標に変換)
        V = np.array([
            [cam_x[0], cam_y[0], cam_z[0], tx],
            [cam_x[1], cam_y[1], cam_z[1], ty],
            [cam_x[2], cam_y[2], cam_z[2], tz],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32) # 列優先表現

        return V
    
    @CoordinateState.overrides(CoordinateState)
    def forward_axis(self, rot: np.ndarray) -> np.ndarray:
        """座標系の前方ベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: 前方ベクトル[3x1]
        """
        # 行列は列優先表現
        return axis_x(rot) # forward : X軸方向ベクトル
    
    @CoordinateState.overrides(CoordinateState)
    def right_axis(self, rot: np.ndarray) -> np.ndarray:
        """座標系の右方ベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: 右方ベクトル[3x1]
        """
        # 行列は列優先表現
        return axis_y(rot) # right : Y軸方向ベクトル
    
    @CoordinateState.overrides(CoordinateState)
    def up_axis(self, rot: np.ndarray) -> np.ndarray:
        """座標系の上方ベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: 上方ベクトル[3x1]
        """
        # 行列は列優先表現
        return axis_z(rot) # up : Z軸方向ベクトル