"""座標系の設定
"""
import os
import sys

module_parent_dir = '/'.join([os.path.dirname(__file__), '..'])
sys.path.append(module_parent_dir)

import abc
import inspect

import numpy as np

from type_hint import *

from BasicModule.rotation import axis_x, axis_y, axis_z


"""座標系設定の状態

  1. 右手系 or 左手系
  2. Yup or Zup
  3. Xforward, Yforward or Zforward

  クロス積(外積)は, 右手系と左手系で定義が異なるので注意.
  https://yaju3d.hatenablog.jp/entry/2013/05/26/215841
"""
class CoordinateState(metaclass=abc.ABCMeta):

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

    def __init__(self):
        self.coor_style: Optional(str) = None
        self.up_axis: Optional(str) = None
        self.forward_axis: Optional(str) = None

    @abc.abstractclassmethod
    def look_at(self, 
               target_pos: np.ndarray, 
               camera_pos: np.ndarray, 
               up_axis: Tuple[float, float, float]) -> np.ndarray:
        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__
        raise NotImplementedError(f"No implement {func_name} on {class_name}")
    
    @abc.abstractclassmethod
    def quat_to_rot(self, qw : float, qx : float, qy : float, qz : float) -> np.ndarray:
        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__
        raise NotImplementedError(f"No implement {func_name} on {class_name}")
        
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
        super(CoorRightYupXforwardState, self).__init__()
        self.coor_style: Optional(str) = "right"
        self.up_axis: Optional(str) = "y"
        self.forward_axis: Optional(str) = "x"

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
        super(CoorRightZupYforwardState, self).__init__()
        self.coor_style: Optional(str) = "right"
        self.up_axis: Optional(str) = "z"
        self.forward_axis: Optional(str) = "y"


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
    def quat_to_rot(qw : float, qx : float, qy : float, qz : float) -> np.ndarray:
        """クォータニオンから回転行列に変換

        右手系クォータニオン
        rot = 
        [[1-2*qy^2-2*qz^2, 2*qx*qy-2*qw*qz, 2*qx*qz+2*qw*qy],
         [2*qx*qy+2*qw*qz, 1-2*qx^2-2*qz^2, 2*qy*qz+-2*qw*qx],
         [2*qx*qz-2*qw*qy, 2*qy*qz+2*qw*qx, 1-2*qx^2-2*qy^2]]

        Args:
            rot (np.ndarray): クォータニオン [4x1]

        Returns:
            np.ndarray: 回転行列
        """

        # 右手系クォータニオン
        r11 = 1 - 2 * qy ** 2 - 2 * qz ** 2 # [0,0]
        r12 = 2 * qx * qy - 2 * qw * qz
        r13 = 2 * qx * qz + 2 * qw * qy
        r21 = 2 * qx * qy + 2 * qw * qz
        r22 = 1 - 2 * qx ** 2 - 2 * qz ** 2 # [1,1]
        r23 = 2 * qy * qz - 2 * qw * qx
        r31 = 2 * qx * qz - 2 * qw * qy
        r32 = 2 * qy * qz + 2 * qw * qx
        r33 = 1 - 2 * qx ** 2 - 2 * qy ** 2 # [2,2]

        rot = np.array([
            [r11,r12,r13],
            [r21,r22,r23],
            [r31,r32,r33]
        ], dtype=np.float32)

        return rot
    
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
        super(CoorLeftYupZforwardState, self).__init__()
        self.coor_style: Optional(str) = "left"
        self.up_axis: Optional(str) = "y"
        self.forward_axis: Optional(str) = "z"

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
    def quat_to_rot(qw : float, qx : float, qy : float, qz : float) -> np.ndarray:
        """クォータニオンから回転行列に変換

        左手クォータニオン
        rot = 
        [[1-2*qy^2-2*qz^2, 2*qx*qy-2*qw*qz, 2*qx*qz+2*qw*qy],
         [2*qx*qy+2*qw*qz, 1-2*qx^2-2*qz^2, 2*qy*qz+-2*qw*qx],
         [2*qx*qz-2*qw*qy, 2*qy*qz+2*qw*qx, 1-2*qx^2-2*qy^2]]

        Args:
            rot (np.ndarray): クォータニオン [4x1]

        Returns:
            np.ndarray: 回転行列
        """

        # 右手系クォータニオン
        r11 = 1 - 2 * qy ** 2 - 2 * qz ** 2 # [0,0]
        r12 = 2 * qx * qy - 2 * qw * qz
        r13 = 2 * qx * qz + 2 * qw * qy
        r21 = 2 * qx * qy + 2 * qw * qz
        r22 = 1 - 2 * qx ** 2 - 2 * qz ** 2 # [1,1]
        r23 = 2 * qy * qz - 2 * qw * qx
        r31 = 2 * qx * qz - 2 * qw * qy
        r32 = 2 * qy * qz + 2 * qw * qx
        r33 = 1 - 2 * qx ** 2 - 2 * qy ** 2 # [2,2]

        rot = np.array([
            [r11,r12,r13],
            [r21,r22,r23],
            [r31,r32,r33]
        ], dtype=np.float32)

        return rot
    
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
        super(CoorLeftZupXforwardState, self).__init__()
        self.coor_style: Optional(str) = "left"
        self.up_axis: Optional(str) = "z"
        self.forward_axis: Optional(str) = "x"

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