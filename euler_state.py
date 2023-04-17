"""オイラー角
Default:
    内因性, ZXY

24パターン:
    内因性 or 外因性
    古典オイラー角
        XYX, XZX, YXY, YZY, ZXZ, ZYZ
    ブライアント角
        XYZ, XZY, YXZ, YZX, ZXY, ZYX

ジンバルロック発生条件:
    古典オイラー角 -> 第二軸が0,pi
    ブライアント角 -> 第二軸が±pi/2
    ※ ジンバルロック発生時は, 第三軸の角度=0とすることで,
    回転行列からオイラー角を一意に決める.

行列は列優先表現とする. numpyのデータ配列は行優先.

内因性と外因性の関係:
    内因性ZXY = 外因性YXZ : (R(Z)@R(X)@R(Y))
"""

'''
内因性 XYX 完
内因性 XZX 完
内因性 YXY 完
内因性 YZY
内因性 ZXZ
内因性 ZYZ
内因性 XYZ 完
内因性 XZY 完
内因性 YXZ 完
内因性 YZX 完
内因性 ZXY 完
内因性 ZYX 完
---------
外因性 XYX
外因性 XZX
外因性 YXY
外因性 YZY
外因性 ZXZ
外因性 ZYZ
外因性 XYZ
外因性 XZY
外因性 YXZ
外因性 YZX
外因性 ZXY
外因性 ZXY
'''


import os
import sys
import math
import abc
import inspect

import numpy as np

from type_hint import *

from rotation import ax_rot, ay_rot, az_rot


class EulerState(abc.ABCMeta):

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

    def __init__(self, gibmal_eps: float=0.001):
        self.gimbal_eps: float = gibmal_eps

    @abc.abstractmethod
    def to_rot(self,
               ax_deg: float, 
               ay_deg: float, 
               az_deg: float) -> np.ndarray:
        
        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__
        raise NotImplementedError(f"No implement {func_name} on {class_name}")
    
    @abc.abstractclassmethod
    def from_rot(self,
                 rot: np.ndarray) -> Tuple[float, float, float]:
    
        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__
        raise NotImplementedError(f"No implement {func_name} on {class_name}")


# 内因性 XYX(X1YX2)
class EulerInnerXYXState(EulerState):

    def __init__(self):
        super(EulerInnerXYXState, self).__init__()

    @EulerState.overrides(EulerState)
    def to_rot(self,
               ax_deg: float,
               ay_deg: float,
               az_deg: float) -> np.ndarray:
        
        return ax_rot(ax_deg) @ ay_rot(ay_deg) @ ax_rot(ax_deg)
    
    @EulerState.overrides(EulerState)
    def from_rot(self,
                 rot: np.ndarray) -> Tuple[float, float, float]:
        """回転行列からオイラー角を計算
        Y軸回りの回転が0,πのときジンバルロック発生.

        Args:
            rot (np.ndarray): 回転行列

        [[C(Y), S(Y)S(X2), C(X2)S(Y)],
         [S(X1)S(Y), C(X1)C(X2)-C(Y)S(X1)S(X2), -C(X1)S(X2)-C(Y)C(X2)S(X1)],
         [-C(X1)S(Y), C(X2)S(X1)+C(X1)C(Y)S(X2), C(X1)C(Y)C(X2)-S(X1)S(X2)]]

        Y=0のとき
        r22=cos(X1+X2), r32=sin(X1+X2)
        Z=piのとき
        r22=cos(X1-X2), r32=sin(X1-X2)
        Returns:
            Tuple[float, float, float]: XYX euler
        """
        if rot.shape != (3,3):
            raise ValueError(f"Not match shape (3,3). Given is {rot.shape}")
        
        r11, r12, r13 = rot[0,0], rot[0,1], rot[0,2]
        r21, r22, r23 = rot[1,0], rot[1,1], rot[1,2]
        r31, r32, r33 = rot[2,0], rot[2,1], rot[2,2]

        ax1_deg, ay_deg, ax2_deg = 0.0, 0.0, 0.0

        # ジンバルロックの確認
        # r11=cos(Y)の値で場合分け
        if math.abs(r11 - 1.0) < self.gimbal_lock:
            # r11 == +1, Y=0
            ax1_deg = math.degrees(math.atan2(r32, r22))
            ay_deg = 0.0
            ax2_deg = 0.0 # Y軸のジンバルロックに従属
        elif math.abs(r11 + 1.0) < self.gimbal_lock:
            # r11 == -1, Y=pi
            ax1_deg = math.degrees(math.atan2(r32, r22))
            ay_deg = math.degrees(math.pi)
            ax2_deg = 0.0 # Y軸のジンバルロックに従属
        else:
            ax1_deg = math.degrees(math.atan2(r21, -r31))
            ay_deg = math.degrees(math.acos(r11))
            ax2_deg = math.degrees(math.atan2(r12, r13))

# 内因性 XZX(X1ZX2)
class EulerInnerXZXState(EulerState):

    def __init__(self):
        super(EulerInnerXZXState, self).__init__()

    @EulerState.overrides(EulerState)
    def to_rot(self,
               ax_deg: float,
               ay_deg: float,
               az_deg: float) -> np.ndarray:
        
        return ax_rot(ax_deg) @ az_rot(az_deg) @ ax_rot(ax_deg)   
    
    @EulerState.overrides(EulerState)
    def from_rot(self,
                 rot: np.ndarray) -> Tuple[float, float, float]:
        """回転行列からオイラー角を計算
        Z軸回りの回転が0,πのときジンバルロック発生.

        Args:
            rot (np.ndarray): 回転行列

        [[C(Z), -C(X2)S(Z), S(Z)S(X2)],
         [C(X1)S(Z), C(X1)S(Z)C(X2)-S(X1)S(X2), -C(X2)S(X1)-C(X1)C(Z)S(X2)],
         [S(X1)S(Z), C(X1)S(X2)+C(Z)C(X2)S(X1), C(X1)C(X2)-C(Z)S(X1)S(X2)]]

        Z=0のとき
        r23=-sin(X1+X2), r33=cos(X1+X2)
        Z=piのとき
        r23=-sin(X1-X2), r33=cos(X1-X2)
        Returns:
            Tuple[float, float, float]: XZX euler
        """
        if rot.shape != (3,3):
            raise ValueError(f"Not match shape (3,3). Given is {rot.shape}")
        
        r11, r12, r13 = rot[0,0], rot[0,1], rot[0,2]
        r21, r22, r23 = rot[1,0], rot[1,1], rot[1,2]
        r31, r32, r33 = rot[2,0], rot[2,1], rot[2,2]

        ax1_deg, az_deg, ax2_deg = 0.0, 0.0, 0.0

        # ジンバルロックの確認
        # r11=cos(Z)の値で場合分け
        if math.abs(r11 - 1.0) < self.gimbal_eps:
            # r11 == +1, Z=0
            ax1_deg = math.degrees(math.atan2(-r23, r33))
            az_deg = 0.0 
            ax2_deg = 0.0 # Z軸のジンバルロックに従属
        elif math.abs(r11 + 1.0) < self.gimbal_eps:
            # r11 == -1, Z=pi
            ax1_deg = math.degrees(math.atan2(-r23, r33))
            az_deg = math.degrees(math.pi)
            ax2_deg = 0.0 # Z軸のジンバルロックに従属
        else:
            ax1_deg = math.degrees(math.atan2(r31, r21))
            az_deg = math.degrees(math.acos(r11))
            ax2_deg = math.degrees(math.atan2(r13, -r12))

        # XZX euler
        return (ax1_deg, az_deg, ax2_deg)

# 内因性 YXY(Y1XY2)
class EulerInnerYXYState(EulerState):

    def __init__(self):
        super(EulerInnerYXYState, self).__init__()

    @EulerState.overrides(EulerState)
    def to_rot(self,
               ax_deg: float,
               ay_deg: float,
               az_deg: float) -> np.ndarray:
        
        return ay_rot(ay_deg) @ ax_rot(ax_deg) @ ay_rot(ay_deg)
    
    @EulerState.overrides(EulerState)
    def from_rot(self,
                 rot: np.ndarray) -> Tuple[float, float, float]:
        """回転行列からオイラー角を計算
        X軸回りの回転が0,πのときジンバルロック発生.

        Args:
            rot (np.ndarray): 回転行列

        [[C(Y1)C(Y2)-C(X)S(Y1)S(Y2), S(Y1)S(X), C(Y1)S(Y2)+C(X)C(Y2)S(Y1)],
         [S(X)S(Y2), C(X), -C(Y2)S(X)],
         [-C(Y2)S(Y1)-C(Y1)C(X)S(Y2), C(Y1)S(X), C(Y1)C(X)C(Y2)-S(Y1)S(Y2)]]

        X=0のとき
        r13=sin(Y1+Y2), r33=cos(Y1+Y2)
        Z=piのとき
        r13=-sin(Y1-Y2), r33-=cos(Y1-Y2)
        Returns:
            Tuple[float, float, float]: YXY euler
        """
        if rot.shape != (3,3):
            raise ValueError(f"Not match shape (3,3). Given is {rot.shape}")
        
        r11, r12, r13 = rot[0,0], rot[0,1], rot[0,2]
        r21, r22, r23 = rot[1,0], rot[1,1], rot[1,2]
        r31, r32, r33 = rot[2,0], rot[2,1], rot[2,2]

        ay1_deg, ax_deg, ay2_deg = 0.0, 0.0, 0.0

        # ジンバルロックの確認
        # r22=cos(X)の値で場合分け
        if math.abs(r22 - 1.0) < self.gimbal_eps:
            # r22 == +1, X=0
            ay1_deg = math.degrees(math.atan2(r11, r33))
            ax_deg = 0.0
            ay2_deg = 0.0 # X軸のジンバルロックに従属
        elif math.abs(r22 + 1.0) < self.gimbal_eps:
            ay1_deg = math.degrees(math.atan2(r11, r33))
            ax_deg = math.degrees(math.pi)
            ay2_deg = 0.0 # Y軸のジンバルロックに従属
        else:
            ay1_deg = math.degrees(math.atan2(r12, r32))
            ax_deg = math.degrees(math.acos(r22))
            ay2_deg = math.degrees(math.atan2(r21, -r23))

        # Euler YXY
        return (ay1_deg, ax_deg, ay2_deg)

# 内因性 YZY
class EulerInnerYZYState(EulerState):

    def __init__(self):
        super(EulerInnerYZYState, self).__init__()

    @EulerState.overrides(EulerState)
    def to_rot(self,
               ax_deg: float,
               ay_deg: float,
               az_deg: float) -> np.ndarray:
        
        return ay_rot(ay_deg) @ az_rot(az_deg) @ ay_rot(ay_deg)


# 内因性 ZXZ
class EulerInnerZXZState(EulerState):

    def __init__(self):
        super(EulerInnerZXZState, self).__init__()

    @EulerState.overrides(EulerState)
    def to_rot(self,
               ax_deg: float,
               ay_deg: float,
               az_deg: float) -> np.ndarray:
        
        return az_rot(az_deg) @ ax_rot(ax_deg) @ az_rot(az_deg)

# 内因性 ZYZ
class EulerInnerZYZState(EulerState):

    def __init__(self):
        super(EulerInnerZYZState, self).__init__()

    @EulerState.overrides(EulerState)
    def to_rot(self,
               ax_deg: float,
               ay_deg: float,
               az_deg: float) -> np.ndarray:
        
        return az_rot(az_deg) @ ay_rot(ay_deg) @ az_rot(az_deg)

# 内因性 XYZ
class EulerInnerXYZState(EulerState):

    def __init__(self):
        super(EulerInnerXYZState, self).__init__()

    @EulerState.overrides(EulerState)
    def to_rot(self,
               ax_deg: float,
               ay_deg: float,
               az_deg: float) -> np.ndarray:
        
        return ax_rot(ax_deg) @ ay_rot(ay_deg) @ az_rot(az_deg)
    
    @EulerState.overrides(EulerState)
    def from_rot(self,
                 rot: np.ndarray) -> Tuple[float, float, float]:
        """回転行列からオイラー角を計算
        Y軸回りの回転が±π/2のときジンバルロック発生.

        Args:
            rot (np.ndarray): 回転行列

            R_global = R(X)@R(Y)@R(Z)

            [[C(Y)C(Z), -C(Y)S(Z), S(Y)],
             [C(X)S(Z)+C(Z)S(X)S(Y), C(X)C(Z)-S(X)S(Y)S(Z), -C(Y)S(X)],
             [S(X)S(Z)-C(X)C(Z)S(Y), C(Z)S(X)+C(X)S(Y)S(Z), C(X)C(Y)]]

            Y=+pi/2のとき
            r22 = cos(X+Z), r32 = sin(X+Z) where Z=0
            Y=-pi/2のとき
            r22 = cos(X-Z), r32 = sin(X-Z) where Z=0

        Returns:
            Tuple[float, float, float]: XYZ euler
        """
        if rot.shape != (3,3):
            raise ValueError(f"Not match shape (3,3). Given is {rot.shape}")
        
        r11, r12, r13 = rot[0,0], rot[0,1], rot[0,2]
        r21, r22, r23 = rot[1,0], rot[1,1], rot[1,2]
        r31, r32, r33 = rot[2,0], rot[2,1], rot[2,2]

        ax_deg, ay_deg, az_deg = 0.0, 0.0, 0.0

        # ジンバルロックの確認
        # r13=sin(Y)の値で場合分け
        if math.abs(r13 - 1.0) < self.gimbal_eps:
            # r13 == +1, Y=pi/2
            ax_deg = math.degrees(math.atan2(r32, r22))
            ay_deg = math.degrees(math.pi/2)
            az_deg = 0.0 # Y軸のジンバルロックに従属
        elif math.abs(r13 + 1.0) < self.gimbal_eps:
            # r13 == -1, Y=-pi/2
            ax_deg = math.degrees(math.atan2(r32, r22))
            ay_deg = math.degrees(-math.pi/2)
            az_deg = 0.0 # Y軸のジンバルロックに従属
        else:
            # -1 < r13 < +1
            ax_deg = math.degrees(math.atan2(-r23, r33))
            ay_deg = math.degrees(math.asin(r13))
            az_deg = math.degrees(math.atan2(-r12, r11))

        # XYZ euler
        return (ax_deg, ay_deg, az_deg)
    

# 内因性 XZY
class EulerInnerXZYState(EulerState):

    def __init__(self):
        super(EulerInnerXZYState, self).__init__()

    @EulerState.overrides(EulerState)
    def to_rot(self,
               ax_deg: float,
               ay_deg: float,
               az_deg: float) -> np.ndarray:
        
        return ax_rot(ax_deg) @ az_rot(az_deg) @ ay_rot(ay_deg)
    
    @EulerState.overrides(EulerState)
    def from_rot(self,
                 rot: np.ndarray) -> Tuple[float, float, float]:
        """回転行列からオイラー角を計算
        Z軸回りの回転が±π/2のときジンバルロック発生.

        Args:
            rot (np.ndarray): 回転行列

            R_global = R(X)@R(Z)@R(Y)

            [[C(Z)C(Y), -S(Z), C(Z)S(Y)],
             [S(X)S(Y)+C(X)C(Y)C(Z), C(X)C(Z), C(X)S(Z)S(Y)-C(Y)S(X)],
             [C(Y)S(X)S(Z)-C(X)S(Y), C(Z)S(X), C(X)C(Y)+S(X)S(Z)S(Y)]]

            Z=+pi/2のとき
            r21 = cos(X-Y), r31 = sin(X-Y) where Y=0
            Z=-pi/2のとき
            r21 = -cos(X+Y), r31 = -sin(X+Y) where Y=0

        Returns:
            Tuple[float, float, float]: XZY euler
        """
        if rot.shape != (3,3):
            raise ValueError(f"Not match shape (3,3). Given is {rot.shape}")
        
        r11, r12, r13 = rot[0,0], rot[0,1], rot[0,2]
        r21, r22, r23 = rot[1,0], rot[1,1], rot[1,2]
        r31, r32, r33 = rot[2,0], rot[2,1], rot[2,2]

        ax_deg, ay_deg, az_deg = 0.0, 0.0, 0.0

        # ジンバルロックの確認
        # r12=-sin(Z)の値で場合分け
        if math.abs(r12 - 1.0) < self.gimbal_eps:
            # r12 == +1, Z=-pi/2
            ax_deg = math.degrees(math.atan2(r31, r21))
            az_deg = math.degrees(-math.pi/2)
            ay_deg = 0.0 # Z軸のジンバルロックに従属
        elif math.abs(r12 + 1.0) < self.gimbal_eps:
            # r12 == -1, Z=pi/2
            ax_deg = math.degrees(math.atan2(r31, r21))
            az_deg = math.degrees(math.pi/2)
            ay_deg = 0.0 # Z軸のジンバルロックに従属
        else:
            # -1 < r12 < +1
            ax_deg = math.degrees(math.atan2(r32, r22))
            az_deg = math.degrees(-math.asin(r12))
            ay_deg = math.degrees(math.atan2(r13, r11))

        # XZY euler
        return (ax_deg, az_deg, ay_deg)


# 内因性 YXZ
class EulerInnerYXZState(EulerState):

    def __init__(self):
        super(EulerInnerYXZState, self).__init__()

    @EulerState.overrides(EulerState)
    def to_rot(self,
               ax_deg: float,
               ay_deg: float,
               az_deg: float) -> np.ndarray:
        
        return ay_rot(ay_deg) @ ax_rot(ax_deg) @ az_rot(az_deg)
    
    @EulerState.overrides(EulerState)
    def from_rot(self,
                 rot: np.ndarray) -> Tuple[float, float, float]:
        """回転行列からオイラー角を計算
        X軸回りの回転が±π/2のときジンバルロック発生.

        Args:
            rot (np.ndarray): 回転行列

            R_global = R(Y)@R(X)@R(Z)

            [[C(Y)C(Z)+S(Y)S(X)S(Z), C(Z)S(Y)S(X)-C(Y)S(Z), C(X)S(Y)],
             [C(X)S(Z), C(X)C(Z), -S(X)],
             [C(Y)S(X)S(Z)-C(Z)S(Y), C(Y)C(Z)S(X)+S(Y)S(Z), C(Y)C(X)]]

            X=+pi/2のとき
            r12=sin(Y-Z), r32=cos(Y-Z) where Z=0
            X=-pi/2のとき
            r12=-sin(Y+Z), r32=-cos(Y+Z) where Z=0

        Returns:
            Tuple[float, float, float]: YXZ euler
        """
        if rot.shape != (3,3):
            raise ValueError(f"Not match shape (3,3). Given is {rot.shape}")
        
        r11, r12, r13 = rot[0,0], rot[0,1], rot[0,2]
        r21, r22, r23 = rot[1,0], rot[1,1], rot[1,2]
        r31, r32, r33 = rot[2,0], rot[2,1], rot[2,2]

        ax_deg, ay_deg, az_deg = 0.0, 0.0, 0.0

        # ジンバルロックの確認
        # r23=-sin(X)の値で場合分け
        if math.abs(r23 - 1.0) < self.gimbal_eps:
            # r23 == +1, X=-pi/2
            ay_deg = math.degrees(math.atan2(r12, r32))
            ax_deg = math.degrees(-math.pi/2)
            az_deg = 0.0 # X軸のジンバルロックに従属
        elif math.abs(r23 + 1.0) < self.gimbal_eps:
            # r23 == -1, X=pi/2
            ay_deg = math.degrees(math.atan2(r12, r32))
            ax_deg = math.degrees(math.pi/2)
            az_deg = 0.0 # X軸のジンバルロックに従属
        else:
            # -1 < r23 < +1
            ay_deg = math.degrees(math.atan2(r13, r33))
            ax_deg = math.degrees(-math.asin(r23))
            az_deg = math.degrees(math.atan2(r21, r22))

        # YXZ euler
        return (ay_deg, ax_deg, az_deg)


# 内因性 YZX
class EulerInnerYZXState(EulerState):

    def __init__(self):
        super(EulerInnerYZXState, self).__init__()

    @EulerState.overrides(EulerState)
    def to_rot(self,
               ax_deg: float,
               ay_deg: float,
               az_deg: float) -> np.ndarray:
        
        return ay_rot(ay_deg) @ az_rot(az_deg) @ ax_rot(ax_deg)
    
    @EulerState.overrides(EulerState)
    def from_rot(self,
                 rot: np.ndarray) -> Tuple[float, float, float]:
        """回転行列からオイラー角を計算
        Z軸回りの回転が±π/2のときジンバルロック発生.

        Args:
            rot (np.ndarray): 回転行列

            R_global = R(Y)@R(Z)@R(X)

            [[C(Y)C(Z), S(Y)S(X)-C(Y)C(X)S(Z), C(X)S(Y)+C(Y)S(Z)S(X)],
             [S(Z), C(Z)C(X), -C(Z)S(X)],
             [-C(Z)S(Y), C(Y)S(X)+C(X)S(Y)S(Z), C(Y)C(X)-S(Y)S(Z)S(X)]]

            Z=+pi/2のとき
            r13=sin(Y+X), r33=cos(Y+X) where X=0
            Z=-pi/2のとき
            r13=sin(Y-X), r33=cos(Y-X) where X=0

        Returns:
            Tuple[float, float, float]: YZX euler
        """
        if rot.shape != (3,3):
            raise ValueError(f"Not match shape (3,3). Given is {rot.shape}")
        
        r11, r12, r13 = rot[0,0], rot[0,1], rot[0,2]
        r21, r22, r23 = rot[1,0], rot[1,1], rot[1,2]
        r31, r32, r33 = rot[2,0], rot[2,1], rot[2,2]

        ax_deg, ay_deg, az_deg = 0.0, 0.0, 0.0

        # ジンバルロックの確認
        # r21=sin(Z)の値で場合分け
        if math.abs(r21 - 1.0) < self.gimbal_eps:
            # r21 == +1, Z=pi/2
            ay_deg = math.degrees(math.atan2(r13, r33))
            az_deg = math.degrees(math.pi/2)
            ax_deg = 0.0 # Z軸のジンバルロックに従属
        elif math.abs(r21 + 1.0) < self.gimbal_eps:
            # r21 == -1, Z=-pi/2
            ay_deg = math.degrees(math.atan2(r13, r33))
            az_deg = math.degrees(-math.pi_2)
            ax_deg = 0.0 # Z軸のジンバルロックに従属
        else:
            # -1 < r21 < +1
            ay_deg = math.degrees(math.atan2(-r31, r11))
            az_deg = math.degrees(math.asin(r21))
            ax_deg = math.degrees(math.atan2(-r23, r22))

        # YZX euler
        return (ay_deg, az_deg, ax_deg)

# 内因性 ZXY
class EulerInnerZXYState(EulerState):

    def __init__(self):
        super(EulerInnerZXYState, self).__init__()

    @EulerState.overrides(EulerState)
    def to_rot(self,
               ax_deg: float,
               ay_deg: float,
               az_deg: float) -> np.ndarray:
        
        return az_rot(az_deg) @ ax_rot(ax_deg) @ ay_rot(ay_deg)
    
    @EulerState.overrides(EulerState)
    def from_rot(self,
                 rot: np.ndarray) -> Tuple[float, float, float]:
        """回転行列からオイラー角を計算
        X軸回りの回転が±π/2のときジンバルロック発生.

        Args:
            rot (np.ndarray): 回転行列

            R_global = R(Z)@R(X)@R(Y)

            [[C(Z)C(Y)-S(Z)S(X)S(Y), -C(X)S(Z), C(Z)S(Y)+C(Y)S(Z)S(X)],
             [C(Y)S(Z)+C(Z)S(X)S(Y), C(Z)C(X), S(Z)S(Y)-C(Z)C(X)S(Y)],
             [-C(X)S(Y), S(X), C(X)C(Y)]]

            X=+pi/2のとき
            r11=cos(Z+Y), r21=sin(Z+Y) where Y=0
            X=-pi/2のとき
            r11=cos(Z-Y), r21=sin(Z-Y) where Y=0

        Returns:
            Tuple[float, float, float]: ZXY euler
        """
        if rot.shape != (3,3):
            raise ValueError(f"Not match shape (3,3). Given is {rot.shape}")
        
        r11, r12, r13 = rot[0,0], rot[0,1], rot[0,2]
        r21, r22, r23 = rot[1,0], rot[1,1], rot[1,2]
        r31, r32, r33 = rot[2,0], rot[2,1], rot[2,2]

        ax_deg, ay_deg, az_deg = 0.0, 0.0, 0.0

        # ジンバルロックの確認
        # r32=sin(X)の値をチェックして場合分け
        if math.abs(r32 - 1.0) < self.gimbal_eps: 
            # r32 == +1, X=pi/2
            az_deg = math.degrees(math.atan2(r21, r11))
            ax_deg = math.degrees(math.pi/2)
            ay_deg = 0.0 # X軸のジンバルロックに従属
        elif math.abs(r32 + 1.0) < self.gimbal_eps: 
            # r32 == -1, X=-pi/2
            az_deg = math.degrees(math.atan2(r21, r11))
            ax_deg = math.degrees(-math.pi/2)
            ay_deg = 0.0 # X軸のジンバルロックに従属
        else: 
            # -1 < r32 < +1
            az_deg = math.degrees(math.atan2(-r12, r22))
            ax_deg = math.degrees(math.asin(r32))
            ay_deg = math.degrees(math.atan2(-r31, r33))

        # ZXY euler
        return (az_deg, ax_deg, ay_deg)

# 内因性 ZYX
class EulerInnerZYXState(EulerState):
    
    def __init__(self):
        super(EulerInnerZYXState, self).__init__()

    @EulerState.overrides(EulerState)
    def to_rot(self,
               ax_deg: float,
               ay_deg: float,
               az_deg: float) -> np.ndarray:
        
        return az_rot(az_deg) @ ay_rot(ay_deg) @ ax_rot(ax_deg)
    
    @EulerState.overrides(EulerState)
    def from_rot(self,
                 rot: np.ndarray) -> Tuple[float, float, float]:
        """回転行列からオイラー角を計算
        Y軸回りの回転が±π/2のときジンバルロック発生.

        Args:
            rot (np.ndarray): 回転行列

            R_global = R(Z)@R(Y)@R(X)

            [[C(Z)C(Y), C(Z)S(Y)S(X)-C(X)S(Z), S(Z)S(X)+C(Z)C(X)S(Y)],
             [C(Y)S(Z), C(Z)C(X)+S(Z)S(Y)S(X), C(X)S(Z)S(Y)-C(Z)S(X)],
             [-S(Y), C(Y)S(X), C(Y)C(X)]]

            Y=+pi/2のとき
            r13=cos(Z-X), r23=sin(Z-X) where X=0
            Y=-pi/2のとき
            r13=-cos(Z+X), r23=-sin(Z+X) where X=0
            
        Returns:
            Tuple[float, float, float]: ZYX euler
        """
        if rot.shape != (3,3):
            raise ValueError(f"Not match shape (3,3). Given is {rot.shape}")
        
        r11, r12, r13 = rot[0,0], rot[0,1], rot[0,2]
        r21, r22, r23 = rot[1,0], rot[1,1], rot[1,2]
        r31, r32, r33 = rot[2,0], rot[2,1], rot[2,2]

        ax_deg, ay_deg, az_deg = 0.0, 0.0, 0.0

        # ジンバルロックの確認
        # r31=-sin(Y)の値をチェックして場合分け
        if math.abs(r31 - 1.0) < self.gimbal_eps: 
            # r31 == +1, Y=-pi/2
            az_deg = math.degrees(math.atan2(r23, r13))
            ay_deg = math.degrees(-math.pi/2)
            ax_deg = 0.0 # Y軸回りのジンバルロックに従属
        elif math.abs(r31 + 1.0) < self.gimbal_eps: 
            # r31 == -1, Y=pi/2
            az_deg = math.degrees(math.atan2(r23, r13))
            ay_deg = math.degrees(math.pi/2)
            ax_deg = 0.0 # Y軸回りのジンバルロックに従属
        else: 
            # -1 < r31 < +1
            az_deg = math.degrees(math.atan2(r21, r11))
            ay_deg = math.degrees(-math.asin(r31))
            ax_deg = math.degrees(math.atan2(r32, r33))

        # ZYX euler
        return (az_deg, ay_deg, ax_deg)

# -----------------------------------------------------

# 外因性 XYX
class EulerOuterXYXState(EulerState):

    def __init__(self):
        super(EulerOuterXYXState, self).__init__()
    
    @EulerState.overrides(EulerState)
    def to_rot(self,
               ax_deg: float,
               ay_deg: float,
               az_deg: float) -> np.ndarray:
        
        return ax_rot(ax_deg) @ ay_rot(ay_deg) @ ax_rot(ax_deg)

# 外因性 XZX
class EulerOuterXZXState(EulerState):

    def __init__(self):
        super(EulerOuterXZXState, self).__init__()

    @EulerState.overrides(EulerState)
    def to_rot(self,
               ax_deg: float,
               ay_deg: float,
               az_deg: float) -> np.ndarray:
        
        return ax_rot(ax_deg) @ az_rot(az_deg) @ ax_rot(ax_deg)

# 外因性 YXY
class EulerOuterYXYState(EulerState):

    def __init__(self):
        super(EulerOuterYXYState, self).__init__()

    @EulerState.overrides(EulerState)
    def to_rot(self,
               ax_deg: float,
               ay_deg: float,
               az_deg: float) -> np.ndarray:
        
        return ay_rot(ay_deg) @ ax_rot(ax_deg) @ ay_rot(ay_deg)

# 外因性 YZY
class EulerOuterYZYState(EulerState):

    def __init__(self):
        super(EulerOuterYZYState, self).__init__()

    @EulerState.overrides(EulerState)
    def to_rot(self,
               ax_deg: float,
               ay_deg: float,
               az_deg: float) -> np.ndarray:
        
        return ay_rot(ay_deg) @ az_rot(az_deg) @ ay_rot(ay_deg)

# 外因性 ZXZ
class EulerOuterZXZState(EulerState):

    def __init__(self):
        super(EulerOuterZXZState, self).__init__()

    @EulerState.overrides(EulerState)
    def to_rot(self,
               ax_deg: float,
               ay_deg: float,
               az_deg: float) -> np.ndarray:
        
        return az_rot(az_deg) @ ax_rot(ax_deg) @ az_rot(az_deg)

# 外因性 ZYZ
class EulerOuterZYZState(EulerState):

    def __init__(self):
        super(EulerOuterZYZState, self).__init__()

    @EulerState.overrides(EulerState)
    def to_rot(self,
               ax_deg: float,
               ay_deg: float,
               az_deg: float) -> np.ndarray:
        
        return az_rot(az_deg) @ ay_rot(ay_deg) @ az_rot(az_deg)

# 外因性 XYZ
class EulerOuterXYZState(EulerState):

    def __init__(self):
        super(EulerOuterXYZState, self).__init__()

    @EulerState.overrides(EulerState)
    def to_rot(self,
               ax_deg: float,
               ay_deg: float,
               az_deg: float) -> np.ndarray:
        
        return az_rot(az_deg) @ ay_rot(ay_deg) @ ax_rot(ax_deg)

# 外因性 XZY
class EulerOuterXZYState(EulerState):

    def __init__(self):
        super(EulerOuterXZYState, self).__init__()

    @EulerState.overrides(EulerState)
    def to_rot(self,
               ax_deg: float,
               ay_deg: float,
               az_deg: float) -> np.ndarray:
        
        return ay_rot(ay_deg) @ az_rot(az_deg) @ ax_rot(ax_deg)

# 外因性 YXZ
class EulerOuterYXZState(EulerState):

    def __init__(self):
        super(EulerOuterYXZState, self).__init__()

    @EulerState.overrides(EulerState)
    def to_rot(self,
               ax_deg: float,
               ay_deg: float,
               az_deg: float) -> np.ndarray:
        
        return az_rot(az_deg) @ ax_rot(ax_deg) @ ay_rot(ay_deg)

# 外因性 YZX
class EulerOuterYZXState(EulerState):

    def __init__(self):
        super(EulerOuterYZXState, self).__init__()

    @EulerState.overrides(EulerState)
    def to_rot(self,
               ax_deg: float,
               ay_deg: float,
               az_deg: float) -> np.ndarray:
        
        return ax_rot(ax_deg) @ az_rot(az_rot) @ ay_rot(ay_deg)

# 外因性 ZXY
class EulerOuterZXYState(EulerState):

    def __init__(self):
        super(EulerOuterZXYState, self).__init__()

    @EulerState.overrides(EulerState)
    def to_rot(self,
               ax_deg: float,
               ay_deg: float,
               az_deg: float) -> np.ndarray:
        
        return ay_rot(ay_deg) @ ax_rot(ax_deg) @ az_rot(az_deg)

# 外因性 ZYX
class EulerOuterZYXState(EulerState):
    
    def __init__(self):
        super(EulerOuterZYXState, self).__init__()

    @EulerState.overrides(EulerState)
    def to_rot(self,
               ax_deg: float,
               ay_deg: float,
               az_deg: float) -> np.ndarray:
        
        return ax_rot(ax_deg) @ ay_rot(ay_deg) @ az_rot(az_deg)
    
    