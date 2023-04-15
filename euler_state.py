"""オイラー角
Default:
    内因性, ZXY

24パターン:
    内因性 or 外因性
    古典オイラー角
        XYX, XZX, YXY, YZY, ZXZ, ZYZ
    ブライアント角
        XYZ, XZY, YXZ, YZX, ZXY, ZYX

行列は列優先表現とする. numpyのデータ配列は行優先.

内因性と外因性の関係:
    内因性ZXY = 外因性YXZ : (R(Z)@R(X)@R(Y))
"""

'''
内因性 XYX
内因性 XZX
内因性 YXY
内因性 YZY
内因性 ZXZ
内因性 ZYZ
内因性 XYZ
内因性 XZY
内因性 YXZ
内因性 YZX 完
内因性 ZXY 完
内因性 ZXY 完
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

    def __check_gimbal_lock(self, elem_rot: float) -> int:
        """ジンバルロックのチェック

        Args:
            elem_rot (float): ジンバルロック確認に使用する回転行列の1要素

        Returns:
            int: 0 -> No ginbal lock
                 +1/-1 -> ginbal lock
                 +1 -> PI/2 (elem_rot == 1), 
                 -1 -> -PI/2 (elem_rot == -1)
        """
        if math.abs(elem_rot - 1.0) < self.gimbal_eps:
            return 1
        elif math.abs(elem_rot + 1.0) < self.gimbal_eps:
            return -1
        else:
            return 0


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


# 内因性 XYX
class EulerInnerXYXState(EulerState):

    def __init__(self):
        super(EulerInnerXYXState, self).__init__()

    @EulerState.overrides(EulerState)
    def to_rot(self,
               ax_deg: float,
               ay_deg: float,
               az_deg: float) -> np.ndarray:
        
        return ax_rot(ax_deg) @ ay_rot(ay_deg) @ ax_rot(ax_deg)

# 内因性 XZX
class EulerInnerXZXState(EulerState):

    def __init__(self):
        super(EulerInnerXZXState, self).__init__()

    @EulerState.overrides(EulerState)
    def to_rot(self,
               ax_deg: float,
               ay_deg: float,
               az_deg: float) -> np.ndarray:
        
        return ax_rot(ax_deg) @ az_rot(az_deg) @ ax_rot(ax_deg)   

# 内因性 YXY
class EulerInnerYXYState(EulerState):

    def __init__(self):
        super(EulerInnerYXYState, self).__init__()

    @EulerState.overrides(EulerState)
    def to_rot(self,
               ax_deg: float,
               ay_deg: float,
               az_deg: float) -> np.ndarray:
        
        return ay_rot(ay_deg) @ ax_rot(ax_deg) @ ay_rot(ay_deg)

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

            Z=+pi/2,-pi/2のとき,
            r12 = -cos(Y+X), r13 = sin(Y+X) where X=0

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
            pass
            ay_deg = math.degrees(math.atan2(r13, -r12))
            az_deg = math.degrees(math.pi/2)
            ax_deg = 0.0 # Z軸のジンバルロックに従属
        elif math.abs(r21 + 1.0) < self.gimbal_eps:
            # r21 == -1, Z=-pi/2
            ay_deg = math.degrees(math.atan2(r13, -r12))
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

            X=+pi/2,-pi/2のとき,
            r11 = cos(Z+Y), r21 = sin(Z+Y) where Y=0

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
            az_deg = math.atan2(r21, r11)
            ax_deg = math.degrees(-math.pi/2)
            ay_deg = 0.0 # X軸のジンバルロックに従属
        else: 
            # -1 < r32 < +1
            az_deg = math.degrees(math.atan2(-r12, r22))
            ax_deg = math.degrees(math.asin(r32))
            ay_deg = math.degrees(math.atan2(-31, r33))

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

            Y=+pi/2,-pi/2のとき,
            r13=cos(Z-X), r23=sin(Z-X) where X=0

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
    
    