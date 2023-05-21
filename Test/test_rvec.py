"""回転姿勢(回転ベクトル)の性質に関するテスト
"""

import os
import sys

module_parent_dir = '/'.join([os.path.dirname(__file__), '..'])
sys.path.append(module_parent_dir)

import math
import inspect

import numpy as np
import scipy as sp

np.set_printoptions(suppress=True) # 指数表記禁止

from BasicModule.quartanion import quat, dot_quat, cat_quat, quat_to_rot, norm_quat, normalize_quat, inv_quat
from BasicModule.rvec import rvec, rvec_to_rot, rotate_points_by_rvec, rvec_to_quat
from BasicModule.rotation import ax_rot, ay_rot, az_rot, rot_to_quat, rot_to_rvec
from BasicModule.euler import rot_to_euler, euler_to_rot
from BasicModule.geometry_context import GeometryContext

from type_hint import *
from test_util import test_decorator


def test_move_point_with_rvec():
    """回転ベクトルによる点群の移動
    """
    n = np.array([1,0,0], dtype=np.float32) # x軸
    theta = math.radians(60)
    
    x_rvec = rvec(n, theta) # 回転ベクトル
    xR = ax_rot(theta) # 回転行列

    # 点群
    points = np.random.randn(3,5)
    print("points\n", points)

    A = xR @ points
    print("A = xR @ points\n", A)
    B = rotate_points_by_rvec(points,x_rvec)
    print("B = rotate_points_by_rvec(points,x_rvec)\n", B)

    print("diff: A - B\n", A - B)


def test_mirror_rvec():
    """回転ベクトルの系の反転
    右手系(初期) → 左手系 : Y軸反転の場合
    X,Z軸の符号を入れ変えればよい. 回転量の符号は変えない.
    ただし, 幾何的には, 時計回り正が反時計回り正になる.
    """
    n = np.array([1,1,1], dtype=np.float32)
    n /= np.linalg.norm(n)

    theta = math.radians(60)

    rvec_xyz = rvec(n, theta) # 回転ベクトル(初期:右手系)
    print("rvec_xyz=", rvec_xyz)

    rvec_mirror_xyz = rvec_xyz.copy()
    rvec_mirror_xyz[1] *= -1 # Y軸反転 (左手系)
    print("rvec_mirror_xyz=", rvec_mirror_xyz)

    # 回転行列に変換
    rot_xyz = rvec_to_rot(rvec_xyz)
    print("rot_xyz\n", rot_xyz)
    print("det(rot_xyz)=", np.linalg.det(rot_xyz))

    '''右手系と左手系の回転行列の違いは, 転置して非対角成分の符号が逆'''    
    rot_mirror_xyz = rvec_to_rot(rvec_mirror_xyz)
    print("rot_mirror_xyz\n", rot_mirror_xyz)
    print("det(rot_mirror_xyz)=", np.linalg.det(rot_mirror_xyz))
    
    '''このまま, rot_mirror_xyzをY軸反転させた姿勢にしても,
       鏡映変換のままになる(det(R)=-1)
    '''
    rot_ret_xyz = rot_mirror_xyz.copy()
    rot_ret_xyz[:,1] *= -1 # Y軸反転
    print("rot_ret_xyz\n", rot_ret_xyz)
    print("det(rot_ret_xyz)=", np.linalg.det(rot_ret_xyz))


if __name__ == "__main__":
    test_move_point_with_rvec()
    test_mirror_rvec()
