"""回転(姿勢)の性質に関するテスト
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

from BasicModule.quartanion import quat, dot_quat, update_quat, quat_to_rot, norm_quat, normalize_quat, inv_quat
from BasicModule.rvec import rvec, rvec_to_rot, rotate_points_by_rvec, rvec_to_quat
from BasicModule.rotation import ax_rot, ay_rot, az_rot, rot_to_quat, rot_to_rvec
from BasicModule.euler import rot_to_euler, euler_to_rot
from BasicModule.geometry_context import GeometryContext

from type_hint import *
from test_util import test_decorator








@test_decorator
def test_euler_rot_quat():
    """オイラー角指定による回転要素(クォータニオン)の更新と逆更新
    """

    geo_ctx = GeometryContext()

    # 状態としてのオイラー角
    state_theta1_deg = 30
    state_theta2_deg = 45
    state_theta3_deg = 60
    print("Init Euler({},{},{})".format(state_theta1_deg, state_theta2_deg, state_theta3_deg))

    # 回転行列に変換
    state_rot = euler_to_rot(theta1_deg=state_theta1_deg,
                             theta2_deg=state_theta2_deg,
                             theta3_deg=state_theta3_deg,
                             euler_state=geo_ctx.euler_state)
    print("Init Rot\n", state_rot)
    
    # 初期状態のクォータニオン
    state_quat = rot_to_quat(state_rot)
    print("norm(state_quat)=", norm_quat(state_quat))
    print("Init Quat", state_quat)

    # 更新としてのオイラー角
    add_theta1_deg = 45
    add_theta2_deg = 45
    add_theta3_deg = 45

    # 回転行列に変換
    add_rot = euler_to_rot(theta1_deg=add_theta1_deg,
                           theta2_deg=add_theta2_deg,
                           theta3_deg=add_theta3_deg,
                           euler_state=geo_ctx.euler_state)
    print("det(add_rot)=", np.linalg.det(add_rot))
    
    # クォータニオンに変換
    add_quat = rot_to_quat(add_rot)
    print("norm(add_quat)=", norm_quat(add_quat))

    # 更新
    state_new_quat = update_quat(state_quat, add_quat)
    print("New Quat", state_new_quat)

    # 新しい回転状態
    state_new_rot = quat_to_rot(state_new_quat)
    print("det(state_new_rot)=", np.linalg.det(state_new_rot))
    print("New Rot\n", state_new_rot)
    

    # オイラー角に変換
    state_new_theta1_deg, \
    state_new_theta2_deg, \
    state_new_theta3_deg = rot_to_euler(state_new_rot, geo_ctx.euler_state)
    print("New Euler({},{},{})".format(state_new_theta1_deg, state_new_theta2_deg, state_new_theta3_deg))

    # 逆クォータニオン
    inv_add_quat = inv_quat(add_quat)
    print("norm(inv_add_quat)=", norm_quat(inv_add_quat))
    
    # 逆更新
    state_return_quat = update_quat(state_new_quat, inv_add_quat)
    print("Return Quat", state_return_quat)
    
    # 差分(終点-始点)
    diff_quat = state_return_quat - state_quat
    print("diff_quat=", diff_quat)

    # 終点(回転行列)
    state_return_rot = quat_to_rot(state_return_quat)
    print("Return Rot\n", state_return_rot)

    # オイラー角に変換
    state_return_theta1_deg, \
    state_return_theta2_deg, \
    state_return_theta3_deg = rot_to_euler(state_return_rot, geo_ctx.euler_state)
    print("Return Euler({},{},{})".format(state_return_theta1_deg, state_return_theta2_deg, state_return_theta3_deg))






def test_replace_rot():
    """R=[rx,ry,rz]のうち, 2軸を入れ替えたもの.
    e.g, rR=[rz,ry,rx]は回転行列の定義を満たさない.
    鏡映変換行列になってしまう.
    回転行列 det(R)=+1
    鏡映行列 det(rR)=-1
    @note R,rRともに直行行列 : det(U)=±1
    """
    # 回転 : 内因性ZXYオイラー(θ1=30,θ2=45,θ3=60)
    ax_deg = 30
    ay_deg = 45
    az_deg = 60
    print(f"(θx,θy,θz)=({ax_deg},{ay_deg},{az_deg})")
    wRc = az_rot(math.radians(az_deg)) @ ax_rot(math.radians(ax_deg)) @ ay_rot(math.radians(ay_deg))
    sR = wRc.T # (3,3)
    print("sR\n", sR)

    # x軸とz軸を入れ替え
    mirror_rot = np.zeros((3,3), dtype=np.float32)
    mirror_rot[:,0] = sR[:,2]
    mirror_rot[:,1] = sR[:,1]
    mirror_rot[:,2] = sR[:,0]
    print("mirror_rot(x<->z)\n", mirror_rot)
    print("det(mirror_rot)=", np.linalg.det(mirror_rot)) # 回転行列の性質det(R)=+1を満たさない


def test_neg1_rot():
    """R=[rx,ry,rz]のうち, 1軸反転(鏡映変換)したもの
    """
    # 回転 : 内因性ZXYオイラー(θ1=30,θ2=45,θ3=60)
    ax_deg = 30
    ay_deg = 45
    az_deg = 60
    print(f"(θx,θy,θz)=({ax_deg},{ay_deg},{az_deg})")
    wRc = az_rot(math.radians(az_deg)) @ ax_rot(math.radians(ax_deg)) @ ay_rot(math.radians(ay_deg))
    sR = wRc.T # (3,3)
    print("sR\n", sR)

    # y軸の反転
    mirror_rot = np.zeros((3,3), dtype=np.float32)
    mirror_rot[:,0] = sR[:,0]
    mirror_rot[:,1] = -sR[:,1]
    mirror_rot[:,2] = sR[:,2]
    mirror_rot = mirror_rot
    print("mirror_rot(neg y)\n", mirror_rot)
    print("det(mirror_rot)=", np.linalg.det(mirror_rot))


def test_neg2_rot():
    """R=[rx,ry,rz]のうち2軸を反転させたもの
    """
    # 回転 : 内因性ZXYオイラー(θ1=30,θ2=45,θ3=60)
    ax_deg = 30
    ay_deg = 45
    az_deg = 60
    print(f"(θx,θy,θz)=({ax_deg},{ay_deg},{az_deg})")
    wRc = az_rot(math.radians(az_deg)) @ ax_rot(math.radians(ax_deg)) @ ay_rot(math.radians(ay_deg))
    sR = wRc.T # (3,3)
    print("sR\n", sR)

    # 2軸反転 (x軸とz軸)
    neg2_sR = np.zeros((3,3), dtype=np.float32)
    neg2_sR[:,0] = -sR[:,0]
    neg2_sR[:,1] = sR[:,1]
    neg2_sR[:,2] = -sR[:,2]
    print("neg2_sR(-x,-z)\n", neg2_sR)
    print("det(neg2_sR)=", np.linalg.det(neg2_sR))

    # y軸を方向ベクトルとした180回転の回転ベクトルを作成
    dir_vec = sR[:,1]
    dir_vec /= np.linalg.norm(dir_vec)
    dir_vec = dir_vec.reshape(3,1)
    add_rvec = rvec(dir_vec, math.radians(180))
    print("add_rvec\n", add_rvec)

    # 回転行列に変換
    add_rot = rvec_to_rot(add_rvec)
    print("add_rot\n", add_rot)
    print("det(add_rot)=", np.linalg.det(add_rot))

    # 2軸反転(x軸とz軸)のそれ以外の軸(y軸)周りに180度回転した姿勢と一致する.
    A = add_rot @ sR
    print("neg2_sR = add_rot @ sR  = \n", A)

    # 差分
    print("neg2_sR - add_rot @ sR = \n", neg2_sR - A)


def test_neg3_rot():
    """R=[rx,ry,rz]の3軸すべてを反転させたもの
    """
    # 回転 : 内因性ZXYオイラー(θ1=30,θ2=45,θ3=60)
    ax_deg = 30
    ay_deg = 45
    az_deg = 60
    print(f"(θx,θy,θz)=({ax_deg},{ay_deg},{az_deg})")
    wRc = az_rot(math.radians(az_deg)) @ ax_rot(math.radians(ax_deg)) @ ay_rot(math.radians(ay_deg))
    sR = wRc.T # (3,3)
    print("sR\n", sR)

    # 3軸反転 (x軸,y軸,z軸)
    neg3_sR = np.zeros((3,3), dtype=np.float32)
    neg3_sR[:,0] = -sR[:,0]
    neg3_sR[:,1] = -sR[:,1]
    neg3_sR[:,2] = -sR[:,2]
    print("neg3_sR\n", neg3_sR)
    print("det(neg3_sR)=", np.linalg.det(neg3_sR))


def test_check_rvec():
    """回転ベクトルの動作確認
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


if __name__ == "__main__":
    # test_rot_euler()
    test_rot_quat()
    # test_euler_rot_quat()
    # test_replace_rot()
    # test_neg1_rot()
    # test_neg2_rot()
    # test_neg3_rot()
    # test_check_rvec()