"""姿勢表現の変換に関するテスト
1. 回転行列
2. オイラー角
3. クォータニオン
4. 回転ベクトル

○ オイラー角 <-> 回転行列 (誤差あり)
○ オイラー角 <-> 回転ベクトル (誤差あり)
○ オイラー角 <-> クォータニオン (誤差あり)
○ 回転行列 <-> 回転ベクトル (誤差あり)
○ 回転行列 <-> クォータニオン (誤差あり)
○ 回転ベクトル <-> クォータニオン (誤差あり)

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

from BasicModule.quartanion import norm_quat, quat_to_rot, quat_to_rvec, quat_to_euler
from BasicModule.rvec import rvec, rvec_to_rot, rvec_to_quat, rvec_to_euler
from BasicModule.rotation import rot_to_quat, rot_to_rvec, rot_to_euler
from BasicModule.euler import euler_to_rot, euler_to_rvec, euler_to_quat
from BasicModule.geometry_context import GeometryContext

from type_hint import *
from test_util import test_decorator


@test_decorator
def test_euler_to_rot_and_reverse():
    """オイラー角 → 回転行列
       逆変換
    """
    geo_ctx = GeometryContext() # Inner ZXY

    theta1_deg = 30
    theta2_deg = 45
    theta3_deg = 60

    # 回転行列に変換
    rot = euler_to_rot(theta1_deg=theta1_deg,
                       theta2_deg=theta2_deg,
                       theta3_deg=theta3_deg,
                       euler_state=geo_ctx.euler_state)
    
    print("rot\n", rot)

    # オイラー角に逆変換
    euler1_deg, euler2_deg, euler3_deg = rot_to_euler(rot, 
                                                      geo_ctx.euler_state) # 誤差が出る

    print(f"theta1_deg: {theta1_deg}, euler1_deg: {euler1_deg}")
    print(f"theta2_deg: {theta2_deg}, euler2_deg: {euler2_deg}")
    print(f"theta3_deg: {theta3_deg}, euler3_deg: {euler3_deg}")


@test_decorator
def test_rot_to_quat_and_reverse():
    """(オイラー角->)回転行列 → クォータニオン
       逆変換
    """
    geo_ctx = GeometryContext()

    # オイラー角
    init_theta1_deg = 30
    init_theta2_deg = 45
    init_theta3_deg = 60

    # 回転行列に変換
    init_rot = euler_to_rot(theta1_deg=init_theta1_deg,
                            theta2_deg=init_theta2_deg,
                            theta3_deg=init_theta3_deg,
                            euler_state=geo_ctx.euler_state)
    print("init_rot\n", init_rot)

    # クォータニオンに変換
    init_quat = rot_to_quat(init_rot)
    print("init_quat=", init_quat)
    print("norm(init_qual)=", norm_quat(init_quat))

    # 回転行列に戻す
    ret_rot = quat_to_rot(init_quat)
    print("ret_rot\n", ret_rot)
    print("det(ret_rot)=", np.linalg.det(ret_rot))

    # オイラー角に戻す
    ret_theta1_deg, ret_theta2_deg, ret_theta3_deg = rot_to_euler(ret_rot, 
                                                                  geo_ctx.euler_state) # 誤差が出る
    print("auler angle")
    print(f"init_theta1_deg: {init_theta1_deg}, ret_theta1_deg: {ret_theta1_deg}")
    print(f"init_theta2_deg: {init_theta2_deg}, ret_theta2_deg: {ret_theta2_deg}")
    print(f"init_theta3_deg: {init_theta3_deg}, ret_theta3_deg: {ret_theta3_deg}")


@test_decorator
def test_rvec_to_rot_and_reverse():
    """回転ベクトル → 回転行列
       逆変換
    """

    # 方向ベクトル
    nx = np.array([1,0,0], dtype=np.float32) # x軸
    ny = np.array([0,1,0], dtype=np.float32) # y軸
    nz = np.array([0,0,1], dtype=np.float32) # z軸
    nxyz = np.array([1,1,1], dtype=np.float32)
    nxyz /= np.linalg.norm(nxyz) # (1,1,1)方向
    print("norm(nxyz)=", np.linalg.norm(nxyz))

    # 回転ベクトル
    theta_deg = 180 #60
    theta_rad = math.radians(theta_deg)
    # init_rvec_x = rvec(nx, theta_rad)
    # init_rvec_y = rvec(ny, theta_rad)
    # init_rvec_z = rvec(nz, theta_rad)
    init_rvec_xyz = rvec(nxyz, theta_rad)
    # print("init_rvec_x=", init_rvec_x)
    # print("init_rvec_y=", init_rvec_y)
    # print("init_rvec_z=", init_rvec_z)
    print("init_rvec_xyz=", init_rvec_xyz)
    

    # 回転行列に変換
    # init_rot_x = rvec_to_rot(init_rvec_x)
    # init_rot_y = rvec_to_rot(init_rvec_y)
    # init_rot_z = rvec_to_rot(init_rvec_z)
    init_rot_xyz = rvec_to_rot(init_rvec_xyz)

    # print("init_rot_x\n", init_rot_x)
    # print("init_rot_y\n", init_rot_y)
    # print("init_rot_z\n", init_rot_z)
    print("init_rot_xyz\n", init_rot_xyz)


    # 回転ベクトルに戻す
    # ret_rvec_x = rot_to_rvec(init_rot_x)
    # ret_rvec_y = rot_to_rvec(init_rot_y)
    # ret_rvec_z = rot_to_rvec(init_rot_z)
    ret_rvec_xyz = rot_to_rvec(init_rot_xyz)

    # print(f"init_rvec_x : {init_rvec_x}, ret_rvec_x : {ret_rvec_x}")
    # print(f"init_rvec_y : {init_rvec_y}, ret_rvec_y : {ret_rvec_y}")
    # print(f"init_rvec_z : {init_rvec_z}, ret_rvec_z : {ret_rvec_z}")
    print(f"init_rvec_xyx : {init_rvec_xyz}, ret_rvec_xyz : {ret_rvec_xyz}")


@test_decorator
def test_rvec_to_quat_and_reverse():
    """回転ベクトル -> クォータニオン
       逆変換
    """
    # 方向ベクトル
    nx = np.array([1,0,0], dtype=np.float32) # x軸
    ny = np.array([0,1,0], dtype=np.float32) # y軸
    nz = np.array([0,0,1], dtype=np.float32) # z軸
    nxyz = np.array([1,1,1], dtype=np.float32)
    nxyz /= np.linalg.norm(nxyz) # (1,1,1)方向
    print("norm(nxyz)=", np.linalg.norm(nxyz))

    # 回転ベクトル
    theta_deg = 60
    theta_rad = math.radians(theta_deg)
    # init_rvec_x = rvec(nx, theta_rad)
    # init_rvec_y = rvec(ny, theta_rad)
    # init_rvec_z = rvec(nz, theta_rad)
    init_rvec_xyz = rvec(nxyz, theta_rad)
    # print("init_rvec_x=", init_rvec_x)
    # print("init_rvec_y=", init_rvec_y)
    # print("init_rvec_z=", init_rvec_z)
    print("init_rvec_xyz=", init_rvec_xyz)


    # クォータニオンに変換
    # init_quat_x = rvec_to_quat(init_rvec_x)
    # init_quat_y = rvec_to_quat(init_rvec_y)
    # init_quat_z = rvec_to_quat(init_rvec_z)
    init_quat_xyz = rvec_to_quat(init_rvec_xyz)

    # print("init_quat_x=", init_quat_x)
    # print("init_quat_y=", init_quat_y)
    # print("init_quat_z=", init_quat_z)
    print("init_quat_xyz=", init_quat_xyz)

    # 回転ベクトルに戻す
    # ret_rvec_x = quat_to_rvec(init_quat_x)
    # ret_rvec_y = quat_to_rvec(init_quat_y)
    # ret_rvec_z = quat_to_rvec(init_quat_z)
    ret_rvec_xyz = quat_to_rvec(init_quat_xyz)


    # print(f"init_rvec_x : {init_rvec_x}, ret_rvec_x : {ret_rvec_x}")
    # print(f"init_rvec_y : {init_rvec_y}, ret_rvec_y : {ret_rvec_y}")
    # print(f"init_rvec_z : {init_rvec_z}, ret_rvec_z : {ret_rvec_z}")
    print(f"init_rvec_xyx : {init_rvec_xyz}, ret_rvec_xyz : {ret_rvec_xyz}")

@test_decorator
def test_euler_to_rvec():
    """オイラー角 -> 回転ベクトル
       逆変換
    """
    geo_ctx = GeometryContext()

    # オイラー角
    init_theta1_deg = 30
    init_theta2_deg = 45
    init_theta3_deg = 60

    # 回転ベクトルに変換
    init_rvec = euler_to_rvec(theta1_deg=init_theta1_deg,
                              theta2_deg=init_theta2_deg,
                              theta3_deg=init_theta3_deg,
                              euler_state=geo_ctx.euler_state)
    print("init_rvec\n", init_rvec)

    # オイラー角に戻す
    ret_theta1_deg, ret_theta2_deg, ret_theta3_deg = rvec_to_euler(init_rvec, geo_ctx.euler_state)

    print("auler angle")
    print(f"init_theta1_deg: {init_theta1_deg}, ret_theta1_deg: {ret_theta1_deg}")
    print(f"init_theta2_deg: {init_theta2_deg}, ret_theta2_deg: {ret_theta2_deg}")
    print(f"init_theta3_deg: {init_theta3_deg}, ret_theta3_deg: {ret_theta3_deg}")

@test_decorator
def test_euler_to_quat():
    """オイラー角 -> クォータニオン
       逆変換
    """

    geo_ctx = GeometryContext()

    # オイラー角
    init_theta1_deg = 30
    init_theta2_deg = 45
    init_theta3_deg = 60

    # クォータニオンに変換
    init_quat = euler_to_quat(theta1_deg=init_theta1_deg,
                              theta2_deg=init_theta2_deg,
                              theta3_deg=init_theta3_deg,
                              euler_state=geo_ctx.euler_state)

    print("init_quat=", init_quat)

    # オイラー角に戻す
    ret_theta1_deg, ret_theta2_deg, ret_theta3_deg = quat_to_euler(init_quat, geo_ctx.euler_state)

    print("auler angle")
    print(f"init_theta1_deg: {init_theta1_deg}, ret_theta1_deg: {ret_theta1_deg}")
    print(f"init_theta2_deg: {init_theta2_deg}, ret_theta2_deg: {ret_theta2_deg}")
    print(f"init_theta3_deg: {init_theta3_deg}, ret_theta3_deg: {ret_theta3_deg}")



if __name__ == "__main__":
    # test_euler_to_rot_and_reverse() # OK
    # test_rot_to_quat_and_reverse() # OK
    # test_rvec_to_rot_and_reverse() # OK
    # test_rvec_to_quat_and_reverse() # OK
    # test_euler_to_rvec() # OK
    test_euler_to_quat() # OK
