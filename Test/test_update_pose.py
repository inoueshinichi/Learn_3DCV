"""回転姿勢(オイラー角-回転行列-回転ベクトル-クォータニオン)を用いた回転状態の更新チェック
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

from BasicModule.quartanion import quat, dot_quat, cat_quat, update_quat, quat_to_rot, norm_quat, normalize_quat, inv_quat
from BasicModule.rvec import rvec, rvec_to_rot, rotate_points_by_rvec, rvec_to_quat
from BasicModule.rotation import ax_rot, ay_rot, az_rot, rot_to_quat, rot_to_rvec, rot_to_euler
from BasicModule.euler import euler_to_rot, euler_to_rvec, euler_to_quat
from BasicModule.geometry_context import GeometryContext

from type_hint import *
from test_util import test_decorator


@test_decorator
def test_update_pose():
    """回転姿勢(Pose)の更新と逆更新
    [想定]
    UI表現 : オイラー角
    内部表現 : 回転行列
    更新表現 : クォータニオン
    1. オイラー角 -> 回転行列 -> クォータニオン
    2. クォータニオンで回転姿勢(Pose)を更新
    3. クォータニオン -> 回転行列 -> オイラー角
    4. グラフィックスパイプラインは, 回転行列で行う
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
    print(f"new euler theta1_deg : {state_new_theta1_deg}")
    print(f"new euler theta2_deg : {state_new_theta2_deg}")
    print(f"new euler theta3_deg : {state_new_theta3_deg}")

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

    # 差分(終点-始点)
    diff_rot = state_return_rot - state_rot
    print("diff_rot\n", diff_rot)

    # オイラー角に変換
    state_return_theta1_deg, \
    state_return_theta2_deg, \
    state_return_theta3_deg = rot_to_euler(state_return_rot, geo_ctx.euler_state)
    
    print("Euler angle")
    print(f"init_theta1_deg : {state_theta1_deg}, ret_theta1_deg : {state_return_theta1_deg}")
    print(f"init_theta2_deg : {state_theta2_deg}, ret_theta2_deg : {state_return_theta2_deg}")
    print(f"init_theta3_deg : {state_theta3_deg}, ret_theta3_deg : {state_return_theta3_deg}")


if __name__ == "__main__":
    test_update_pose() # OK