"""回転行列(姿勢)の性質に関するテスト
"""

import math
import numpy as np
import scipy as sp

np.set_printoptions(suppress=True) # 指数表記禁止

from rvec import make_rvec, rvec_to_rot
from rotation import ax_rot, ay_rot, az_rot
from euler import rot_to_euler
from geometry_context import GeometryContext


def test_replace_rot():
    """(rx,ry,rz)のうち, 2軸を入れ替えたもの
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
    """(rx,ry,rz)のうち, 1軸反転(鏡映変換)したもの
    """
    # 回転 : 内因性ZXYオイラー(θ1=30,θ2=45,θ3=60)
    ax_deg = 30
    ay_deg = 45
    az_deg = 60
    print(f"(θx,θy,θz)=({ax_deg},{ay_deg},{az_deg})")
    wRc = az_rot(math.radians(az_deg)) @ ax_rot(math.radians(ax_deg)) @ ay_rot(math.radians(ay_deg))
    sR = wRc.T # (3,3)
    print("sR\n", sR)



def test_neg2_rot():
    # 回転 : 内因性ZXYオイラー(θ1=30,θ2=45,θ3=60)
    ax_deg = 30
    ay_deg = 45
    az_deg = 60
    print(f"(θx,θy,θz)=({ax_deg},{ay_deg},{az_deg})")
    wRc = az_rot(math.radians(az_deg)) @ ax_rot(math.radians(ax_deg)) @ ay_rot(math.radians(ay_deg))
    sR = wRc.T # (3,3)
    print("sR\n", sR)

    

    # # y軸の反転
    # mirror_rot_2 = np.zeros((3,3), dtype=np.float32)
    # mirror_rot_2[:,0] = sR[:,0]
    # mirror_rot_2[:,1] = -sR[:,1]
    # mirror_rot_2[:,2] = sR[:,2]
    # mirror_rot_2 = mirror_rot_2
    # print("mirror_rot_2(neg y)\n", mirror_rot_2)
    # print("det(mirror_rot_2)=", np.linalg.det(mirror_rot_2))

    # 2軸反転 (x軸とz軸)
    neg2_sR = np.zeros((3,3), dtype=np.float32)
    neg2_sR[:,0] = -sR[:,0]
    neg2_sR[:,1] = sR[:,1]
    neg2_sR[:,2] = -sR[:,2]
    print("neg2_sR\n", neg2_sR)
    print("det(neg2_sR)=", np.linalg.det(neg2_sR))

    # y軸を方向ベクトルとした180回転の回転ベクトルを作成
    dir_vec = sR[:,1]
    dir_vec /= np.linalg.norm(dir_vec)
    dir_vec = dir_vec.reshape(3,1)
    add_rvec = make_rvec(dir_vec, math.radians(180))
    print("add_rvec\n", add_rvec)

    # 回転行列に変換
    add_rot = rvec_to_rot(add_rvec)
    print("add_rot\n", add_rot)
    print("det(add_rot)=", np.linalg.det(add_rot))

    # 2軸反転(x軸とz軸)のそれ以外の軸(y軸)周りに180度回転した姿勢と一致する.
    print("neg2_sR = add_rot @ sR  = \n", add_rot @ sR)