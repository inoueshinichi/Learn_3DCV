"""座標変換に関するAPI
"""


import math
import numpy as np
import scipy as sp
from typing import (
    Tuple, 
    Dict, 
    List, 
    Union,
    Callable,
    TypeVar,
    Generic,
    NoReturn
)

    """残り実装すべき関数一覧
    ・回転ベクトルの逆変換
    ・球面線形補完
    ・オイラー角の微分方程式
    ・方向余弦行列→並進・回転・拡大縮小に分解
    """

    """実装済み
    ax_rot()
    ay_rot()
    az_rot()
    eulerZXY2dcm()
    dcm2eulerZXY()
    rotvec2dcm()
    dcm2rotvec()
    dcm2quat()
    quat2dcm()
    rotvec2quat()
    quat2rotvec()
    delta_dcm()
    """

def translate(tx:float, ty:float, tz:float) -> np.ndarray:
    """並進ベクトル

    Args:
        tx (float): X
        ty (float): Y
        tz (float): Z

    Returns:
        np.ndarray: 並進ベクトル(3x1)
    """
    trans = np.array([tx, ty, tz], dtype=np.float32)
    return trans


def ax_rot(ax_deg:float) -> np.ndarray:
    """X軸中心の回転ベクトル

    Args:
        ax_deg (float): 角度[deg]

    Returns:
        np.ndarray: 回転行列(3x3)
    """
    ax_rad = math.radians(ax_deg)
    ax_rot = np.array([
        1, 0, 0,
        0, math.cos(ax_rad), -math.sin(ax_rad),
        0, math.sin(ax_rad), math.cos(ax_rad)
    ], dtype=np.float32).reshape(3,3)
    return ax_rot


def ay_rot(ay_deg:float) -> np.ndarray:
    """Y軸中心の回転ベクトル

    Args:
        ay_deg (float): 角度[deg]

    Returns:
        np.ndarray: 回転行列(3x3)
    """
    ay_rad = math.radians(ay_deg)
    ay_rot = np.array([
        math.cos(ay_rad), 0, math.sin(ay_rad),
        0, 1, 0,
        -math.sin(ay_rad), 0, math.cos(ay_rad)
    ], dtype=np.float32).reshape(3,3)
    return ay_rot


def az_rot(az_deg:float) -> np.ndarray:
    """Z軸中心の回転行列

    Args:
        az_deg (float): 角度[deg]

    Returns:
        np.ndarray: 回転行列(3x3)
    """
    az_rad = math.radians(az_deg)
    az_rot = np.array([
        math.cos(az_rad), -math.sin(az_rad), 0,
        math.sin(az_rad), math.cos(az_rad), 0,
        0, 0, 1
    ], dtype=np.float32).reshape(3,3)
    return az_rot


def eulerZXY2dcm(ax_deg:float, ay_deg:float, az_deg:float) -> np.ndarray:
    """ZXYオイラー角から方向余弦行列を計算

    Args:
        ax_deg (float): 角度[deg]
        ay_deg (float): 角度[deg]
        az_deg (float): 角度[deg]

    Returns:
        [type]: 方向余弦行列(3x3)
    """
    dcm = az_rot(az_deg) @ ax_rot(ax_deg) @ ay_rot(ay_deg)
    return dcm


def dcm2eulerZXY(dcm: np.ndarray, gimbal_threshold=0.001) -> Tuple[float, float, float]:
    """方向余弦行列からオイラー角を計算
    X軸回りの回転が±π/2になるとジンバルロックが発生するので、
    sin(x)の値をチェックして場合分け。

    Arguments:
        rot {np.ndarray} -- (3x3)

    Returns:
        np.ndarray -- ax_deg, ay_deg, az_deg
    """
    r11, r12, r13 = dcm[0,0], dcm[0,1], dcm[0,2]
    r21, r22, r23 = dcm[1,0], dcm[1,1], dcm[1,2]
    r31, r32, r33 = dcm[2,0], dcm[2,1], dcm[2,2]

    ax_deg, ay_deg, az_deg = 0.0, 0.0, 0.0

    # r32=sin(x)の値をチェックして場合分け
    if math.abs(r32 - 1.0) < gimbal_threshold:
        # r32 == 1のとき(ジンバルロック発生)
        az_deg = math.atan2(r21, r11)
        ax_deg = math.degrees(math.pi/2)
        ay_deg = 0 # Y'軸周りの回転を0度に固定

    elif math.abs(r32 + 1.0) < gimbal_threshold:
        # r32 == -1のとき(ジンバルロック発生)
        az_deg = math.atan2(r21, r11)
        ax_deg = math.degrees(-math.pi/2)
        ay_deg = 0 # Y'軸周りの回転を0度に固定
        
    else:
        # -1 < r32 < 1のとき
        az_deg = math.degrees(math.atan2(-r12, r22))
        ax_deg = math.degrees(math.asin(r32))
        ay_deg = math.degrees(math.atan2(-r31, r33))
        
    return ax_deg, ay_deg, az_deg


def rotvec2dcm(rotvec: np.ndarray) -> np.ndarray:
    """回転ベクトルから方向余弦行列を計算
    ロドリゲスの公式
    Arguments:
        rotvec {np.ndarray} -- (3x1)

    Returns:
        np.ndarray -- (3x3)
    """
    rad_ang = np.linalg.norm(rotvec) # 回転角[-pi +pi]
    u_vec = rotvec / rad_ang    # 方向ベクトル(単位ベクトル)

    cos_ = math.cos(rad_ang)
    sin_ = math.sin(rad_ang)
    I = np.ones((3, 3), dtype=np.float32)
    uu = u_vec @ u_vec.T
    J = np.array([
        [0., -rotvec[2], rotvec[1]],
        [rotvec[2], 0.,  -rotvec[0]],
        [-rotvec[1], rotvec[0], 0.]
    ], dtype=np.float32)

    dcm = cos_ * I + (1 - cos_) * uu + sin_ * J

    return dcm


def dcm2rotvec(dcm: np.ndarray) -> Tuple[float, np.ndarray]:
    """方向余弦行列から回転ベクトルを算出

    Arguments:
        dcm {np.ndarray} -- 方向余弦行列(3x3)

    Returns:
        np.ndarray -- 回転角[deg], 回転ベクトル(3x1)
    """
    # 方向余弦行列から直接回転ベクトルの回転角を求める
    A = 0.5 * (dcm - dcm.T)
    rho = np.array([A[2, 1], A[0, 2], A[2, 0]], dtype=np.float32)
    s = np.linang.norm(rho)                           # sinθ
    c = 0.5 * (dcm[0, 0] + dcm[1, 1] + dcm[2, 2] - 1) # cosθ
    
    # 方向ベクトルと回転角
    rot_vec = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    theta = 0.0
    
    # s(sin)とc(cos)の値で場合分け
    if s == 0 and c == 1:
        # ゼロベクトル
        rot_vec = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        theta = 0.0
        
    elif s == 0 and c == -1:
        # theta = 180deg
        B = dcm + np.eye(3, 3) # R + I
        b1, b2, b3 = np.hsplit(B)
        col_vecs = [np.squeeze(b1), np.squeeze(b2), np.squeeze(b3)]
        
        # B(R+I)の非ゼロの列ベクトルを取得
        non_zero_vec = None
        non_zero_norm = 0.0
        for vec in col_vecs:
            norm = np.linalg.norm(vec)
            if norm != 0.0:
                non_zero_norm = norm
                non_zero_vec = vec
                break
                
        # 回転角(180deg)
        theta = math.pi
        # 方向ベクトル
        u_vec = non_zero_vec / non_zero_norm
        # 回転ベクトル
        rot_vec = math.pi * u_vec
        
        # 符号の反転チェック
        if np.linalg.norm(rot_vec) == math.pi:
           r1, r2, r3 = rot_vec
           if (r1 == 0 and r2 == 0 and r3 < 0) or
              (r1 == 0 and r2 < 0) or
              (r1 < 0):
                  rot_vec *= -1             
    
     else: # sinθ != 0
         # 方向ベクトル
         u_vec = rho / s
         # 回転角
         theta = math.arctan2(s, c)
         # 回転ベクトル
         rot_vec = theta * u_vec
         
    # 出力
    return math.degrees(theta), rot_vec


def dcm2quat(dcm: np.ndarray) -> np.ndarray:
    """方向余弦行列からクォータニオンを計算

    Arguments:
        dcm {np.ndarray} -- 方向余弦行列(3x3)

    Returns:
        np.ndarray -- クォータニオン(4x1)
    """

    r11, r12, r13 = dcm[0,0], dcm[0,1], dcm[0,2]
    r21, r22, r23 = dcm[1,0], dcm[1,1], dcm[1,2]
    r31, r32, r33 = dcm[2,0], dcm[2,1], dcm[2,2]

    # dcmの対角成分からw(実部)を算出
    tr = r11 + r22 + r33
    q1 = 0.5*np.sqrt(tr + 1)

    # 虚部(x,y,z)を算出, k2,k3,k4の最大成分を検索して...
    k2 = 0.5*np.sqrt(r11 - r22 - r33 + 1)
    k3 = 0.5*np.sqrt(-r11 + r22 - r33 + 1)
    k4 = 0.5*np.sqrt(-r11 - r22 + r33 + 1)

    if k2 >= k3 and k2 >= k4:
        q2 = k2*np.sign(r32 - r23)
        q3 = k3*np.sign(q2*(r21 + r12))
        q4 = k4*np.sign(q2*(r31 + r13))
    elif k3 >= k2 and k3 >= k4:
        q3 = k3*np.sign(r13 - r31)
        q2 = k2*np.sign(q3*(r12 + r21))
        q4 = k4*np.sign(q3*(r32 + r23))
    else:
        q4 = k4*np.sign(r21 - r12)
        q2 = k2*np.sign(q4*(r13 + r31))
        q3 = k3*np.sign(q4*(r23 + r32))

    return np.array([q1, q2, q3, q4], dtype=np.float32)


def quat2dcm(quat: np.ndarray) -> np.ndarray:
    """クォータニオンから方向余弦行列を計算

    Arguments:
        quat {np.ndarray} -- クォータニオン(4x1)

    Returns:
        np.ndarray -- 方向余弦行列(3x3)
    """

    r11 = 1 - 2 * (quat[2] ** 2 + quat[3] ** 2)
    r12 = 2 * (quat[1] * quat[2] - quat[0] * quat[3])
    r13 = 2 * (quat[1] * quat[3] + quat[0] * quat[2])
    r21 = 2 * (quat[1] * quat[2] + quat[0] * quat[3])
    r22 = 1 - 2 * (quat[1] ** 2 + quat[3] ** 2)
    r23 = 2 * (quat[2] * quat[3] - quat[0] * quat[1])
    r31 = 2 * (quat[1] * quat[3] - quat[0] * quat[2])
    r32 = 2 * (quat[2] * quat[3] + quat[0] * quat[1])
    r33 = 1 - 2 * (quat[1] ** 2 + quat[2] ** 2)

    dcm = np.array([
        [r11,r12,r13],
        [r21,r22,r23],
        [r31,r32,r33]
    ], dtype=np.float32)

    return dcm


def delta_dcm(dcm:np.ndaray, omega:np.ndarray) -> np.ndarray:
    """方向余弦行列の微分

    Args:
        dcm (np.ndaray): (3,3)
        omega (np.ndarray): (3,)

    Returns:
        np.ndarray: (3,3)
    """
    assert omgea.ndim == 1, "Dimension of omega must be 1 dim."
    assert omega.shape[0] == 3, "omega must be (3,)."

    # 歪対象行列
    oc = np.zeros((3,3), dtype=np.float32)
    oc[0, 1] = omega[2]
    oc[1, 0] = -omega[2]
    oc[0, 2] = -omega[1]
    oc[2, 0] = omega[1]
    oc[1, 2] = omega[0]
    oc[2, 1] = -omega[0]

    return oc @ dcm


def rotvec2quat(u_vec: np.array, ang_deg: float) -> np.array:
    """回転量と方向ベクトルからクォータニオンを算出

    Arguments:
        angle_deg {float} -- 回転角度
        u_vec {np.array} -- e.g X軸回り(1,0,0), Y軸回り(0,1,0), Z軸回り(0,0,1)

    Returns:
        np.array -- 回転クォータニオン (pw, px, py, pz) (4x1)
    """
    norm = np.linalg.norm(u_vec)
    if norm != 1:
        u_vec /= norm

    ang_rad = math.radians(ang_deg)
    sine = math.sin(ang_rad/2)
    o1 = math.cos(ang_rad/2)   # 実部
    o2 = u_vec[0] * sine       # x虚部
    o3 = u_vec[1] * sine       # y虚部
    o4 = u_vec[2] * sine       # z虚部

    return np.array([o1, o2, o3, o4], dtype=np.float32) 


def quat2rotvec(quat:np.ndarray) -> Tuple[flaot, np.ndarray]:
    """クォータニオンから回転ベクトルを計算

    Args:
        quat (np.ndarray): クォータニオン(4x1)

    Returns:
        Tuple[flaot, np.ndarray]: 回転角[deg], 回転ベクトル(3x1)
    """

    # クォータニオンから方向余弦行列
    dcm = quat2dcm(quat)

    # 方向余弦行列から回転ベクトル
    angle_rad, rot_vec = dcm2rotvec(dcm)

    return math.degrees(angle_rad), rot_vec







def quat_rotate_one_point(point: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """クォータニオンによる3次点の回転
    ただし、3次元空間中の任意の1点を回転させる関数
    複数点を回転させる場合は、方向余弦行列を作ってから計算する
    r' = q*r*conj_q

    Arguments:
        point {np.ndarray} -- (3x1)
        quat  {np.ndarray} -- (4x1)

    Returns:
        np.ndarray -- (3x1)
    """
    assert np.linalg.norm(quat) == 1, "quat must be nomalized."
    conj_q = conj_quat(quat) # 共役クォータニオン

    rotated_r = multiply_quat(conj_q, multiply_quat(np.r_[0, point], quat))
    return rotated_r[1:]
    


def lookAt(c, t, u = np.array([0,0,1])):
    # ワールド座標系からカメラ座標系までのフレーム行列を作成
    # カメラの上方向はX軸を決めるためだけなので、ワールド座標系の上方向で構わない
    # u=[0,0,1]の場合はX軸とY軸を中心とした回転が -90 < angle < 90にしかできない。
    # ただし、座標系の真上から見下ろすカメラ姿勢の場合は、Z軸(orX軸)方向がカメラの上方向となるようにするために、
    # 右手座標系の場合は, u=[0,-1,0], 左手座標系の場合は、u=[0,1,0]にする (X軸をカメラの上方向とする場合はu=[-1,0,0]or[1,0,0])
    # 上記の設定にした場合、 X軸とY軸回りに -90 < angle < 90の制限がかかる
    # もし90を超過したら投影される画像が一瞬で上下反転する

    # 平行移動(座標変換)
    M = np.eye(4, dtype=np.float64)
    M[:-1, -1] = -c

    # カメラの視線方向(Z軸)を作成
    z = (c - t) / np.linalg.norm(c - t)

    # カメラのX軸方向を作成
    x = np.cross(u, z)
    x /= np.linalg.norm(x)

    # Z方向とX方向の外積からY軸方向を作成
    y = np.cross(z, x)
    y /= np.linalg.norm(y)

    # 回転行列
    R = np.array([
        [x[0], x[1], x[2], 0.],
        [y[0], y[1], y[2], 0.],
        [z[0], z[1], z[2], 0.],
        [0., 0., 0., 1.]
    ], dtype=np.float64)

    w2c_transform = R @ M

    print("R", R)
    print("M", M)
    print("R@M", R @ M)

    return w2c_transform
    
        
        
            
        

