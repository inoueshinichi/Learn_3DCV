
import math
import numpy as np
import scipy as sp

np.set_printoptions(suppress=True) # 指数表記禁止

from rvec import make_rvec, rvec_to_rot
from rotation import ax_rot, ay_rot, az_rot
from euler import rot_to_euler
from geometry_context import GeometryContext


    




# 座標系の軸反転なし(鏡映状態でない)
def test1_decomp_camera():

    # Calibration行列
    sK = np.array([
        [1000,0,500],
        [0,1000,300],
        [0,0,1]], dtype=np.float32) # (3,3)
    print("sK\n", sK)
    
    # 回転 : 内因性ZXYオイラー(θ1=30,θ2=30,θ3=30)
    ax_deg = 30
    ay_deg = 30
    az_deg = 30
    print(f"(θx,θy,θz)=({ax_deg},{ay_deg},{az_deg})")
    wRc = az_rot(math.radians(az_deg)) @ ax_rot(math.radians(ax_deg)) @ ay_rot(math.radians(ay_deg))
    sR = wRc.T # (3,3)
    print("sR\n", sR)

    # 並進
    st = np.array([[50],[40],[30]], dtype=np.float32)
    print("st\n", st)

    # View行列
    sV = np.hstack((sR,st))
    print("sV\n", sV)

    # Camera(Projection)行列
    sP = sK @ sV
    print("sP\n", sP)

    # RQ分解
    dK, dR = sp.linalg.rq(sP[:3,:3])
    print("dK\n", dK)
    print("dR\n", dR)
    print("det(dR)=", np.linalg.det(dR))

    T = np.diag(np.sign(np.diag(dK)))
    print("T\n", T)
    print("det(T)=", np.linalg.det(T))

    if np.linalg.det(T) < 0:
        T[1,1] *= -1
        print("mod T\n", T)
        print("det(mod T)=", np.linalg.det(T))

    K = dK @ T
    K /= K[2,2]
    R = T @ dR
    print("K\n", K)
    print("R\n", R)
    print("det(R)=", np.linalg.det(R))

# 座標系の軸反転なし(鏡映状態でない)
def test2_decomp_camera():

    # Calibration行列
    sK = np.array([
        [1000,0,500],
        [0,1000,300],
        [0,0,1]], dtype=np.float32) # (3,3)
    print("sK\n", sK)
    
    # 回転 : 内因性ZXYオイラー(θ1=60,θ2=45,θ3=30)
    ax_deg = 60
    ay_deg = 45
    az_deg = 30
    print(f"(θx,θy,θz)=({ax_deg},{ay_deg},{az_deg})")
    wRc = az_rot(math.radians(az_deg)) @ ax_rot(math.radians(ax_deg)) @ ay_rot(math.radians(ay_deg))
    sR = wRc.T # (3,3)
    print("sR\n", sR)

    # 並進
    st = np.array([[50],[40],[30]], dtype=np.float32)
    print("st\n", st)

    # View行列
    sV = np.hstack((sR,st))
    print("sV\n", sV)

    # Camera(Projection)行列
    sP = sK @ sV
    print("sP\n", sP)

    # RQ分解
    dK, dR = sp.linalg.rq(sP[:3,:3])
    print("dK\n", dK)
    print("dR\n", dR)
    print("det(dR)=", np.linalg.det(dR))

    T = np.diag(np.sign(np.diag(dK)))
    print("T\n", T)
    print("det(T)=", np.linalg.det(T))

    if np.linalg.det(T) < 0:
        T[1,1] *= -1
        print("mod T\n", T)
        print("det(mod T)=", np.linalg.det(T))

    K = dK @ T
    K /= K[2,2]
    R = T @ dR
    print("K\n", K)
    print("R\n", R)
    print("det(R)=", np.linalg.det(R))


# 座標系の軸反転なし(鏡映状態でない)
def test3_decomp_camera():

    # Calibration行列
    sK = np.array([
        [1000,0,500],
        [0,1000,300],
        [0,0,1]], dtype=np.float32) # (3,3)
    print("sK\n", sK)
    
    # 回転 : 内因性ZXYオイラー(θ1=30,θ2=30,θ3=30)
    ax_deg = 30
    ay_deg = 30
    az_deg = 30
    print(f"(θx,θy,θz)=({ax_deg},{ay_deg},{az_deg})")
    wRc = az_rot(math.radians(az_deg)) @ ax_rot(math.radians(ax_deg)) @ ay_rot(math.radians(ay_deg))
    sR = wRc.T # (3,3)
    print("sR\n", sR)

    # 並進
    st = np.array([[100],[0],[100]], dtype=np.float32)
    print("st\n", st)

    # View行列
    sV = np.hstack((sR,st))
    print("sV\n", sV)

    # Camera(Projection)行列
    sP = sK @ sV
    print("sP\n", sP)

    # RQ分解
    dK, dR = sp.linalg.rq(sP[:3,:3])
    print("dK\n", dK)
    print("dR\n", dR)
    print("det(dR)=", np.linalg.det(dR))

    T = np.diag(np.sign(np.diag(dK)))
    print("T\n", T)
    print("det(T)=", np.linalg.det(T))

    if np.linalg.det(T) < 0:
        T[1,1] *= -1
        print("mod T\n", T)
        print("det(mod T)=", np.linalg.det(T))

    K = dK @ T
    K /= K[2,2]
    R = T @ dR
    print("K\n", K)
    print("R\n", R)
    print("det(R)=", np.linalg.det(R))


# 座標系を鏡映状態にする(xyzの3軸のうち1軸の符号を入れ替える : 今回 y軸を反転)
def test4_decomp_camera():

    # Calibration行列
    sK = np.array([
        [1000,0,500],
        [0,-1000,300],
        [0,0,1]], dtype=np.float32) # (3,3)
    print("sK\n", sK)
    
    # 回転
    ax_deg = 30
    ay_deg = 30
    az_deg = 30
    print(f"(θx,θy,θz)=({ax_deg},{ay_deg},{az_deg})")
    # 内因性ZXYオイラー(θ1=az_deg,θ2=ax_deg,θ3=ay_deg)
    wRc = az_rot(math.radians(az_deg)) @ ax_rot(math.radians(ax_deg)) @ ay_rot(math.radians(ay_deg))
    sR = wRc.T # (3,3)
    print("sR\n", sR)
    print("det(sR)=", np.linalg.det(sR))


    '''鏡映状態にするためにsRの2軸目の符号を反転させる'''
    nR = sR.copy()
    nR[1,:] *= -1
    print("sRの2軸目の符号を反転させる : sR[1,:] *= -1") # det(nR)=-1
    print("nR\n", nR)
    print("det(nR)=", np.linalg.det(nR))
    

    # 並進
    st = np.array([[100],[0],[100]], dtype=np.float32)
    print("st\n", st)

    # View行列
    sV = np.hstack((nR,st))
    print("sV\n", sV)

    # Camera(Projection)行列
    sP = sK @ sV
    print("sP\n", sP)

    # RQ分解
    dK, dR = sp.linalg.rq(sP[:3,:3])
    print("dK\n", dK)
    print("dR\n", dR)
    print("det(dR)=", np.linalg.det(dR))

    T = np.diag(np.sign(np.diag(dK)))
    print("T\n", T)
    print("det(T)=", np.linalg.det(T))

    if np.linalg.det(T) < 0:
        T[1,1] *= -1
        print("mod T\n", T)
        print("det(mod T)=", np.linalg.det(T))

    K = dK @ T
    K /= K[2,2]
    R = T @ dR
    print("K\n", K)
    print("R\n", R)
    print("det(R)=", np.linalg.det(R))


# 座標系を鏡映状態にする(xyzの3軸のうち任意の2軸を入れ替える : 今回はx軸とz軸を入れ替え)
def test5_decomp_camera():

    # Projection行列
    sK = np.array([
        [1000,0,500],
        [0,-1000,300],
        [0,0,1]], dtype=np.float32) # (3,3)
    print("sK\n", sK)
    
    # 回転
    ax_deg = 30
    ay_deg = 30
    az_deg = 30
    print(f"(θx,θy,θz)=({ax_deg},{ay_deg},{az_deg})")
    # 内因性ZXYオイラー(θ1=az_deg,θ2=ax_deg,θ3=ay_deg)
    wRc = az_rot(math.radians(az_deg)) @ ax_rot(math.radians(ax_deg)) @ ay_rot(math.radians(ay_deg))
    sR = wRc.T # (3,3)
    print("sR\n", sR)
    print("det(sR)=", np.linalg.det(sR))


    '''sRの1軸目と3軸目の入れ替える'''
    nR = sR.copy()
    nR[0,:] = sR[2,:] # x軸 <- z軸
    nR[2,:] = sR[0,:] # z軸 <- x軸
    print("sRのx軸目とz軸目を入れ替える. nR[:,0] = sR[:,2], nR[:,2] = sR[:,0]") # det(nR)=-1
    print("nR\n", nR)
    print("det(nR)=", np.linalg.det(nR))

    # 並進
    st = np.array([[100],[0],[100]], dtype=np.float32)
    print("st\n", st)

    # View行列
    sV = np.hstack((nR,st))
    print("sV\n", sV)

    # Camera行列
    sP = sK @ sV
    print("sP\n", sP)

    # RQ分解
    dK, dR = sp.linalg.rq(sP[:3,:3])
    print("dK\n", dK)
    print("dR\n", dR)
    print("det(dR)=", np.linalg.det(dR))

    T = np.diag(np.sign(np.diag(dK)))
    print("T\n", T)
    print("det(T)=", np.linalg.det(T))

    if np.linalg.det(T) < 0:
        T[1,1] *= -1
        print("mod T\n", T)
        print("det(mod T)=", np.linalg.det(T))

    K = dK @ T
    K /= K[2,2]
    R = T @ dR
    print("K\n", K)
    print("R\n", R)
    print("det(R)=", np.linalg.det(R))

# 座標系を鏡映状態にする(xyzの全軸を反転させる)
def test6_decomp_camera():

    # Calibration行列
    sK = np.array([
        [-1000,0,500],
        [0,-1000,300],
        [0,0,-1]], dtype=np.float32) # (3,3)
    print("sK\n", sK)
    
    # 回転
    ax_deg = 30
    ay_deg = 30
    az_deg = 30
    print(f"(θx,θy,θz)=({ax_deg},{ay_deg},{az_deg})")
    # 内因性ZXYオイラー(θ1=az_deg,θ2=ax_deg,θ3=ay_deg)
    wRc = az_rot(math.radians(az_deg)) @ ax_rot(math.radians(ax_deg)) @ ay_rot(math.radians(ay_deg))
    sR = wRc.T # (3,3)
    print("sR\n", sR)
    print("det(sR)=", np.linalg.det(sR))


    '''鏡映状態にするためにsRの全軸の符号を反転させる'''
    nR = sR.copy()
    nR[:,:] *= -1
    print("sRの全軸の符号を反転させる : sR[:,:] *= -1") # det(nR)=-1
    print("nR\n", nR)
    print("det(nR)=", np.linalg.det(nR))
    

    # 並進
    st = np.array([[100],[0],[100]], dtype=np.float32)
    print("st\n", st)

    # View行列
    sV = np.hstack((nR,st))
    print("sV\n", sV)

    # Camera(Projection)行列
    sP = sK @ sV
    print("sP\n", sP)

    # RQ分解
    dK, dR = sp.linalg.rq(sP[:3,:3])
    print("dK\n", dK)
    print("dR\n", dR)
    print("det(dR)=", np.linalg.det(dR))

    T = np.diag(np.sign(np.diag(dK)))
    print("T\n", T)
    print("det(T)=", np.linalg.det(T))

    if np.linalg.det(T) < 0:
        T[1,1] *= -1
        print("mod T\n", T)
        print("det(mod T)=", np.linalg.det(T))

    K = dK @ T
    K /= K[2,2]
    R = T @ dR
    print("K\n", K)
    print("R\n", R)
    print("det(R)=", np.linalg.det(R))

if __name__ == "__main__":
    # test1_decomp_camera() # det(T)=+1, det(dR)=+1 -> OK
    # test2_decomp_camera() # det(T)=+1, det(dR)=+1 -> OK
    # test3_decomp_camera() # det(T)=+1, det(dR)=+1 -> OK
    # test4_decomp_camera() # y軸反転, -fy -> OK
    # test5_decomp_camera() # det(R)=-1 ?
    # test6_decomp_camera() # det(T)=+1, det(dR)=+1 -> OK

    # 回転行列の性質に関するテスト
    test_neg2_rot()

