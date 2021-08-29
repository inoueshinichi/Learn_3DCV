"""Numpy_STLのテスト
"""

import os
import sys
import math
from typing import List, Dict, Tuple, Union
from copy import deepcopy
from stl import mesh
from PIL import Image, ImageDraw
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True) # 指数表記の禁止
import cv2

# 自作
from transform import *




def transform_eulerZXY(ax_deg, ay_deg, az_deg, pos_, ex_, ey_, ez_):
    dcm = dcm_eulerZXY(ax_deg, ay_deg, az_deg)
    pos = np.copy(pos_)
    ex = np.copy(ex_)
    ey = np.copy(ey_)
    ez = np.copy(ez_)

    # 親座標系の座標値に変換
    pos = dcm[:3,:3] @ pos
    ex = dcm[:3,:3] @ ex
    ey = dcm[:3,:3] @ ey
    ez = dcm[:3,:3] @ ez

    return (pos, ex, ey, ez), dcm




def transform_translate(tx, ty, tz, pos_, ex_, ey_, ez_):
    trans = translate(tx, ty, tz)
    pos = np.copy(pos_)
    ex = np.copy(ex_)
    ey = np.copy(ey_)
    ez = np.copy(ez_)
    _pos = np.array([0,0,0,1], dtype=np.float64)
    _pos[:-1] = pos
    _ex = np.array([0,0,0,1], dtype=np.float64)
    _ex[:-1] = ex
    _ey = np.array([0,0,0,1], dtype=np.float64)
    _ey[:-1] = ey
    _ez = np.array([0,0,0,1], dtype=np.float64)
    _ez[:-1] = ez
    # print("_pos {0}, _ex {1}, _ey {2}, _ez {3}".format(_pos, _ex, _ey, _ez))

    # 親座標系の座標値に変換
    pos = trans @ _pos
    ex = trans @ _ex
    ey = trans @ _ey
    ez = trans @ _ez
    # print("pos {0}, ex {1}, ey {2}, ez {3}".format(pos, ex, ey, ez))
    return (pos[:-1], ex[:-1], ey[:-1], ez[:-1]), trans


def pose_matrix(tx, ty, tz, ax_deg, ay_deg, az_deg):
    trans = translate(tx, ty, tz)
    dcm = dcm_eulerZXY(ax_deg, ay_deg, az_deg)

    print("dcm", dcm)
    print("trans", trans)
    T = trans @ dcm
    print("T", T)
    return T

def transform_matrix(tx, ty, tz, ax_deg, ay_deg, az_deg, pos_, ex_, ey_, ez_):
    T = pose_matrix(tx, ty, tz, ax_deg, ay_deg, az_deg)
    pos = np.copy(pos_)
    ex = np.copy(ex_)
    ey = np.copy(ey_)
    ez = np.copy(ez_)
    _pos = np.array([0,0,0,1], dtype=np.float64)
    _pos[:-1] = pos
    _ex = np.array([0,0,0,1], dtype=np.float64)
    _ex[:-1] = ex
    _ey = np.array([0,0,0,1], dtype=np.float64)
    _ey[:-1] = ey
    _ez = np.array([0,0,0,1], dtype=np.float64)
    _ez[:-1] = ez

    pos = T @ _pos
    ex = T @ _ex
    ey = T @ _ey
    ez = T @ _ez
    return (pos[:-1], ex[:-1], ey[:-1], ez[:-1]), T





def my_quiver(axes, pos, ex, ey, ez):
    axes.quiver3D(pos[0], pos[1], pos[2], ex[0]-pos[0], ex[1]-pos[1], ex[2]-pos[2], color="magenta")   # x
    axes.quiver3D(pos[0], pos[1], pos[2], ey[0]-pos[0], ey[1]-pos[1], ey[2]-pos[2], color="yellow") # y
    axes.quiver3D(pos[0], pos[1], pos[2], ez[0]-pos[0], ez[1]-pos[1], ez[2]-pos[2], color="cyan")  # z



def main():

    # STLファイル
    stl_mesh = mesh.Mesh.from_file("./Panther+Origami/files/panter.stl")
    data = deepcopy(stl_mesh)

    # カメラ座標系
    pos = np.array([0,0,0])
    ex = np.array([1,0,0])
    ey = np.array([0,1,0])
    ez = np.array([0,0,1])

    

    '''描画'''
    figure = plt.figure()
    axes = mplot3d.Axes3D(figure)
    axes.view_init(elev=45, azim=45)
    scale = data.points.flatten(-1)
    axes.auto_scale_xyz(scale, scale, scale)

    # STLファイルの頂点のスケールを落とす
    data.vectors[:,:,:] /= 40
    data.vectors[:,:,0] += 2

    # STL
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(data.vectors))
    
    # 原点
    axes.quiver3D(0, 0, 0, 1, 0, 0, length=1, color="red") # x
    axes.quiver3D(0, 0, 0, 0, 1, 0, length=1, color="green") # y
    axes.quiver3D(0, 0, 0, 0, 0, 1, length=1, color="blue") # z

    # 座標系A
    (A2O_pos, A2O_ex, A2O_ey, A2O_ez), dcm = transform_eulerZXY(ax_deg=30, ay_deg=30, az_deg=30, pos_=pos, ex_=ex, ey_=ey, ez_=ez)
    print("A2O_pos {0}, A2O_ex {1}, A2O_ey {2}, A2O_ez {3}".format(A2O_pos, A2O_ex, A2O_ey, A2O_ez))
    my_quiver(axes, A2O_pos, A2O_ex, A2O_ey, A2O_ez)

    point0 = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,1]])
    point1_x = (dcm[:3,:3] @ point0[0]).reshape(-1, 1)
    point1_y = (dcm[:3,:3] @ point0[1]).reshape(-1, 1)
    point1_z = (dcm[:3,:3] @ point0[2]).reshape(-1, 1)
    point1_xyz = (dcm[:3,:3] @ point0[3]).reshape(-1, 1)
    point1 = np.concatenate((point1_x, point1_y, point1_z), axis=1).T
    print("point1_x{0}, point1_y{1}, point1_z{2}".format(point1_x, point1_y, point1_z))
    print("point1", point1)
    axes.scatter3D(xs=point0[:,0], ys=point0[:,1],zs=point0[:,2]) # (ex_O, ey_O, ez_O)
    axes.scatter3D(xs=point1[:,0], ys=point1[:,1],zs=point1[:,2]) # (ex_A, ey_A, ez_A)
    axes.scatter3D(xs=point1_xyz[0], ys=point1_xyz[1], zs=point1_xyz[2]) 

    # 座標系B
    print("pos {0}, ex {1}, ey {2}, ez {3}".format(pos, ex, ey, ez))
    (B2O_pos, B2O_ex, B2O_ey, B2O_ez), trans = transform_translate(tx=-2, ty=-2, tz=1, pos_=pos, ex_=ex, ey_=ey, ez_=ez)
    print("B2O_pos {0}, B2O_ex {1}, B2O_ey {2}, B2O_ez {3}".format(B2O_pos, B2O_ex, B2O_ey, B2O_ez))
    my_quiver(axes, B2O_pos, B2O_ex, B2O_ey, B2O_ez)

    point0t = np.array([[1,0,0,1],[0,1,0,1],[0,0,1,1],[1,1,1,1]])
    point2_x = (trans @ point0t[0]).reshape(-1, 1)[:-1]
    point2_y = (trans @ point0t[1]).reshape(-1, 1)[:-1]
    point2_z = (trans @ point0t[2]).reshape(-1, 1)[:-1]
    point2_xyz = (trans @ point0t[3]).reshape(-1, 1)[:-1]
    point2 = np.concatenate((point2_x, point2_y, point2_z), axis=1).T
    print("point2_x{0}, point2_y{1}, point2_z{2}".format(point2_x, point2_y, point2_z))
    print("point2", point2)
    axes.scatter3D(xs=point2[:,0], ys=point2[:,1],zs=point2[:,2])         # (ex_B, ey_B, ez_B)
    axes.scatter3D(xs=point2_xyz[0], ys=point2_xyz[1], zs=point2_xyz[2]) 

    # 座標系C
    (C2O_pos, C2O_ex, C2O_ey, C2O_ez), C2O_pose_matrix =  transform_matrix(tx=-2, ty=0, tz=-2, 
                                                       ax_deg=30, ay_deg=30, az_deg=30,
                                                       pos_=pos, ex_=ex, ey_=ey, ez_=ez)
    my_quiver(axes, C2O_pos, C2O_ex, C2O_ey, C2O_ez)

    point0t = np.array([[1,0,0,1],[0,1,0,1],[0,0,1,1],[1,1,1,1]])
    point3_x = (C2O_pose_matrix @ point0t[0]).reshape(-1, 1)[:-1]
    point3_y = (C2O_pose_matrix @ point0t[1]).reshape(-1, 1)[:-1]
    point3_z = (C2O_pose_matrix @ point0t[2]).reshape(-1, 1)[:-1]
    point3_xyz = (C2O_pose_matrix @ point0t[3]).reshape(-1, 1)[:-1]
    point3 = np.concatenate((point3_x, point3_y, point3_z), axis=1).T
    print("point3_x{0}, point3_y{1}, point3_z{2}".format(point3_x, point3_y, point3_z))
    print("point3", point3)
    axes.scatter3D(xs=point3[:,0], ys=point3[:,1],zs=point3[:,2])         # (ex_B, ey_B, ez_B)
    axes.scatter3D(xs=point3_xyz[0], ys=point3_xyz[1], zs=point3_xyz[2]) 


    # 座標系D = 座標系O -> 座標系C -> 座標系D
    (D2C_pos, D2C_ex, D2C_ey, D2C_ez), D2C_pose_matrix = transform_matrix(tx=0, ty=1, tz=0, 
                                                       ax_deg=0, ay_deg=0, az_deg=45,
                                                       pos_=pos, ex_=ex, ey_=ey, ez_=ez)
    my_quiver(axes, D2C_pos, D2C_ex, D2C_ey, D2C_ez)
    
    pos_t = np.append(pos, 1)
    ex_t = np.append(ex, 1)
    ey_t = np.append(ey, 1)
    ez_t = np.append(ez, 1)
    D2O_pose_matrix = C2O_pose_matrix @ D2C_pose_matrix
    D2O_pos_t = D2O_pose_matrix @ pos_t
    D2O_ex_t = D2O_pose_matrix @ ex_t
    D2O_ey_t = D2O_pose_matrix @ ey_t
    D2O_ez_t = D2O_pose_matrix @ ez_t
    my_quiver(axes, D2O_pos_t[:-1], D2O_ex_t[:-1], D2O_ey_t[:-1], D2O_ez_t[:-1])

    # LookAt方式によるカメラ座標系の設定
    c = np.array([-2.3, -3.0, 2])
    t = np.array([-2.3, -3.0, 0])
    w2c_transform = lookAt(c=c, t=t, u=np.array([0,1,0])) 
    pose_camera = np.linalg.inv(w2c_transform)
    camera_pos = pose_camera @ np.array([0,0,0,1])
    camera_ex = pose_camera @ np.array([1,0,0,1])
    camera_ey = pose_camera @ np.array([0,1,0,1])
    camera_ez = pose_camera @ np.array([0,0,1,1])
    my_quiver(axes, camera_pos[:-1], camera_ex[:-1], camera_ey[:-1], camera_ez[:-1])


    axes.set_xlim([-3,3])
    axes.set_ylim([-3,3])
    axes.set_zlim([-3,3])
    plt.show()


if __name__ == "__main__":
    main()