"""座標系の設定
"""
import os
import sys

module_parent_dir = '/'.join([os.path.dirname(__file__), '..'])
sys.path.append(module_parent_dir)

import abc
import inspect

import numpy as np

from BasicModule.utility import get_axis_x, get_axis_y, get_axis_z

from type_hint import *

"""
Perspective Division
  カメラ座標(Xc,Yc,Zc,1)をWc'(=およそZc)で, 
  NDC座標(Xc'/Wc',Yc'/Wc',Zc'/Wc')=(Xndc,Yndc,Zndc)に正規化すること.

  [Xc',Yc',Zc',Wc'].T = P[[p11,p12,p13,p14],
                          [p21,p22,p23,p24],
                          [p31,p32,p33,p34],
                          [p41,p42,p43,p44]] @ [Xc,Yc,Zc,1].T

  Xc' = p11*Xc + p12*Yc + p13*Zc + p14*1
  Yc' = p21*Xc + p22*Yc + p23*Zc + p24*1
  Zc' = p31*Xc + p32*Yc + p33*Zc + p34*1
  Wc' = p41*Xc + p42*Yc + p43*Zc + p44*1 
  
  NDC座標(Xc'/Wc',Yc'/Wc',Zc'/Wc')=(Xndc[-1,1],Yndc[-1,1],Zndc[0,1]or[-1,1])

  @warning ピンホールカメラモデルでは p41=p42=p43=p44=Wc'=0になり, Zc=1の面に像が投影される.
  その際, NDC座標ではなく, NIC座標(Zc=1)の平面の撮像素子矩形領域に像が投影される.
  NIC座標(Xc'/Zc',Yc'/Zc',W=Z=1)=(Xnic[-1,1],Ynic[-1,1],1)


座標系定義 (右手系と左手系の区別以外は, 座標軸の向きを[x,y,z]のどれに割り当てるかで複数通りあるが, ここでは,一般的なくくり方で表記する)
  | 手系 | Framework | object-up | object-forward | object-right | camera-up | camera-forward | camera-right | 奥行きクリッピング範囲(NDCのZ) | 備考 |
  | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
  | 右手 | OpenGL系  | +Yo | -Zo | +Xo | +Yc | -Zc | +Xc | Z[-1,+1] | カメラ座標系のY軸が画像座標系と反転している. カメラ視線方向がZ軸正の向き. |
  | 右手 | OpenCV系  | +Zo | +Yo | +Xo | -Yc | +Zc | +Xc | Z[0,+1]  | カメラ座標系のXY軸が画像座標と一致. カメラ視線方向がZ軸正の向き. |
  | 左手 | DirectX系 | +Yo | +Zo | +Xo | +Yc | +Zc | +Xc | Z[0,+1]  | カメラ座標系のY軸が画像座標系と反転している. カメラ視線方向がZ軸正の向き. |
  | 左手 | UE系      | +Zo | +Xo | +Yo | -Yc | -Zc | +Xc | Z[0,+1]  | カメラ座標系のXY軸が画像座標と一致. カメラ視線方向がZ軸負の向き. |

  @warning OpenCV系は, 一番しっくり来る座標系. 物理・工学・CV系で多様される. 
  @warning OpenGL系は, glClipControl(GL_UPPER_LEFT)関数で,`Perspective Division`時にY軸反転できる. 
  @warning OpenGL系は, glClipControl(GL_ZERO_TO_ONE)関数で, NDCの奥行きクリッピング範囲をZ/[0,+1¥]に変更できる.
  @warning DirectX系は, `Perspective Division`の際にY軸符号を反転させる.

  @note OpenGL系(custom) : glClipControl(GL_UPPER_LEFT) と glClipControl(GL_ZERO_TO_ONE) を適用

  
  @note 最もしっくり来る座標系定義がRH_PZup_PYforward_PXrightになるOpenCV系, 
  もしくは, LH_PZup_PYforward_PXrightになるDirectX系. 
  
  @warning DirectXは, ベクトルは行ベクトルで, 行優先表現なので, 
           全ての列優先表現行列を転置して, 行列の積を右側から掛けることになる.
  
  @note クロス積(外積)は, 右手系と左手系で定義が異なるが, 数式としては同じ.
  https://yaju3d.hatenablog.jp/entry/2013/05/26/215841

  @warning 回転ベクトルとクォータニオンは右手系と左手系でベクトル成分の符号が異なる.
           鏡映軸以外の符号を反転させる. 回転量はそのままでよい.
"""

# 右手座標系 PosYup-PosZforward-NegXright (OpenGL, AutoDesk Maya, SolidWorks 系統)
# CoorRH_PYuPZfNXrState
# 右手座標系 PosZup-PosYforward-PosXright (OpenCV, Blender, AutoCAD 系統)
# CoorRH_PZuPYfPXrState
# 左手座標系 PosYup-PosZforward-PosXright (Direct3D, Metal, Unity 系統)
# CoorLH_PYuPZfPXrState
# 左手座標系 PosZup-NegYforward-PosXright (Unreal Engine 系統)
# CoorLH_PZuNYfPXrState



class CoordinateState(metaclass=abc.ABCMeta):

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

    def __init__(self):
        # 行列スタイル (行優先表現: "row-major", 列優先表現: "col-major")
        self.mat_str: Optional[str] = None

        # 座標系スタイル (左手系: "left", 右手系: "right")
        self.coor_str: Optional[str] = None

        # オブジェクト座標系の方向
        self.obj_forward_str: Optional[str] = None
        self.obj_up_str: Optional[str] = None
        self.obj_right_str: Optional[str] = None

        # カメラ座標系の方向
        self.cam_forward_str: Optional[str] = None
        self.cam_up_str: Optional[str] = None
        self.cam_right_str: Optional[str] = None

        # 透視投影変換行列[4x4]
        self.perspective_matrix: np.ndarray = np.ones(4, dtype=np.float32)

        # Inverse Z for NDC 
        # False: NDC Z[0,1] <-> Depth [near,far]
        # True: NDC Z[1,0] <-> Depth [near,far]
        self.is_inverse_z: bool = False

        # Infinity Z for NDC

        # カメラ座標系の各軸方向
        self.cam_x: np.ndarray = np.zeros(3, dtype=np.float32)
        self.cam_y: np.ndarray = np.zeros(3, dtype=np.float32)
        self.cam_z: np.ndarray = np.zeros(3, dtype=np.float32)

        # カメラ座標(位置)
        self.cam_pos: np.ndarray = np.zeros(3, dtype=np.float32)

    @abc.abstractclassmethod
    def look_at(self, 
               target_pos: np.ndarray, 
               camera_pos: np.ndarray, 
               up_axis: Tuple[float, float, float]) -> np.ndarray:
        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__
        raise NotImplementedError(f"No implement {func_name} on {class_name}")
    
    @abc.abstractclassmethod
    def perspective_division(self, 
                             camera_points: np.ndarray, 
                             ndc: bool = True,
                             ) -> np.ndarray:
        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__
        raise NotImplementedError(f"No implement {func_name} on {class_name}")
    
    """ Object Direction """
        
    @abc.abstractclassmethod
    def obj_forward_axis(self, 
                         rot: np.ndarray,
                         ) -> np.ndarray:
        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__
        raise NotImplementedError(f"No implement {func_name} on {class_name}")
    
    @abc.abstractclassmethod
    def obj_right_axis(self, 
                       rot: np.ndarray,
                       ) -> np.ndarray:
        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__
        raise NotImplementedError(f"No implement {func_name} on {class_name}")
    
    @abc.abstractclassmethod
    def obj_up_axis(self, 
                    rot: np.ndarray,
                    ) -> np.ndarray:
        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__
        raise NotImplementedError(f"No implement {func_name} on {class_name}")
    
    """ Object Axis """
    
    @abc.abstractclassmethod
    def obj_x_axis(self, 
                   rot: np.ndarray,
                   ) -> np.ndarray:
        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__
        raise NotImplementedError(f"No implement {func_name} on {class_name}")
    
    @abc.abstractclassmethod
    def obj_y_axis(self, 
                   rot: np.ndarray,
                   ) -> np.ndarray:
        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__
        raise NotImplementedError(f"No implement {func_name} on {class_name}")
    
    @abc.abstractclassmethod
    def obj_z_axis(self, 
                   rot: np.ndarray,
                   ) -> np.ndarray:
        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__
        raise NotImplementedError(f"No implement {func_name} on {class_name}")
    
    """ Object Direction """
    
    @abc.abstractclassmethod
    def cam_forward_axis(self) -> np.ndarray:
        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__
        raise NotImplementedError(f"No implement {func_name} on {class_name}")
    
    @abc.abstractclassmethod
    def cam_right_axis(self) -> np.ndarray:
        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__
        raise NotImplementedError(f"No implement {func_name} on {class_name}")
    
    @abc.abstractclassmethod
    def cam_up_axis(self) -> np.ndarray:
        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__
        raise NotImplementedError(f"No implement {func_name} on {class_name}")
    
    """ Object Axis """

    @abc.abstractclassmethod
    def cam_x_axis(self) -> np.ndarray:
        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__
        raise NotImplementedError(f"No implement {func_name} on {class_name}")
    
    @abc.abstractclassmethod
    def cam_y_axis(self) -> np.ndarray:
        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__
        raise NotImplementedError(f"No implement {func_name} on {class_name}")
    
    @abc.abstractclassmethod
    def cam_z_axis(self) -> np.ndarray:
        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__
        raise NotImplementedError(f"No implement {func_name} on {class_name}")
    

# 右手座標系 PosYup-PosZforward-NegXright (OpenGL, AutoDesk Maya, SolidWorks 系統)
class CoorRH_PYuPZfNXrState(CoordinateState):

    def __init__(self):
        """
        オブジェクトの座標系
          y (正面方向)
          |
          /---x
         z
        """ 
        super(CoorRH_PYuPZfNXrState, self).__init__()
        self.mat_str = "col-major"
        self.coor_str = "right"
        self.obj_forward_str = "nz"
        self.obj_up_str = "py"        
        self.obj_right_str = "px"
        self.cam_forward_str = "pz"
        self.cam_up_str = "py"
        self.cam_right_str = "px"
        

    @CoordinateState.overrides(CoordinateState)
    def look_at(self, 
               target_pos: np.ndarray, 
               camera_pos: np.ndarray, 
               up_axis: Tuple[float, float, float] = (0.0, 1.0, 0.0)) -> np.ndarray:
        """カメラのView行列[4x4]を求める
        視線方向: Z軸の正の向き
        右方向: X軸の正の向き
        上方向: Y軸の正の向き
         y z 視線方向
         |/
         /---x
        Args:
            target_pos (np.ndarray): ターゲット位置ベクトル[3x1]
            camera_pos (np.ndarray): カメラ中心位置ベクトル[3x1]
            up_axis (Tuple[float, float, float]): カメラの上向きベクトル

        Returns:
            np.ndarray: カメラのView行列[4x4]
        """
        # 視線ベクトル
        ct_vec = target_pos - camera_pos
        cam_eye = ct_vec / np.linalg.norm(ct_vec)

        # カメラ上向き
        cam_up = np.array(up_axis, dtype=np.float32)

        # カメラX軸正向きベクトル(単位ベクトル)
        cam_x = np.cross(cam_eye, cam_up)
        cam_x /= np.linalg.norm(cam_x)

        # カメラY軸正向きベクトル(単位ベクトル)
        cam_y = np.cross(cam_x, cam_eye)
        cam_y /= np.linalg.norm(cam_y)

        # カメラZ軸正向きベクトル(単位ベクトル)
        cam_z = np.cross(cam_y, cam_x)
        cam_z /= np.linalg.norm(cam_z)

        # カメラ位置
        tx = -1.0 * np.dot(cam_x, camera_pos)
        ty = -1.0 * np.dot(cam_y, camera_pos)
        tz = -1.0 * np.dot(cam_z, camera_pos)

        # 4x4行列(ΣWの座標をΣCの座標に変換)
        V = np.array([
            [cam_x[0], cam_y[0], cam_z[0], tx],
            [cam_x[1], cam_y[1], cam_z[1], ty],
            [cam_x[2], cam_y[2], cam_z[2], tz],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32) # 列優先表現

        # カメラ座標系の基底軸(方向)を保存
        self.cam_x = cam_x
        self.cam_y = cam_y
        self.cam_z = cam_z

        # カメラの座標系の位置を保存
        self.cam_pos = np.array([tx, ty, tz], dtype=np.float32)

        return V
    
    @CoordinateState.overrides(CoordinateState)
    def perspective_division(self, 
                             camera_points: np.ndarray, 
                             nic: bool = True,
                             ) -> np.ndarray:
        """カメラ座標系からNIC座標系(or NDC座標系)に変換

        Args:
            camera_points (np.ndarray): カメラ座標系の座標点群 [Nx4] (Xc,Yc,Zc,1)
            nic (bool): NDC座標系への変換フラグ
        Returns:
            np.ndarray: NIC座標(or NDC座標) (Xnic,Ynic) or (Xndc, Yndc, Zndc)
        """
        if nic:
            # ピンホールカメラモデルによる撮像素子平面への投影
            pass
        else:
            # Zバッファによるカリングを考慮したピクセルレンダリング
            pass

    """ Object Direction """

    @CoordinateState.overrides(CoordinateState)
    def obj_forward_axis(self, 
                         rot: np.ndarray,
                         ) -> np.ndarray:
        """オブジェクト座標系の前方ベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: 前方ベクトル[3x1]
        """
        # 行列は列優先表現
        return get_axis_z(rot) # forward : Z軸正方向

    @CoordinateState.overrides(CoordinateState)
    def obj_right_axis(self, 
                       rot: np.ndarray,
                       ) -> np.ndarray:
        """オブジェクト座標系の右方ベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: 右方ベクトル[3x1]
        """
        # 行列は列優先表現
        return -1.0 * get_axis_x(rot) # right : X軸負方向
    
    @CoordinateState.overrides(CoordinateState)
    def obj_up_axis(self, 
                    rot: np.ndarray,
                    ) -> np.ndarray:
        """オブジェクト座標系の上方ベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: 上方ベクトル[3x1]
        """
        # 行列は列優先表現
        return get_axis_y(rot) # up : Y軸正方向
    
    """ Object Axis """
    
    @CoordinateState.overrides(CoordinateState)
    def obj_x_axis(self, 
                   rot: np.ndarray,
                   ) -> np.ndarray:
        """オブジェクト座標系のX軸正向きベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: X軸正向きベクトル[3x1]
        """
        # 行列は列優先表現
        return get_axis_x(rot) # forward : X軸正方向

    @CoordinateState.overrides(CoordinateState)
    def obj_y_axis(self, 
                   rot: np.ndarray,
                   ) -> np.ndarray:
        """オブジェクト座標系のY軸正向きベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: Y軸正向きベクトル[3x1]
        """
        # 行列は列優先表現
        return get_axis_y(rot) # forward : Y軸正方向
    
    @CoordinateState.overrides(CoordinateState)
    def obj_z_axis(self, 
                   rot: np.ndarray,
                   ) -> np.ndarray:
        """オブジェクト座標系のZ軸正向きベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: X軸正向きベクトル[3x1]
        """
        # 行列は列優先表現
        return get_axis_z(rot) # forward : Z軸正方向
    
    """ Camera Direction """
    
    @CoordinateState.overrides(CoordinateState)
    def cam_forward_axis(self) -> np.ndarray:
        """カメラ座標系の前方ベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: 前方ベクトル[3x1]
        """
        # 行列は列優先表現
        return self.cam_z
    
    @CoordinateState.overrides(CoordinateState)
    def cam_up_axis(self) -> np.ndarray:
        """カメラ座標系の上方ベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: 上方ベクトル[3x1]
        """
        # 行列は列優先表現
        return -1.0 * self.cam_y

    @CoordinateState.overrides(CoordinateState)
    def cam_right_axis(self) -> np.ndarray:
        """カメラ座標系の右方ベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: 右方ベクトル[3x1]
        """
        # 行列は列優先表現
        return self.cam_x
    
    @CoordinateState.overrides(CoordinateState)
    def cam_x_axis(self) -> np.ndarray:
        """カメラ座標系のX軸正向きベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: X軸正向きベクトル[3x1]
        """
        # 行列は列優先表現
        return self.cam_x
    
    @CoordinateState.overrides(CoordinateState)
    def cam_y_axis(self) -> np.ndarray:
        """カメラ座標系の上方ベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: 上方ベクトル[3x1]
        """
        # 行列は列優先表現
        return self.cam_y

    @CoordinateState.overrides(CoordinateState)
    def cam_z_axis(self) -> np.ndarray:
        """カメラ座標系の右方ベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: 右方ベクトル[3x1]
        """
        # 行列は列優先表現
        return self.cam_z

        
# 右手座標系 PosZup-PosYforward-PosXright (OpenCV, Blender, AutoCAD 系統)
class CoorRH_PZuPYfPXrState(CoordinateState):

    def __init__(self):
        """
        オブジェクトの座標系
          z y(正面方向)
          |/
          /---x
        """
        super(CoorRH_PZuPYfPXrState, self).__init__()
        self.mat_str = "col-major"
        self.coor_str = "right"
        self.obj_forward_str = "py"
        self.obj_up_str = "pz"
        self.obj_right_str = "px"
        self.cam_forward_str = "pz"
        self.cam_up_str = "ny"
        self.cam_right_str = "px"


    @CoordinateState.overrides(CoordinateState)
    def look_at(self, 
               target_pos: np.ndarray, 
               camera_pos: np.ndarray, 
               up_axis: Tuple[float, float, float] = (0.0 ,0.0 ,1.0)) -> np.ndarray:
        """カメラのView行列[4x4]を求める
        視線方向: Z軸の正の向き
        右方向: X軸の正の向き
        上方向: Y軸の負の向き
          z(視線方向)
         /
        /---x
        |
        y
        Args:
            target_pos (np.ndarray): ターゲット位置ベクトル[3x1]
            camera_pos (np.ndarray): カメラ中心位置ベクトル[3x1]
            up_axis (Tuple[float, float, float]): カメラの上向きベクトル

        Returns:
            np.ndarray: カメラのView行列[4x4]
        """
        # 視線ベクトル
        ct_vec = target_pos - camera_pos
        cam_eye = ct_vec / np.linalg.norm(ct_vec)

        # カメラ上向き
        cam_up = np.array(up_axis, dtype=np.float32)

        # カメラX軸正向きベクトル(単位ベクトル)
        cam_x = np.cross(cam_eye, cam_up)
        cam_x /= np.linalg.norm(cam_x)

        # カメラY軸正向きベクトル(単位ベクトル)
        cam_y = np.cross(cam_eye, cam_x)
        cam_y /= np.linalg.norm(cam_y)

        # カメラZ軸正向きベクトル(単位ベクトル)
        cam_z = np.cross(cam_x, cam_y)
        cam_z /= np.linalg.norm(cam_z)

        # カメラ位置
        tx = -1.0 * np.dot(cam_x, camera_pos)
        ty = -1.0 * np.dot(cam_y, camera_pos)
        tz = -1.0 * np.dot(cam_z, camera_pos)

        # 4x4行列(ΣWの座標をΣCの座標に変換)
        V = np.array([
            [cam_x[0], cam_y[0], cam_z[0], tx],
            [cam_x[1], cam_y[1], cam_z[1], ty],
            [cam_x[2], cam_y[2], cam_z[2], tz],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32) # 列優先表現

        # カメラ座標系の基底軸(方向)を保存
        self.cam_x = cam_x
        self.cam_y = cam_y
        self.cam_z = cam_z

        # カメラの座標系の位置を保存
        self.cam_pos = np.array([tx, ty, tz], dtype=np.float32)

        return V
    
    @CoordinateState.overrides(CoordinateState)
    def perspective_division(self, 
                             camera_points: np.ndarray, 
                             nic: bool = True,
                             ) -> np.ndarray:
        """カメラ座標系からNIC座標系(or NDC座標系)に変換

        Args:
            camera_points (np.ndarray): カメラ座標系の座標点群 [Nx4] (Xc,Yc,Zc,1)
            nic (bool): NDC座標系への変換フラグ
        Returns:
            np.ndarray: NIC座標(or NDC座標) (Xnic,Ynic) or (Xndc, Yndc, Zndc)
        """
        if nic:
            # ピンホールカメラモデルによる撮像素子平面への投影
            pass
        else:
            # Zバッファによるカリングを考慮したピクセルレンダリング
            pass

    """ Object Direction """

    @CoordinateState.overrides(CoordinateState)
    def obj_forward_axis(self, 
                         rot: np.ndarray,
                         ) -> np.ndarray:
        """オブジェクト座標系の前方ベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: 前方ベクトル[3x1]
        """
        # 行列は列優先表現
        return get_axis_y(rot) # forward : Y軸正方向

    @CoordinateState.overrides(CoordinateState)
    def obj_right_axis(self, 
                       rot: np.ndarray,
                       ) -> np.ndarray:
        """オブジェクト座標系の右方ベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: 右方ベクトル[3x1]
        """
        # 行列は列優先表現
        return get_axis_x(rot) # right : X軸正方向
    
    @CoordinateState.overrides(CoordinateState)
    def obj_up_axis(self, 
                    rot: np.ndarray,
                    ) -> np.ndarray:
        """オブジェクト座標系の上方ベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: 上方ベクトル[3x1]
        """
        # 行列は列優先表現
        return get_axis_z(rot) # up : Z軸正方向
    
    """ Object Axis """
    
    @CoordinateState.overrides(CoordinateState)
    def obj_x_axis(self, 
                   rot: np.ndarray,
                   ) -> np.ndarray:
        """オブジェクト座標系のX軸正向きベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: X軸正向きベクトル[3x1]
        """
        # 行列は列優先表現
        return get_axis_x(rot) 

    @CoordinateState.overrides(CoordinateState)
    def obj_y_axis(self, 
                   rot: np.ndarray,
                   ) -> np.ndarray:
        """オブジェクト座標系のY軸正向きベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: Y軸正向きベクトル[3x1]
        """
        # 行列は列優先表現
        return get_axis_y(rot)
    
    @CoordinateState.overrides(CoordinateState)
    def obj_z_axis(self, 
                   rot: np.ndarray,
                   ) -> np.ndarray:
        """オブジェクト座標系のZ軸正向きベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: X軸正向きベクトル[3x1]
        """
        # 行列は列優先表現
        return get_axis_z(rot)
    
    """ Camera Direction """
    
    @CoordinateState.overrides(CoordinateState)
    def cam_forward_axis(self) -> np.ndarray:
        """カメラ座標系の前方ベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: 前方ベクトル[3x1]
        """
        # 行列は列優先表現
        return self.cam_z
    
    @CoordinateState.overrides(CoordinateState)
    def cam_up_axis(self) -> np.ndarray:
        """カメラ座標系の上方ベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: 上方ベクトル[3x1]
        """
        # 行列は列優先表現
        return -1.0 * self.cam_y

    @CoordinateState.overrides(CoordinateState)
    def cam_right_axis(self) -> np.ndarray:
        """カメラ座標系の右方ベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: 右方ベクトル[3x1]
        """
        # 行列は列優先表現
        return self.cam_x

    """ Camera Axis """
    
    @CoordinateState.overrides(CoordinateState)
    def cam_x_axis(self) -> np.ndarray:
        """カメラ座標系のX軸正向きベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: X軸正向きベクトル[3x1]
        """
        # 行列は列優先表現
        return self.cam_x
    
    @CoordinateState.overrides(CoordinateState)
    def cam_y_axis(self) -> np.ndarray:
        """カメラ座標系の上方ベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: 上方ベクトル[3x1]
        """
        # 行列は列優先表現
        return self.cam_y

    @CoordinateState.overrides(CoordinateState)
    def cam_z_axis(self) -> np.ndarray:
        """カメラ座標系の右方ベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: 右方ベクトル[3x1]
        """
        # 行列は列優先表現
        return self.cam_z



# 左手座標系 PosYup-PosZforward-PosXright (Direct3D, Metal, Unity 系統)
class CoorLH_PYuPZfPXrState(CoordinateState):

    def __init__(self):
        """オブジェクトの座標系
         y z(正面方向)
         |/
         ---x
        """
        super(CoorLH_PYuPZfPXrState, self).__init__()
        self.mat_str = "col-major"
        self.coor_str = "left"
        self.obj_forward_str = "pz"
        self.obj_up_str = "py"
        self.obj_right_str = "px"
        self.cam_forward_str = "pz"
        self.cam_up_str = "py"
        self.cam_right_str = "px"

    @CoordinateState.overrides(CoordinateState)
    def look_at(self, 
               target_pos: np.ndarray, 
               camera_pos: np.ndarray, 
               up_axis: Tuple[float, float, float] = (0.0, 1.0, 0.0)) -> np.ndarray:
        """カメラのView行列[4x4]を求める
        視線方向: Z軸の正の向き
        右方向: X軸の正の向き
        上方向: Y軸の正の向き
         y z(視線方向)
         |/
         ---x
        Args:
            target_pos (np.ndarray): ターゲット位置ベクトル[3x1]
            camera_pos (np.ndarray): カメラ中心位置ベクトル[3x1]
            up_axis (Tuple[float, float, float]): カメラの上向きベクトル

        Returns:
            np.ndarray: カメラのView行列[4x4]
        """
        # カメラのZ軸正向きベクトル(単位ベクトル) : 視線ベクトル
        ct_vec = target_pos - camera_pos
        cam_eye = ct_vec / np.linalg.norm(ct_vec)

        # カメラ上向き
        cam_up = np.array(up_axis, dtype=np.float32)

        # カメラX軸正向きベクトル(単位ベクトル)
        cam_x = -1.0 * np.cross(cam_eye, cam_up)
        cam_x /= np.linalg.norm(cam_x)

        # カメラY軸正向きベクトル(単位ベクトル)
        cam_y = -1.0 * np.cross(cam_x, cam_eye)
        cam_y /= np.linalg.norm(cam_y)

        # カメラZ軸正向きベクトル(単位ベクトル)
        cam_z = cam_eye

        # カメラ位置
        tx = -1.0 * np.dot(cam_x, camera_pos)
        ty = -1.0 * np.dot(cam_y, camera_pos)
        tz = -1.0 * np.dot(cam_z, camera_pos)

        # 4x4行列(ΣWの座標をΣCの座標に変換)
        V = np.array([
            [cam_x[0], cam_y[0], cam_z[0], tx],
            [cam_x[1], cam_y[1], cam_z[1], ty],
            [cam_x[2], cam_y[2], cam_z[2], tz],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32) # 列優先表現

        # カメラ座標系の基底軸(方向)を保存
        self.cam_x = cam_x
        self.cam_y = cam_y
        self.cam_z = cam_z

        # カメラの座標系の位置を保存
        self.cam_pos = np.array([tx, ty, tz], dtype=np.float32)

        return V
    
    @CoordinateState.overrides(CoordinateState)
    def perspective_division(self, camera_points: np.ndarray, nic: bool = True) -> np.ndarray:
        """カメラ座標系からNIC座標系(or NDC座標系)に変換

        Args:
            camera_points (np.ndarray): カメラ座標系の座標点群 [Nx4] (Xc,Yc,Zc,1)
            nic (bool): NDC座標系への変換フラグ
        Returns:
            np.ndarray: NIC座標(or NDC座標) (Xnic,Ynic) or (Xndc, Yndc, Zndc)
        """
        if nic:
            # ピンホールカメラモデルによる撮像素子平面への投影
            pass
        else:
            # Zバッファによるカリングを考慮したピクセルレンダリング
            pass
    
    """ Object Direction """

    @CoordinateState.overrides(CoordinateState)
    def obj_forward_axis(self, rot: np.ndarray) -> np.ndarray:
        """オブジェクト座標系の前方ベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: 前方ベクトル[3x1]
        """
        # 行列は列優先表現
        return get_axis_z(rot) # forward : Z軸正方向

    @CoordinateState.overrides(CoordinateState)
    def obj_right_axis(self, rot: np.ndarray) -> np.ndarray:
        """オブジェクト座標系の右方ベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: 右方ベクトル[3x1]
        """
        # 行列は列優先表現
        return get_axis_x(rot) # right : X軸正方向
    
    @CoordinateState.overrides(CoordinateState)
    def obj_up_axis(self, rot: np.ndarray) -> np.ndarray:
        """オブジェクト座標系の上方ベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: 上方ベクトル[3x1]
        """
        # 行列は列優先表現
        return get_axis_y(rot) # up : Y軸正方向
    
    """ Object Axis """
    
    @CoordinateState.overrides(CoordinateState)
    def obj_x_axis(self, rot: np.ndarray) -> np.ndarray:
        """オブジェクト座標系のX軸正向きベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: X軸正向きベクトル[3x1]
        """
        # 行列は列優先表現
        return get_axis_x(rot) 

    @CoordinateState.overrides(CoordinateState)
    def obj_y_axis(self, rot: np.ndarray) -> np.ndarray:
        """オブジェクト座標系のY軸正向きベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: Y軸正向きベクトル[3x1]
        """
        # 行列は列優先表現
        return get_axis_y(rot)
    
    @CoordinateState.overrides(CoordinateState)
    def obj_z_axis(self, rot: np.ndarray) -> np.ndarray:
        """オブジェクト座標系のZ軸正向きベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: X軸正向きベクトル[3x1]
        """
        # 行列は列優先表現
        return get_axis_z(rot)
    

    """ Camera Direction """
    
    @CoordinateState.overrides(CoordinateState)
    def cam_forward_axis(self) -> np.ndarray:
        """カメラ座標系の前方ベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: 前方ベクトル[3x1]
        """
        # 行列は列優先表現
        return self.cam_z
    
    @CoordinateState.overrides(CoordinateState)
    def cam_up_axis(self) -> np.ndarray:
        """カメラ座標系の上方ベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: 上方ベクトル[3x1]
        """
        # 行列は列優先表現
        return self.cam_y

    @CoordinateState.overrides(CoordinateState)
    def cam_right_axis(self) -> np.ndarray:
        """カメラ座標系の右方ベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: 右方ベクトル[3x1]
        """
        # 行列は列優先表現
        return self.cam_x

    """ Camera Axis """
    
    @CoordinateState.overrides(CoordinateState)
    def cam_x_axis(self) -> np.ndarray:
        """カメラ座標系のX軸正向きベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: X軸正向きベクトル[3x1]
        """
        # 行列は列優先表現
        return self.cam_x
    
    @CoordinateState.overrides(CoordinateState)
    def cam_y_axis(self) -> Optional[np.ndarray]:
        """カメラ座標系の上方ベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: 上方ベクトル[3x1]
        """
        # 行列は列優先表現
        return self.cam_y

    @CoordinateState.overrides(CoordinateState)
    def cam_z_axis(self) -> Optional[np.ndarray]:
        """カメラ座標系の右方ベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: 右方ベクトル[3x1]
        """
        # 行列は列優先表現
        return self.cam_z

    
# 左手座標系 PosZup-NegYforward-PosXright (Unreal Engine 系統)
class CoorLH_PZuNYfPXrState(CoordinateState):

    def __init__(self):
        """オブジェクトの座標系
          z 正面方向
          |/
          /---x
         y
        """
        super(CoorLH_PZuNYfPXrState, self).__init__()
        self.mat_str = "col-major"
        self.coor_str = "left"
        self.obj_forward_str = "ny"
        self.obj_up_str = "pz"
        self.obj_right_str = "px"
        self.cam_forward_str = "nz"
        self.cam_up_str = "ny"
        self.cam_right_str = "px"

    @CoordinateState.overrides(CoordinateState)
    def look_at(self, 
               target_pos: np.ndarray, 
               camera_pos: np.ndarray, 
               up_axis: Tuple[float, float, float] = (0.0, 0.0, 1.0)) -> np.ndarray:
        """カメラのView行列[4x4]を求める
        視線方向: Z軸の負の向き
        右方向: X軸の正の向き
        上方向: Y軸の負の向き
           視線方向
           /
          /---x
         z|
          y
        Args:
            target_pos (np.ndarray): ターゲット位置ベクトル[3x1]
            camera_pos (np.ndarray): カメラ中心位置ベクトル[3x1]
            up_axis (Tuple[float, float, float]): カメラの上向きベクトル

        Returns:
            np.ndarray: カメラのView行列[4x4]
        """
        # カメラのZ軸負向きベクトル(単位ベクトル) : 視線ベクトル
        ct_vec = target_pos - camera_pos
        cam_eye = ct_vec / np.linalg.norm(ct_vec)

        # カメラ上向き
        cam_up = np.array(up_axis, dtype=np.float32)

        # カメラX軸正向きベクトル(単位ベクトル)
        cam_x = -1.0 * np.cross(cam_eye, cam_up)
        cam_x /= np.linalg.norm(cam_x)

        # カメラY軸正向きベクトル(単位ベクトル)
        cam_y = np.cross(cam_x, cam_eye)
        cam_y /= np.linalg.norm(cam_y)

        # カメラZ軸正向きベクトル(単位ベクトル)
        cam_z = np.cross(cam_x, cam_y)
        cam_z /= np.linalg.norm(cam_z)
        
        # カメラ位置
        tx = -1.0 * np.dot(cam_x, camera_pos)
        ty = -1.0 * np.dot(cam_y, camera_pos)
        tz = -1.0 * np.dot(cam_z, camera_pos)

        # 4x4行列(ΣWの座標をΣCの座標に変換)
        V = np.array([
            [cam_x[0], cam_y[0], cam_z[0], tx],
            [cam_x[1], cam_y[1], cam_z[1], ty],
            [cam_x[2], cam_y[2], cam_z[2], tz],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32) # 列優先表現

        # カメラ座標系の基底軸(方向)を保存
        self.cam_x = cam_x
        self.cam_y = cam_y
        self.cam_z = cam_z

        # カメラの座標系の位置を保存
        self.cam_pos = np.array([tx, ty, tz], dtype=np.float32)

        return V
    
    @CoordinateState.overrides(CoordinateState)
    def perspective_division(self, 
                             camera_points: np.ndarray, 
                             nic: bool = True,
                             ) -> np.ndarray:
        """カメラ座標系からNIC座標系(or NDC座標系)に変換

        Args:
            camera_points (np.ndarray): カメラ座標系の座標点群 [Nx4] (Xc,Yc,Zc,1)
            nic (bool): NDC座標系への変換フラグ
        Returns:
            np.ndarray: NIC座標(or NDC座標) (Xnic,Ynic) or (Xndc, Yndc, Zndc)
        """
        if nic:
            # ピンホールカメラモデルによる撮像素子平面への投影
            pass
        else:
            # Zバッファによるカリングを考慮したピクセルレンダリング
            pass
    
    """ Object Direction """

    @CoordinateState.overrides(CoordinateState)
    def obj_forward_axis(self, rot: np.ndarray) -> np.ndarray:
        """オブジェクト座標系の前方ベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: 前方ベクトル[3x1]
        """
        # 行列は列優先表現
        return -1.0 * get_axis_y(rot) # forward : Y軸負方向

    @CoordinateState.overrides(CoordinateState)
    def obj_right_axis(self, rot: np.ndarray) -> np.ndarray:
        """オブジェクト座標系の右方ベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: 右方ベクトル[3x1]
        """
        # 行列は列優先表現
        return get_axis_x(rot) # right : X軸正方向
    
    @CoordinateState.overrides(CoordinateState)
    def obj_up_axis(self, rot: np.ndarray) -> np.ndarray:
        """オブジェクト座標系の上方ベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: 上方ベクトル[3x1]
        """
        # 行列は列優先表現
        return get_axis_y(rot) # up : Y軸正方向
    
    
    """ Object Axis """
    
    @CoordinateState.overrides(CoordinateState)
    def obj_x_axis(self, rot: np.ndarray) -> np.ndarray:
        """オブジェクト座標系のX軸正向きベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: X軸正向きベクトル[3x1]
        """
        # 行列は列優先表現
        return get_axis_x(rot) 

    @CoordinateState.overrides(CoordinateState)
    def obj_y_axis(self, rot: np.ndarray) -> np.ndarray:
        """オブジェクト座標系のY軸正向きベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: Y軸正向きベクトル[3x1]
        """
        # 行列は列優先表現
        return get_axis_y(rot)
    
    @CoordinateState.overrides(CoordinateState)
    def obj_z_axis(self, rot: np.ndarray) -> np.ndarray:
        """オブジェクト座標系のZ軸正向きベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: X軸正向きベクトル[3x1]
        """
        # 行列は列優先表現
        return get_axis_z(rot)
    

    """ Camera Direction """
    
    @CoordinateState.overrides(CoordinateState)
    def cam_forward_axis(self) -> np.ndarray:
        """カメラ座標系の前方ベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: 前方ベクトル[3x1]
        """
        # 行列は列優先表現
        return self.cam_z
    
    @CoordinateState.overrides(CoordinateState)
    def cam_up_axis(self) -> np.ndarray:
        """カメラ座標系の上方ベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: 上方ベクトル[3x1]
        """
        # 行列は列優先表現
        return -1.0 * self.cam_y

    @CoordinateState.overrides(CoordinateState)
    def cam_right_axis(self) -> np.ndarray:
        """カメラ座標系の右方ベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: 右方ベクトル[3x1]
        """
        # 行列は列優先表現
        return self.cam_x

    """ Camera Axis """
    
    @CoordinateState.overrides(CoordinateState)
    def cam_x_axis(self) -> np.ndarray:
        """カメラ座標系のX軸正向きベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: X軸正向きベクトル[3x1]
        """
        # 行列は列優先表現
        return self.cam_x
    
    @CoordinateState.overrides(CoordinateState)
    def cam_y_axis(self) -> Optional[np.ndarray]:
        """カメラ座標系の上方ベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: 上方ベクトル[3x1]
        """
        # 行列は列優先表現
        return self.cam_y

    @CoordinateState.overrides(CoordinateState)
    def cam_z_axis(self) -> Optional[np.ndarray]:
        """カメラ座標系の右方ベクトル(基底:単位ベクトル)を求める

        Args:
            rot (np.ndarray): 回転行列[3x3]

        Returns:
            np.ndarray: 右方ベクトル[3x1]
        """
        # 行列は列優先表現
        return self.cam_z
