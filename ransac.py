"""RANSACアルゴリズム 
Reuqirement: [1]最大反復回数, [2]誤差の要求値[閾値], [3]必要インライア数, [4]誤差の定義, [5]推定モデルの定義
1. データセットからランダム(一様分布)にデータを選ぶ
2. 選んだデータとモデルからパラメータを推定
3. 各データ点と推定モデルから誤差を計算
4. 誤差が要求値以下ならインライアに登録
5. インライアが必要個数を上回ったら推定モデル(パラメータ)を正解モデル候補に登録
6. 1~5を最大反復回数だけ繰り返す
7. 正解候補モデルの中から最小誤差となるモデルをロバスト推定モデルとして返す
"""

import os
import sys
import math
import abc
import random
import inspect

import numpy as np

from type_hint import *

class RansacModel(abc.ABCMeta):

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
        pass

    @abc.abstractclassmethod
    def fit(self, data: np.ndarray) -> Any:
        # モデルを推定する
        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__
        raise NotImplementedError(f"No implement {func_name} on {class_name}")
    
    @abc.abstractclassmethod
    def get_error(self, data: np.ndarray, estimated_model: Any) -> np.ndarray:
        # データの誤差を返す [1xN]
        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__
        raise NotImplementedError(f"No implement {func_name} on {class_name}")


class Ransac:
    def __init__(self, 
                 required_min_num_data: int,
                 required_inliers: int,
                 match_threshold: float,
                 max_iter: int = 1000, 
                 seed: Optional[int] = None):
    
        self.required_min_num_data: int = required_min_num_data
        self.required_inliers: int = required_inliers # インライア個数の要求値
        self.match_threshold : float = match_threshold # 誤差閾値. この閾値以下のデータをインライアとみなす
        self.max_iter: int = max_iter # 最大反復回数
        self.seed: Optional[int] = seed
        
    def set_seed(self, seed: int):
        self.seed = seed

    def execute(self, 
                data: np.ndarray, 
                model: RansacModel) -> Union[Any, np.ndarray]:
        """Ransacを実行

        Args:
            data (np.ndarray): データ[NxD] N: データ点数, D: データ次元数
            model (RansacModel): RansacModelの派生クラス

        Returns:
            Union[Any, np.ndarray]: (ロバスト推定モデル, インライアマスク)
        """
        if self.seed is None:
            self.seed = random.seed(None) # 現在時刻から計算

        all_num: int = data.shape[1] # データセット数
        candidate_model_list : List[Tuple[Any, np.ndarray]] = [] # (正解候補モデル, インライアマスク)

        for _ in range(self.max_iter):
            # ランダムに選択したインデックス
            indices = [random.randrange(0, all_num) for _ in range(self.required_min_num_data)]
            
            # モデルの推定    
            selected_data = data[indices] # 同じデータの重複あり
            estimated_model = model.fit(selected_data)

            # インライアの判別
            inlier_mask = np.zeros((1, all_num), dtype=np.bool)
            errors: np.ndarray = model.get_error(data, estimated_model)
            inlier_mask = errors < self.match_threshold # np.float64 = float

            # 正解候補モデルの登録
            if np.sum(inlier_mask.astype(np.int32)) >= self.required_inliers:
                candidate_model_list.append((estimated_model, inlier_mask))

        # ロバスト推定モデルとインライアマスクの決定
        model_errors = np.zeros(len(candidate_model_list), dtype=np.float64)
        for i in range(0, len(candidate_model_list)):
            candidate_model = candidate_model_list[i][0]
            model_errors[i] = np.sum(model.get_error(data, candidate_model)) # モデル誤差
        robust_index = np.argmax(model_errors)

        return candidate_model_list[robust_index]



        




