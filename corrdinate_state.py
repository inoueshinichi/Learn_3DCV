"""座標系設定の状態
"""

import abc

class CoorState(abc.ABCMeta):

    def __init__(self):
        pass


# 右手座標系 Yup
class CoorRightYupState(CoorState):

    def __init__(self):
        super(CoorRightYupState, self).__init__()
        pass

# 右手座標系 Zup
class CoorRightZupState(CoorState):

    def __init__(self):
        super(CoorRightZupState, self).__init__()
        pass

# 左手座標系 Yup
class CoorLeftYupState(CoorState):

    def __init__(self):
        super(CoorLeftYupState, self).__init__()

# 左手座標系 Zup
class CoorLeftZupState(CoorState):

    def __init__(self):
        super(CoorLeftZupState, self).__init__()

