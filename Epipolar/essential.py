"""基本行列Eの推定

@note 対応点
x1j = [x1j,y1j,w1j]^T z=+1 or -1
x2j = [x2j,y2j,w2j]^T z=+1 or -1

E = [[e1,e2,e3],
     [e4,e5,e6],
     [e7,e8,e9]]
基本行列Eはスケールが不変なので, 8自由度になる.

エピポーラ拘束 
x1j^T*E*x2j = 0


"""

import os
import sys
import math

import numpy as np

import rotation
import rvec
import quartanion
import euler

from geometry_context import GeometryContext
from euler_state import EulerState

from type_hint import *

from ransac import Ransac, RansacModel


