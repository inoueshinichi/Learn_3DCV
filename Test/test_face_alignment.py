"""複数枚の顔画像のいち合わせ
   最初の画像の位置にその他をあわせる
"""
import os
import sys
from tkinter import RIDGE

# os.sepはプラットフォーム固有の区切り文字(Windows: `\`, Unix: `/`)
module_parent_dir = os.sep.join([os.path.dirname(__file__), '..'])
# print("module_parent_dir", module_parent_dir)
sys.path.append(module_parent_dir)

from xml.dom import minidom

import numpy as np

from type_hint import *
from test_utils import test_elapsed_time
from Homography.rigid import rigid_alignment


def read_points_from_xml(xml_filename) -> Dict[str, np.ndarray]:
    """顔の位置合わせ用の制御点を読み込む

    Args:
        xml_filename (_type_): XMLファイルのパス
    """

    xmldoc = minidom.parse(xml_filename)
    face_list = xmldoc.getElementsByTagName('face')
    faces = {}
    for xml_face in face_list:
        filename = xml_face.attributes['file'].value
        # 右目
        xf = int(xml_face.attributes['xf'].value)
        yf = int(xml_face.attributes['yf'].value)
        # 左目
        xs = int(xml_face.attributes['xs'].value)
        ys = int(xml_face.attributes['ys'].value)
        # 口
        xm = int(xml_face.attributes['xm'].value)
        ym = int(xml_face.attributes['ym'].value)

        faces[filename] = np.array([xf, yf, xs, ys, xm, ym])
    
    return faces

@test_elapsed_time
def main():
    xml_file = '../data/jkfaces/jkfaces2008_small/jkfaces.xml'
    points = read_points_from_xml(xml_file)

    # 位置合わせする
    rigid_alignment(faces=points, 
                    path='../data/jkfaces/aligned_image/',
                    plot_flag=True,
                    )

if __name__ == "__main__":
    main()