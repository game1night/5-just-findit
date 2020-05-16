#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Created on 2020/5/7 18:28

@author: tatatingting
"""

import os
import subprocess
import sys
import time
from io import BytesIO

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def main_split_img(url, url2, len, label_def):
    # 确认地址
    path_dir = os.path.dirname(__file__)
    path_img = os.path.join(path_dir, url)
    # 打开文件
    dirs = os.listdir(path_img)
    # 输出所有文件
    for file in dirs:
        # 当前截图地址
        path_img_url = os.path.join(path_img, file)
        # 获取截图
        img = Image.open(path_img_url)
        w, h = img.size
        # 修正截图方向
        if w > h:
            img = img.rotate(-90, expand=True)
        # 操作截图 img = Image.fromarray(img.astype('uint8')).convert('RGB')
        img = np.array(img)
        print(file, img.shape)
        # 确定块大小
        brick_len = len
        # 确定缓冲块大小 = 步长
        brick_len2 = np.int(len / 2)
        # 确定切割份数，以纵向为例
        brick_h = h // brick_len2 - 2 + 1
        # 开始扫描切割
        for i in range(brick_h):
            # 确认切割起点
            point_start = brick_len2 * i
            if point_start < 0:
                point_start = 0
            # 确认切割尾点
            point_end = point_start + brick_len
            if point_end > h:
                point_end = h
            img_temp = img[point_start: point_end, :]
            # plt.imshow(img_temp)
            # plt.show()
            path_img_url2 = os.path.join(path_dir, url2, '{}_{}_{}'.format(label_def, i, file))
            plt.imsave(path_img_url2, img_temp)

    return None


if __name__ == '__main__':
    main_split_img('img_raw', 'img_tidy', len=216, label_def=0)
    print('完成啦！自己动手进行【图片标注/人工分类】吧~~~')

