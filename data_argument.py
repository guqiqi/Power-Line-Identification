import math
import random

import numpy as np
import os
import cv2
# from keras.preprocessing import image as ksimage


def get_images_by_dir(dirname):
    img_names = os.listdir(dirname)
    img_paths = [dirname + '/' + img_name for img_name in img_names]

    imgs = [cv2.imread(path) for path in img_paths]

    return imgs, img_names


# def ks_rotate(x, theta, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
#     rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
#     h, w = x.shape[row_axis], x.shape[col_axis]
#     transform_matrix = ksimage.transform_matrix_offset_center(rotation_matrix, h, w)
#     x = ksimage.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
#     return x


imgs, img_names = get_images_by_dir('data')
i = 0
for img in imgs:
    # 翻转
    # flip_img = cv2.flip(img, 1)
    # cv2.imwrite('datah/1' + img_names[i], flip_img)
    # flip_img = cv2.flip(img, 0)
    # cv2.imwrite('datav/2' + img_names[i], flip_img)
    # flip_img = cv2.flip(img, -1)
    # cv2.imwrite('datam/3' + img_names[i], flip_img)

    # 旋转
    # rotate_limit = (0, 30)
    # theta = np.pi / 180 * np.random.uniform(rotate_limit[0], rotate_limit[1])  # 逆时针旋转角度
    # # rotate_limit= 30 #自定义旋转角度
    # # theta = np.pi /180 *rotate_limit #将其转换为PI
    # img_rot = ks_rotate(img, theta)
    # cv2.imwrite('datar/4' + img_names[i], img_rot)
    angle = random.random() * 20 - 10
    scale = 1
    w = img.shape[1]
    h = img.shape[0]
    # 角度变弧度
    rangle = np.deg2rad(angle)
    # 计算新图像的宽度和高度，分别为最高点和最低点的垂直距离
    nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
    nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
    # 获取图像绕着某一点的旋转矩阵
    # getRotationMatrix2D(Point2f center, double angle, double scale)
    # Point2f center：表示旋转的中心点
    # double angle：表示旋转的角度
    # double scale：图像缩放因子
    # 参考：https://cloud.tencent.com/developer/article/1425373
    rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)  # 返回 2x3 矩阵
    # 新中心点与旧中心点之间的位置
    rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]
    # 仿射变换
    rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))),
                             flags=cv2.INTER_LANCZOS4)  # ceil向上取整
    cv2.imwrite('datar/3' + img_names[i], rot_img)
    i = i + 1
