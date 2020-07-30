import io
import json
import numpy as np

import cv2

import enhanced_edge_detector
import image_line_detector
import line_detector_lib

dir_name = 'box-data'
file_names = ['9', '18', '23', '28', '30', '31', '35', '38', '39', '51']

min_pixels = [
    # 210, 80,
    # 175
    # xx
    300, 110
    # 430
    # 185
    # 260, 230
    # 295, 220, 280
    # 300
    # 300
]

img_num = 0
j = 0
pix_index = 1
for file in file_names:

    if j < pix_index:
        j = j + 1
        continue

    f = io.open(dir_name + '/' + file_names[j] + '.json', 'r', encoding="utf-8")
    json_data = json.load(f)  # json_data: Dictionary
    tags = json_data['detections']

    img_f = cv2.imread(dir_name + '/' + file + '.jpg')
    for i in range(tags.__len__()):
        print(file + '-' + str(i) + '.jpg')
        img = img_f[int(tags[i]['left_up_point'][1]): int(tags[i]['right_down_point'][1]),
              int(tags[i]['left_up_point'][0]): int(tags[i]['right_down_point'][0])]
        # cv2.imwrite("box-0/" + file + '-' + str(i) + '.jpg', img)

        # edge = line_detector_lib.edge_detector(img, 40, 75)
        # cv2.imwrite("box-1/" + file + '-' + str(i) + '.jpg', edge)
        #
        # edge = enhanced_edge_detector.edge_detector(img, file + '.jpg', 40, 75, 5)
        # cv2.imwrite("box-2/" + file + '-' + str(i) + '.jpg', edge)
        #
        # edge = enhanced_edge_detector.edge_detector(img, file + '.jpg', 40, 75, 10)
        # cv2.imwrite("box-3/" + file + '-' + str(i) + '.jpg', edge)
        #
        # edge = enhanced_edge_detector.edge_detector(img, file + '.jpg', 40, 75, 15)
        # cv2.imwrite("box-4/" + file + '-' + str(i) + '.jpg', edge)

        edge = cv2.imread('box-2/' + file + '-' + str(i) + '.jpg', 0)

        lines = image_line_detector.fixed_line_detector(edge, min_pixels[img_num])
        lines = [
            [4.5950000e+02, 2.9234263e-01],
            [4.3375000e+02, 3.3597589e-01],
            [-6.0500000e+01, 3.0194197e+00],
            [4.2400000e+02, 3.8397244e-01],
            # [-7.0000000e+01,  3.0368729e+00],
            [-8.6000000e+01, 2.9496064e+00],
            # [-5.0000000e+01,  3.0019662e+00],
            [-3.5000000e+01, 3.1066861e+00]]

        # lines = [[2.8900000e+02, 2.5307274e-01],
        #          # [3.2618750e+02, 3.7306413e-01],
        #          [2.5400000e+02, 3.1415927e-01],
        #          [3.3600000e+02, 1.5707964e-01]
        #         ]
        #
        edge_more = cv2.imread('box-1/' + file + '-' + str(i) + '.jpg', 0)

        i1, i2 = image_line_detector.find_line_in_image(img, edge_more, lines)
        # i1 = cv2.imread('box-n/' + file + '-' + str(i) + '.jpg')

        img_f[int(tags[i]['left_up_point'][1]): int(tags[i]['right_down_point'][1]),
        int(tags[i]['left_up_point'][0]): int(tags[i]['right_down_point'][0])] = i1

        cv2.imwrite('box-n/' + file + '-' + str(i) + '.jpg', i1)
        img_num = img_num + 1

    cv2.imwrite('box-f/' + file + '.jpg', img_f)
    j = j + 1

    if j > pix_index:
        break
