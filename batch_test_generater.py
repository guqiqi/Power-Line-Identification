import numpy as np
import os
import cv2
import image_line_detector
import line_detector_lib


def get_images_by_dir(dirname):
    img_names = os.listdir(dirname)
    img_paths = [dirname + '/' + img_name for img_name in img_names]

    imgs = [cv2.imread(path) for path in img_paths]

    return imgs, img_names


def get_images_by_dir_1_channel(dirname):
    img_names = os.listdir(dirname)
    img_paths = [dirname + '/' + img_name for img_name in img_names]

    imgs = [cv2.imread(path, 0) for path in img_paths]

    return imgs, img_names


def batch_edge_detector(imgs, t, T):
    i = 0
    for img in imgs:
        if img is None:
            pass
        else:
            edge = line_detector_lib.edge_detector(img, 40, 75)
            # cv2.imwrite("processed_" + str(t) + "_" + str(T) + "/" + img_names[i], edge)
            cv2.imwrite("p1/" + img_names[i], edge)

            max_remove = 20
            edge = image_line_detector.edge_detector(img, 40, 75, 10)
            # cv2.imwrite("processed_" + str(max_remove) + "_" + str(t) + "_" + str(T) + "/" + img_names[i], edge)
            cv2.imwrite("p2/1" + img_names[i], edge)

        i = i + 1


def batch_line_detector_from_edge(edges, edges_more, img_names, min_pixels):
    i = 0

    for edge in edges:
        print(img_names[i])
        lines = image_line_detector.fixed_line_detector(edge, min_pixels[i])
        # print(lines)

        # lines[1] = [7.5325000e+02, 1.3217305e-01]
        # lines = lines[:5]
        # lines = [[-8.0750000e+02,3.1154127e+00],
        # [-8.2100000e+02,  3.0717795e+00],
        # [ 7.9400000e+02,  1.7453292e-02]]
        # lines = [
        #     [7.3900000e+02, 3.9269909e-02],
        #     [7.8500000e+02, 0.0000000e+00],
        #     [-8.8700000e+02, 3.0892327e+00]
        # ]
        # lines = [[8.9000000e+02, 1.7453292e-02],
        #          [8.7300000e+02, 6.9813170e-02],
        #          [8.6000000e+02, 1.2217305e-01]
        #          ]
        # lines = [[4.2200000e+02, 5.2359879e-02],
        #          [4.1600000e+02, 1.9198622e-01],
        #          [4.2700000e+02, 1.3962634e-01]]

        # lines = [[6.7900000e+02, 1.3962634e-01],
        #          [6.9400000e+02, 1.0471976e-01],
        #          [7.0200000e+02, 5.2359879e-02]]

        # lines = [
        #     [-6.2600000e+02, 3.0543261e+00],
        #     [6.2900000e+02, 5.2359879e-02],
        #     [6.0900000e+02, 1.3962634e-01]
        # ]

        # lines = [[7.8500000e+02, 1.3962634e-01],
        #          [4.3100000e+02, 1.7453292e-02],
        #          [4.1000000e+02, 5.2359879e-02],
        #          [4.1800000e+02, 0.0000000e+00],
        #          [4.2300000e+02, 6.9813170e-02],
        #          [7.6200000e+02, 1.7453292e-01],
        #          [-4.4000000e+02, 3.0892327e+00]
        #          ]

        # lines = [[-8.2000000e+02, 3.1241393e+00],
        #          [7.2200000e+02, 6.9813170e-02],
        #          [6.3200000e+02, 1.7453292e-01]
        #          ]

        # lines = [[9.8784375e+02, 7.7448988e-01],
        # [9.3200000e+02, 8.3775806e-01]]

        # iii = cv2.imread("images/" + img_names[i])
        iii = cv2.imread("test_images/" + img_names[i])

        i1, i2 = image_line_detector.find_line_in_image(iii, edges_more[i], lines)

        # return i1
        # cv2.imwrite("processed_lines/" + img_names[i], i1)
        # cv2.imwrite("processed_lines/1" + img_names[i], edge)
        cv2.imwrite("p2/" + img_names[i], i1)

        i = i + 1


img_names = [
    # '10kV白泉支线#6-#4.MOV_000200.753.jpg',
    # '10kV白泉支线#6-#4.MOV_000220.741.jpg',
    # '35kV港隆线16-27号3.MOV_015514.714.jpg',
    '10kV溪美线#59-#61、10kV后炉支线1-3号杆.MOV_012700.500.jpg',
    # '10kV里洋II线内坑支线1-24号.MOV_025045.464.jpg',
    # '10kV白泉支线#6-#4.MOV_000120.748.jpg'
    # '10kV白泉支线#4-#2.MOV_000000.035.jpg'
    # '10kV白泉支线#4-#2.MOV_000020.054.jpg'
    # '10kV白泉支线#4-#2.MOV_000040.074.jpg',
    # '10kV白泉支线#4-#2.MOV_000100.061.jpg'
    # '10kV白泉支线#4-#2.MOV_000120.079.jpg',
    # '10kV白泉支线#4-#2.MOV_000140.067.jpg',
    # '10kV白泉支线#6-#4.MOV_000000.735.jpg',
    # '10kV白泉支线#6-#4.MOV_000240.761.jpg'
    # '10kV里洋II线内坑支线1-24号.MOV_024829.818.jpg',
    # '10kV里洋II线内坑支线1-24号.MOV_024933.453.jpg',
    # '10kV里洋II线内坑支线1-24号.MOV_024942.405.jpg',
    # '10kV里洋II线内坑支线1-24号.MOV_024952.121.jpg',
    # '10kV里洋II线内坑支线1-24号.MOV_025001.288.jpg',
    # '10kV里洋II线内坑支线1-24号.MOV_025009.907.jpg'
    # '10kV蔡坑支线4-7号.MOV_015740.290.jpg',
    # '10kV坊洋线青洋支线13号-14号.MOV_000306.389.jpg',
    # '10kV黄温支线1A、1B、1-3号.MOV_020259.047.jpg',
    # '10kV黄温支线1A、1B、1-3号.MOV_020537.826.jpg',
    # '10kV九洋线东林支线4-1号杆.MOV_023925.732.jpg',

    # ---------------------------------------------
    # '10kV坊洋线青洋支线13号-14号.MOV_000024.299.jpg',
    # '10kV九洋线东林支线4-1号杆.MOV_023958.780.jpg',
    # '10kV里洋II线内坑支线1-24号.MOV_025013.593.jpg',
    # '10kV辽上支线16号-22号.MOV_000001.446.jpg',
    # '10kV辽上支线16号-22号.MOV_000021.442.jpg',
    #
    # '10kV辽上支线16号-22号.MOV_000141.457.jpg',
    # '10kV青洋支线#23-#24.MOV_001835.682.jpg',
    # '10kV山重祖洋支线5-1,10kV九重联络支线13-12.MOV_025913.717.jpg',
    # '10kV溪敦线1-26号1.MOV_002532.085.jpg',
    # '10kV溪敦线1-26号2.MOV_002805.126.jpg',
    # -----
    # '10kV溪江线1-26号.MOV_004602.609.jpg',
    # '10kV溪美线#28-#21.MOV_010821.787.jpg',
    # '10kV溪美线#47-#41.MOV_010931.490.jpg',
    # '10kV溪美线#47-#41.MOV_010940.375.jpg',
    # '10kV溪美线#47-#41.MOV_010949.090.jpg',
    #
    # '10kV溪美线#47-#41.MOV_011054.451.jpg',
    # '10kV溪美线#47-#41.MOV_011116.813.jpg',
    # '10kV溪美线#47-#41.MOV_011124.786.jpg',
    # '10kV溪美线#47-#41.MOV_011143.087.jpg',
    # - '10kV溪美线#47-#41.MOV_011151.265.jpg',
    #
    # '10kV溪美线#48-#51.MOV_011620.315.jpg',
    # - '10kV溪美线#48-#51.MOV_011707.857.jpg',
    # - '10kV溪美线#53-#52.MOV_011817.028.jpg',
    # '10kV溪美线#59-#61、10kV后炉支线1-3号杆.MOV_012601.833.jpg',

    # -- '10kV溪视线双口宅支线2-13号2.MOV_013600.381.jpg',
    # '10kV溪视线双口宅支线2-13号4.MOV_014522.958.jpg',
    # '10kV溪视线双口宅支线2-13号5.MOV_014554.479.jpg',
    # '10kV溪视线双口宅支线2-13号5.MOV_014648.791.jpg',
    # '10kV兴华线金鸿帮支线1-15号.MOV_030321.617.jpg',

    # '35kV港隆线1-14号杆1.MOV_031511.922.jpg',
    # '35kV港隆线16-27号2.MOV_015302.368.jpg',
    # '35kV港隆线16-27号2.MOV_015341.654.jpg',
    # '35kV港隆线16-27号3.MOV_015442.894.jpg',
    # '35kV港隆线16-27号3.MOV_015447.930.jpg',
    # '35kV港隆线16-27号3.MOV_015534.919.jpg',

]

# img_paths = ['images/' + img_name for img_name in img_names]
img_paths = ['test_images/' + img_name for img_name in img_names]

imgs = [cv2.imread(path) for path in img_paths]

batch_edge_detector(imgs, 40, 80)

min_pixels = [
    # 580,
    # 680,
    # 180,
    200,
    # 200,
    # 500,
    # 380
    # 450
    # 350,
    # 550
    # 400,
    # 500,
    # 350,
    # 650
    # 170,
    # 220,
    # 220,
    # 170,
    # 150,
    # 195
    # 210,
    # 250,
    # 203,
    # 190,
    # 280,

    # ---------------------------------------------
    # 250,
    # 200,
    # 200,
    # 250,
    # 220

    # 200,
    # 190,
    # 170,
    # 220,
    # 220,

    # 150,
    # 280,
    # 200,
    # 200,
    # 150,

    # 200,
    # 200,
    # 220,
    # 190,
    # - 170,

    # 150,
    # - 200,
    # - 200,
    # 200,

    # -- 200,
    # 250,
    # 200,
    # 300,
    # 220
]

# # img_paths = ["processed_20_40_80" + '/' + img_name for img_name in img_names]
# img_paths = ['p2/1' + img_name for img_name in img_names]
# edges = [cv2.imread(path, 0) for path in img_paths]
#
# # edges_more = [cv2.imread(path, 0) for path in
# #               ["processed_40_80" + '/' + img_name for img_name in img_names]]
# edges_more = [cv2.imread(path, 0) for path in
#               ['p1/' + img_name for img_name in img_names]]
#
# batch_line_detector_from_edge(edges, edges_more, img_names, min_pixels)
