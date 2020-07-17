import cv2

import enhanced_edge_detector
import image_line_detector
import line_detector_lib


def batch_edge_detector(imgs, t, T):
    i = 0
    for img in imgs:
        if img is None:
            pass
        else:
            edge = line_detector_lib.edge_detector(img, t, T)
            # cv2.imwrite("processed_" + str(t) + "_" + str(T) + "/" + img_names[i], edge)
            cv2.imwrite("p1/" + img_names[i], edge)

            max_remove = 20
            edge = enhanced_edge_detector.edge_detector(img, t, T, 10)
            # cv2.imwrite("processed_" + str(max_remove) + "_" + str(t) + "_" + str(T) + "/" + img_names[i], edge)
            cv2.imwrite("p2/1" + img_names[i], edge)

        i = i + 1


def batch_line_detector_from_edge(edges, edges_more, img_names, min_pixels):
    i = 0

    for edge in edges:
        print(img_names[i])
        lines = image_line_detector.fixed_line_detector(edge, min_pixels[i])

        # iii = cv2.imread("images/" + img_names[i])
        iii = cv2.imread("test_images/" + img_names[i])

        i1, i2 = image_line_detector.find_line_in_image(iii, edges_more[i], lines)

        # return i1
        # cv2.imwrite("processed_lines/" + img_names[i], i1)
        # cv2.imwrite("processed_lines/1" + img_names[i], edge)
        cv2.imwrite("p2/" + img_names[i], i1)

        i = i + 1


img_names = [
    # '10kV青洋支线#23-#24.MOV_001827.132.jpg',
    # '10kV溪美线#59-#61、10kV后炉支线1-3号杆.MOV_012649.484.jpg',
    # '10kV溪视线双口宅支线2-13号2.MOV_013343.540.jpg',
    # '35kV港隆线16-27号3.MOV_015447.930.jpg'
    # '10kV辽上支线16号-22号.MOV_000101.458.jpg',
    # '10kV溪美线#28-#21.MOV_010840.605.jpg',

    # '10kV溪江线1-26号.MOV_004739.279.jpg',
    # '10kV黄温支线1A、1B、1-3号.MOV_020346.955.jpg',
    # '10kV九洋线东林支线10-5号杆10kV三娘联络支线T接三基杆.MOV_022357.721.jpg',
    #
    # '10kV兴华线金鸿帮支线1-15号.MOV_030329.496.jpg',
]
min_pixels = [
    # 400,
    # 320,
    # 500,
    # 110
    # 330,
    # 530

    # 650
]

# img_paths = ['test_images/' + img_name for img_name in img_names]
# imgs = [cv2.imread(path) for path in img_paths]
#
# batch_edge_detector(imgs, 40, 75)

img_paths = ['p2/1' + img_name for img_name in img_names]
edges = [cv2.imread(path, 0) for path in img_paths]

# edges_more = [cv2.imread(path, 0) for path in
#               ["processed_40_80" + '/' + img_name for img_name in img_names]]
edges_more = [cv2.imread(path, 0) for path in
              ['p1/' + img_name for img_name in img_names]]

batch_line_detector_from_edge(edges, edges_more, img_names, min_pixels)
