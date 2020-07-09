import os
import cv2
import image_line_detector
import line_detector_lib


def get_images_by_dir(dirname):
    img_names = os.listdir(dirname)
    # del img_names[9]
    img_paths = [dirname + '/' + img_name for img_name in img_names]

    imgs = [cv2.imread(path) for path in img_paths]

    return imgs, img_names


def get_images_by_dir_1_channel(dirname):
    img_names = os.listdir(dirname)
    del img_names[3]
    img_paths = [dirname + '/' + img_name for img_name in img_names]

    imgs = [cv2.imread(path, 0) for path in img_paths]

    return imgs, img_names


def batch_edge_detector(imgs, t, T):
    i = 0
    for img in imgs:
        if img is None:
            pass
        # img1 = image_line_detector.line_detector(img)
        # cv2.imwrite("test_processed/hough"+str(i)+".jpg", img1)
        else:
            max_remove = 20

            # edge = image_line_detector.edge_detector(img, t, T, max_remove)
            line_detector_lib.edge_detector(img, t, T)

            # cv2.imwrite("processed_image_full_" + str(t) + "_" + str(T) + "/" + img_names[i] + "edge.jpg", edge)

        i = i + 1


def batch_line_detector_from_edge(edges, edges_more, img_names):
    i = 0

    for edge in edges:
        # print(img_names[i])
        lines = image_line_detector.fixed_line_detector(edge, 300)
        #
        iii = cv2.imread("test_images/" + img_names[i])
        # print(iii.shape)
        #
        # lines[1] = [7.5325000e+02, 1.3217305e-01]
        # print(lines)
        i1, i2 = image_line_detector.find_line_in_image(iii, edges_more[i], lines)

        # return i1

        cv2.imwrite("processed_image_full_line/6" + img_names[i], i1)
        cv2.imwrite("processed_image_full_line/5" + img_names[i], edge)

        i = i + 1


# img_names = ['10kV溪视线双口宅支线2-13号5.MOV_014537.951.jpg', '10kV白泉支线#4-#2.MOV_000100.061.jpg',
#              '10kV白泉支线#6-#4.MOV_000240.761.jpg', '10kV里洋II线内坑支线1-24号.MOV_025009.907.jpg']
img_names = [
    # '10kV白泉支线#6-#4.MOV_000200.753.jpg',
    # '10kV白泉支线#4-#2.MOV_000040.074.jpg',
    # '10kV白泉支线#6-#4.MOV_000220.741.jpg',
    #  '10kV白泉支线#6-#4.MOV_000240.761.jpg',
    # '10kV白泉支线#4-#2.MOV_000100.061.jpg',
    # '10kV白泉支线#4-#2.MOV_000140.067.jpg',
    # '10kV坊洋线青洋支线13号-14号.MOV_000004.286.jpg',
    # '35kV港隆线16-27号1.MOV_014931.000.jpg',
    # '35kV港隆线16-27号3.MOV_015534.919.jpg',
    # '10kV里洋II线内坑支线1-24号.MOV_025009.907.jpg',
    # '35kV港隆线16-27号3.MOV_015514.714.jpg'
    # '10kV溪美线#59-#61、10kV后炉支线1-3号杆.MOV_012700.500.jpg',
    # '10kV里洋II线内坑支线1-24号.MOV_025045.464.jpg'
    '10kV白泉支线#4-#2.MOV_000000.035.jpg'
]

img_paths = ["processed_image_full_1_40_80" + '/' + img_name + "edge.jpg" for img_name in img_names]
print(img_paths)
edges = [cv2.imread(path, 0) for path in img_paths]
edges_more = [cv2.imread(path, 0) for path in
              ["processed_image_full_40_80" + '/' + img_name + "edge.jpg" for img_name in img_names]]

batch_line_detector_from_edge(edges, edges_more, img_names)

# imgs, img_names = get_images_by_dir("test_images")
# batch_edge_detector(imgs, 40, 80)
