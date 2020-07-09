import numpy as np
import os
import cv2
import image_line_detector
import line_detector_lib


def get_images_by_dir(dirname):
    img_names = os.listdir(dirname)
    img_paths = [dirname + '/' + img_name for img_name in img_names]

    # 存在.DS_Store文件，很迷的问题
    del img_paths[2]

    imgs = [cv2.imread(path) for path in img_paths]
    return imgs


# def get_edge_image(images):
#     i = 0
#     for img in images:
#         img1 = line_detector_lib.edge_detector(img)
#         # img1 = image_line_detector.edge_detector(img)
#         cv2.imwrite("processed2/edge" + str(i) + ".jpg", img1)
#
#         i = i + 1


imgs_3channel = get_images_by_dir('test_image')

# get_edge_image(imgs_3channel)

# # 图片增强
i = 0
for img in imgs_3channel:
    # img1 = image_line_detector.line_detector(img)
    # cv2.imwrite("test_processed/hough"+str(i)+".jpg", img1)

    T = 100
    max_remove = 10
    min_line = 100

    for T in range(100, 101, 50):
        for max_remove in range(0, 11, 10):
            i1 = img.copy()
            edge = image_line_detector.edge_detector(i1, 50, T, max_remove)


            cv2.imwrite("processed1/" + str(i) + "edge" + str(T) + "-" + str(max_remove) + ".jpg", edge)

            print(str(i) + "edge" + str(T) + "-" + str(max_remove))
            for min_line in range(100, 300, 100):
                i2 = img.copy()

                lines = cv2.HoughLines(edge, 1, np.pi / 180, min_line)
                if lines is not None:
                    lines1 = lines[:, 0, :]

                    # draw line on img
                    for rho, theta in lines1[:]:
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + 1000 * (-b))
                        y1 = int(y0 + 1000 * (a))
                        x2 = int(x0 - 1000 * (-b))
                        y2 = int(y0 - 1000 * (a))

                        cv2.line(i2, (x1, y1), (x2, y2), (255, 0, 0), 1)
                print(str(i) + "line" + str(T) + "-" + str(max_remove) + "-" + str(min_line))
                cv2.imwrite(
                    "processed1/" + str(i) + "line" + str(T) + "-" + str(max_remove) + "-" + str(min_line) + ".jpg", i2)

    i = i + 1

    # 线性对比度变化，小于1的效果更好
    # img = 1.5 * img
    # img[img > 255] = 255
    # 解决过度曝光问题，反相取小
    # (width, height) = img.shape
    # print(width + " " + height)
    # 显示三通道增强图
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.imshow("image", img)

    # img = img * 1.5
    # img[img > 255] = 255
    # img = img.astype(np.uint8)
    # cv2.namedWindow("1.5", cv2.WINDOW_NORMAL)

    # img2 = np.float(0.75) * img
    # #
    # img2[img2 > 255] = 255
    # img2.astype(np.uint8)
    # cv2.namedWindow('0.75', cv2.WINDOW_NORMAL)
    # cv2.imshow("0.75", img2)
    # cv2.waitKey(0)
    # 变成灰度图
    # img_1channel = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # imgs_1channel.append(img_1channel.astype(np.uint8))

#
#
#
# canny边缘识别
# i = 0
# for img in imgs_1channel:
#     i = i + 1
#     # img = cv2.GaussianBlur(img, (3, 3), 1.5)
#     # img1 = cv2.Canny(img, 50, 300)
#     # cv2.imwrite("test_processed/canny对比度1.5" + str(i) + ".png", img1)
#
#     # sobel
#     x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
#     y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
#
#     absX = cv2.convertScaleAbs(x) # 转回uint8
#     absY = cv2.convertScaleAbs(y)
#
#     dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
#
#     # cv2.imshow("absX", absX)
#     # cv2.imshow("absY", absY)
#
#     # cv2.imshow("Result", dst)
#     #
#     # cv2.waitKey(0)
#     cv2.namedWindow('canny', cv2.WINDOW_NORMAL)
#     cv2.imshow("canny", absX)
#     cv2.waitKey(0)
#     cv2.imwrite("test_processed/sobel分开" + str(i) + ".png", dst)


# img = cv2.imread("images/10kV白泉支线#4-#2.MOV_000000.035.jpg")
#
#
# img1 = cv2.GaussianBlur(img[..., 0], (3, 3), 1.5)
# img1 = cv2.Canny(img1, 50, 300)
# img4 = cv2.GaussianBlur(img1, (5, 5), 1.5)
# cv2.imwrite("processed2/red30052.png", img1)
#
# img4 = cv2.imread("images/10kV白泉支线#4-#2.MOV_000000.035.jpg", 0)
# img4 = cv2.GaussianBlur(img4, (3, 3), 1.5)
# img4 = cv2.Canny(img4, 50, 300)
# img4 = cv2.GaussianBlur(img4, (5, 5), 1.5)
# cv2.imwrite("processed2/gray30052.png", img4)
# img1 = img[..., 0]
# for i in range(10):
#     canny = cv2.Canny(img1, 100-5.0*i, 100+15.0*i)
#     cv2.imwrite("processed2/red" + str(i) + ".png", canny)
#
# img2 = img[..., 1]
# for i in range(10):
#     canny = cv2.Canny(img2, 100-5.0*i, 100+15.0*i)
#     cv2.imwrite("processed2/green" + str(i) + ".png", canny)
#
# img3 = img[..., 2]
# for i in range(10):
#     canny = cv2.Canny(img3, 100-5.0*i, 100+15.0*i)
#     cv2.imwrite("processed2/blue" + str(i) + ".png", canny)

# img4 = cv2.imread("images/10kV白泉支线#4-#2.MOV_000000.035.jpg", 0)
# img4 = cv2.GaussianBlur(img4, (3, 3), 1.5)
# for i in range(10):
#     canny = cv2.Canny(img4, 100-5.0*i, 100+15.0*i)
#     cv2.imwrite("processed2/gray" + str(i) + ".png", canny)

# 变成灰度图，自定义比例
# img = 0.35 * img[..., 0] + 0.3 * img[..., 1] + 0.35 * img[..., 2]
# img = img.astype(np.uint8)
# cv2.namedWindow('2', cv2.WINDOW_NORMAL)
# cv2.imshow('2', img)
# cv2.waitKey(0)
#
# for img in imgs_3channel:
#     cv2.namedWindow('houghline', cv2.WINDOW_NORMAL)
#     # img = cv2.Sobel(img, cv2.CV_64F, 1, 1)
#
#     img1 = cv2.GaussianBlur(img, (3, 3), 1.5)
#     edges = cv2.Canny(img1, 50, 300)
#
#     minLineLength = 5000
#     maxLineGap = 10
#     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 200, minLineLength, maxLineGap)
#
#     for x1, y1, x2, y2 in lines[0]:
#         cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
#
#     cv2.imshow("houghline", img)
#     # cv2.namedWindow('green', cv2.WINDOW_NORMAL)
#     # cv2.namedWindow('blue', cv2.WINDOW_NORMAL)
#     # cv2.imshow('green', img[..., 1])
#     # cv2.imshow('blue', img[..., 2])
#
#     cv2.waitKey(0)
#
# img = cv2.imread("images/10kV白泉支线#4-#2.MOV_000000.035.jpg", 0)
# print(img.shape)
#
# cv2.namedWindow('3', cv2.WINDOW_NORMAL)
# cv2.imshow('3', img)
# cv2.waitKey()
# 绿色变黑

# cv2.namedWindow('4', cv2.WINDOW_NORMAL)
# cv2.imshow('4', img[..., 0])
# cv2.waitKey()
#
# cv2.namedWindow('5', cv2.WINDOW_NORMAL)
# cv2.imshow('5', img[..., 1])
# cv2.waitKey()
#
# cv2.namedWindow('6', cv2.WINDOW_NORMAL)
# cv2.imshow('6', img[..., 2])
# cv2.waitKey()

# img = 1.5 * img
# img[img > 255] = 255
# # 数据类型转换
# img = np.around(img)
# img = img.astype(np.uint8)
# cv2.namedWindow('out', cv2.WINDOW_NORMAL)
# cv2.imshow('out', img)
# cv2.waitKey(0)
#
# kernel_size = (3, 3)
# sigma = 1
# img = cv2.GaussianBlur(img, kernel_size, sigma)
# cv2.namedWindow('3', cv2.WINDOW_NORMAL)
# cv2.imshow('3', img)
# cv2.waitKey(0)
# #
# undistorted = []
# for img in test_imgs:
#     cv2.imshow('image', img)
#     cv2.waitKey(0)
# cv2.destroyAllWindows()
# img = utils.cal_undistort(img,object_points,img_points)
# undistorted.append(img)
