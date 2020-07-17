import numpy as np
import cv2

# def line_detector(edge, min_reserve):
#     # dilate: expand the pixel
#     kernel = np.ones((3, 3), np.uint8)
#     img1 = cv2.dilate(edge, kernel, iterations=1)
#
#     # return img1
#     lines = cv2.HoughLines(img1, 1, np.pi / 180, min_reserve)
#     lines1 = lines[:, 0, :]  # 提取为为二维
#
#     while lines1.shape[0] > 10 and min_reserve < 1000:
#         min_reserve = min_reserve + 100
#         lines = cv2.HoughLines(img1, 1, np.pi / 180, min_reserve)
#         lines1 = lines[:, 0, :]  # 提取为为二维

# segment = cv2.HoughLinesP(img1, 1, np.pi / 180, min_reserve)
# print(segment)

# 合并直线，对于两条直线差rho<10，theta<10^-2认为同一条直线，取平均值
# line_size = lines1.__len__()
# index = 0
# while index < line_size:
#     compare_index = index + 1
#     while compare_index < line_size:
#         if abs(lines1[index][0] - lines1[compare_index][0]) < 10 and abs(
#                 lines1[index][1] - lines1[compare_index][1]) < 1e-4:
#             lines1[index] = [(lines1[index][0] + lines1[compare_index][0]) / 2,
#                              (lines1[index][1] + lines1[compare_index][1]) / 2]
#             lines1 = np.concatenate((lines1[0:compare_index], lines1[compare_index + 1:]), axis=0)
#             line_size = line_size - 1
#             pass
#         compare_index = compare_index + 1
#     index = index + 1
# print(lines1)

# remove unrelated line
# 去掉明显和其他线不在一个方向的
# time_count = []
# theta = []
#
# for i in range(lines1.__len__()):
#     flag = False
#     index = 0
#
#     for j in range(len(theta)):
#         if abs(theta[j] - lines1[i][1]) < np.pi / 20:
#             flag = True
#             index = j
#             break
#
#     if flag:
#         time_count[index] = time_count[index] + 1
#     else:
#         time_count.append(1)
#         theta.append(lines1[i][1])
#
# for i in range(len(theta)):
#     if time_count[i] == 1:
#         for j in range(lines1.__len__()):
#             if lines1[j][i] == theta[i]:
#                 lines1 = np.concatenate((lines1[0:j], lines1[j + 1:]), axis=0)
#                 i = i - 1
# return lines1
#
# for rho, theta in lines1[:]:
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a * rho
#     y0 = b * rho
#     x1 = int(x0 + 1000 * (-b))
#     y1 = int(y0 + 1000 * (a))
#     x2 = int(x0 - 1000 * (-b))
#     y2 = int(y0 - 1000 * (a))
#     cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
#
# cv2.imwrite("test_processed/1_hough.jpg", img)


def find_line_in_image(img, edge, lines):
    height, width = edge.shape
    points = []
    start_points = []
    end_points = []

    print(lines)

    # for rho, theta in lines[:]:
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a * rho
    #     y0 = b * rho
    #     x1 = int(x0 + 1000 * (-b))
    #     y1 = int(y0 + 1000 * (a))
    #     x2 = int(x0 - 1000 * (-b))
    #     y2 = int(y0 - 1000 * (a))
    #
    #     lx = int(img.shape[1] / 2)
    #     ly = int(y0 + (lx - x0) / (-b) * a)
    #
    #     if lx < 0 or ly < 0:
    #         ly = int(img.shape[0] / 2)
    #         lx = int(x0 + (ly - y0) / a * (-b))
    #
    #     print(str(lx) + " " + str(ly))
    #
    #     print("x1:" + str(x1) + " y1:" + str(y1) + " x2:" + str(x2) + " y2:" + str(y2))
    #     cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 4)
    #
    # cv2.imwrite('line.jpg', img)
    for i in range(lines.__len__()):
        start = False
        blank_pixel = 0
        max_blank_pixel = 0

        if lines[i][1] == np.pi / 2:
            for j in range(width):
                is_exist = False
                for m in range(max(int(lines[i][0]) - 5, 0), min(int(lines[i][0]) + 5, height)):
                    if edge[m, j] == 255:
                        points.append([m, j])
                        is_exist = True
                if is_exist:
                    if not start:
                        start = True
                        start_points.append((j, int(lines[i][0])))
                    else:
                        blank_pixel = 0
                else:
                    if start:
                        blank_pixel = blank_pixel + 1

                        if blank_pixel > max_blank_pixel:
                            # mark as end point
                            start = False
                            end_points.append((j, int(lines[i][0])))
            if start_points.__len__() > end_points.__len__():
                end_points.append((width - 1, int(lines[i][0])))
        elif lines[i][1] == 0:
            for j in range(height):
                is_exist = False
                for m in range(max(int(lines[i][0]) - 5, 0), min(int(lines[i][0]) + 5, width)):
                    if edge[j, m] == 255:
                        points.append([j, m])
                        is_exist = True
                if is_exist:
                    if not start:
                        start = True
                        start_points.append((int(lines[i][0]), j))
                    else:
                        blank_pixel = 0
                else:
                    if start:
                        blank_pixel = blank_pixel + 1

                        if blank_pixel > max_blank_pixel:
                            # mark as end point
                            start = False
                            end_points.append((int(lines[i][0]), j))
            if start_points.__len__() > end_points.__len__():
                end_points.append((int(lines[i][0]), height - 1))
        else:
            a = np.cos(lines[i][1])
            b = np.sin(lines[i][1])
            # 根据cos正负，确定增加方向
            sigh = 1.0
            if b < 0:
                sigh = -1.0

            # 给出开始的点坐标
            x = 0
            m = (x - lines[i][0] * a) / (-sigh * b)
            y = lines[i][0] * b + sigh * m * a

            if y < 0 or y > height:
                y = 0
                m = (y - lines[i][0] * b) / (a * sigh)
                x1 = lines[i][0] * a - sigh * m * b

                y = height - 1
                m = (y - lines[i][0] * b) / (a * sigh)
                x2 = lines[i][0] * a - sigh * m * b

                y = height - 1
                x = x2
                if x1 < x2:
                    y = 0
                    x = x1

                # 找点
            while x < width:
                m = (x - lines[i][0] * a) / (-sigh * b)
                y1 = lines[i][0] * b + sigh * m * a

                if y1 > height + 1 or y1 < -1:
                    break
                else:
                    for m in range(max(int(min(y1, y)) - 5, 0), min(int(max(y1, y)) + 5, height)):
                        is_exist = False
                        for j in range(max(int(x) - 5, 0), min(int(x) + 5, width)):
                            if edge[int(m), int(j)] == 255:
                                points.append([int(m), int(x)])
                                is_exist = True
                        if is_exist:
                            if not start:
                                start = True
                                start_points.append((int(x), int(m)))
                            else:
                                blank_pixel = 0
                        else:
                            if start:
                                blank_pixel = blank_pixel + 1

                                if blank_pixel > max_blank_pixel:
                                    # mark as end point
                                    start = False
                                    end_points.append((int(x), int(m)))
                    x = x + 1
                    y = y1

            if start_points.__len__() > end_points.__len__():
                end_points.append((int(x), int(y1)))

    new_img = np.zeros(edge.shape)
    # print(points)
    # for i in range(len(points)):
    #     new_img[points[i][0], points[i][1]] = 255
    #     img[points[i][0], points[i][1]] = (255, 0, 0)

    # print(start_points)
    # print(end_points)
    print(img.shape)
    for i in range(len(start_points)):
        cv2.line(img, start_points[i], end_points[i], (255, 0, 0), 4)
        # cv2.line(new_img, start_points[i], end_points[i], 255, 2)
    return img, new_img


def fixed_line_detector(edges, min_pixel):
    lines = cv2.HoughLines(edges, 1, np.pi / 180, min_pixel)

    lines1 = lines[:, 0, :]  # 提取为为二维

    print(lines1)

    # 合并直线，对于两条直线差rho<10，theta<10^-2认为同一条直线，取平均值
    # line_size = lines1.__len__()
    # index = 0
    # while index < line_size:
    #     compare_index = index + 1
    #     while compare_index < line_size:
    #         if abs(lines1[index][0] - lines1[compare_index][0]) < 9 and abs(
    #                 lines1[index][1] - lines1[compare_index][1]) < 0.1:
    #             lines1[index] = [(lines1[index][0] + lines1[compare_index][0]) / 2,
    #                              (lines1[index][1] + lines1[compare_index][1]) / 2]
    #             lines1 = np.concatenate((lines1[0:compare_index], lines1[compare_index + 1:]), axis=0)
    #             line_size = line_size - 1
    #         else:
    #             compare_index = compare_index + 1
    #     index = index + 1
    # print(lines1)

    # remove unrelated line
    # 去掉明显和其他线不在一个方向的
    # time_count = []
    # theta = []
    #
    # for i in range(lines1.__len__()):
    #     flag = False
    #     index = 0
    #
    #     for j in range(len(theta)):
    #         if abs(theta[j] - lines1[i][1]) < np.pi / 20:
    #             flag = True
    #             index = j
    #             break
    #
    #     if flag:
    #         time_count[index] = time_count[index] + 1
    #     else:
    #         time_count.append(1)
    #         theta.append(lines1[i][1])
    #
    # for i in range(len(theta)):
    #     if time_count[i] == 1:
    #         for j in range(lines1.__len__()):
    #             if lines1[j][1] == theta[i]:
    #                 lines1 = np.concatenate((lines1[0:j], lines1[j + 1:]), axis=0)
    #                 break

    return lines1


def line_detector(edges, min_pixel=100):
    print(edges.shape)
    # min_pixel = 100
    step = 100

    lines = cv2.HoughLines(edges, 1, np.pi / 180, min_pixel)

    while (lines is None) or (lines.size > 15) or (lines.size < 3) and min_pixel < 1000:
        print(min_pixel)
        if lines is None:
            if min_pixel > 0:
                step = int(step / 2)
            min_pixel = min_pixel - step
        elif lines.size > 10:
            min_pixel = min_pixel + step
        elif lines.size < 3:
            min_pixel = min_pixel + 10

        lines = cv2.HoughLines(edges, 1, np.pi / 180, min_pixel)

    lines1 = lines[:, 0, :]  # 提取为为二维

    # 合并直线，对于两条直线差rho<10，theta<10^-2认为同一条直线，取平均值
    line_size = lines1.__len__()
    index = 0
    while index < line_size:
        compare_index = index + 1
        while compare_index < line_size:
            if abs(lines1[index][0] - lines1[compare_index][0]) < 10 and abs(
                    lines1[index][1] - lines1[compare_index][1]) < 1e-4:
                lines1[index] = [(lines1[index][0] + lines1[compare_index][0]) / 2,
                                 (lines1[index][1] + lines1[compare_index][1]) / 2]
                lines1 = np.concatenate((lines1[0:compare_index], lines1[compare_index + 1:]), axis=0)
                line_size = line_size - 1
                pass
            compare_index = compare_index + 1
        index = index + 1
    print(lines1)

    # remove unrelated line
    # 去掉明显和其他线不在一个方向的
    time_count = []
    theta = []

    for i in range(lines1.__len__()):
        flag = False
        index = 0

        for j in range(len(theta)):
            if abs(theta[j] - lines1[i][1]) < np.pi / 20:
                flag = True
                index = j
                break

        if flag:
            time_count[index] = time_count[index] + 1
        else:
            time_count.append(1)
            theta.append(lines1[i][1])

    for i in range(len(theta)):
        if time_count[i] == 1:
            for j in range(lines1.__len__()):
                if lines1[j][1] == theta[i]:
                    lines1 = np.concatenate((lines1[0:j], lines1[j + 1:]), axis=0)
                    break

    return lines1

