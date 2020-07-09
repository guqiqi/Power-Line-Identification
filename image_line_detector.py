""" Modified Canny Edge Detection is based on the following five steps:

    1. Gaussian filter
    2. Gradient Intensity
    3. Non-maximum suppression
    4. Shape curve remove
    5. Edge tracking

    This module contains these five steps as five separate Python functions.
"""

# Module imports

from utils import round_angle

# Third party imports
from scipy.ndimage.filters import gaussian_filter
from scipy import ndimage
import numpy as np
import cv2


def gs_filter(img, sigma):
    """ Step 1: Gaussian filter

    Args:
        img: Numpy ndarray of image
        sigma: Smoothing parameter

    Returns:
        Numpy ndarray of smoothed image
    """
    if type(img) != np.ndarray:
        raise TypeError('Input image must be of type ndarray.')
    else:
        return gaussian_filter(img, (5, 5), sigma)


def gradient_intensity(img):
    """ Step 2: Find gradients

    Args:
        img: Numpy ndarray of image to be processed (denoised image)

    Returns:
        G: gradient-intensed image
        D: gradient directions
    """

    # Kernel for Gradient in x-direction
    Kx = np.array(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.int32
    )
    # Kernel for Gradient in y-direction
    Ky = np.array(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.int32
    )

    img = img.astype(np.int32)

    # Apply kernels to the image
    Ix = ndimage.filters.correlate(img, Kx)
    Iy = ndimage.filters.correlate(img, Ky)

    cv2.imwrite("x.jpg", Ix)
    cv2.imwrite('y.jpg', Iy)

    # return the hypothenuse of (Ix, Iy)
    G = np.hypot(Ix, Iy)
    D = np.arctan2(Iy, Ix)
    return G, D


def suppression(img, D, t=50):
    """ Step 3: Non-maximum suppression(code contains threshold)

    Args:
        img: Numpy ndarray of image to be processed (gradient-intensed image)
        D: Numpy ndarray of gradient directions for each pixel in img
        t: Minimal gradient to reserve
    Returns:
        ...
    """

    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    DZ = np.zeros((M, N), dtype=np.int32)

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            # find neighbour pixels to visit from the gradient directions
            if abs(img[i, j]) > t:
                where = round_angle(D[i, j])
                try:
                    if where == 0:
                        if (img[i, j] >= img[i, j - 1]) and (img[i, j] >= img[i, j + 1]):
                            Z[i, j] = img[i, j]
                            DZ[i, j] = D[i, j]
                    elif where == 90:
                        if (img[i, j] >= img[i - 1, j]) and (img[i, j] >= img[i + 1, j]):
                            Z[i, j] = img[i, j]
                            DZ[i, j] = D[i, j]
                    elif where == 135:
                        if (img[i, j] >= img[i - 1, j + 1]) and (img[i, j] >= img[i + 1, j - 1]):
                            Z[i, j] = img[i, j]
                            DZ[i, j] = D[i, j]
                    elif where == 45:
                        if (img[i, j] >= img[i + 1, j + 1]) and (img[i, j] >= img[i - 1, j - 1]):
                            Z[i, j] = img[i, j]
                            DZ[i, j] = D[i, j]
                except IndexError as e:
                    """ Todo: Deal with pixels at the image boundaries. """
                    pass
    return Z, DZ


def curve_remove(img, edge, min_reserve):
    G, D = gradient_intensity(img)

    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)

    for i in range(M):
        for j in range(N):
            # find neighbour pixels to visit from the gradient directions
            # find 10 pixels without too much curve
            if edge[i][j] != 0:
                where = round_angle(D[i, j])
                try:
                    if where == 0:
                        max_length = 1
                        m = i - 1
                        while max_length < min_reserve:
                            if m >= 0 and round_angle(D[m, j]) == 0:
                                m = m - 1
                                max_length = max_length + 1
                            else:
                                break

                        if max_length >= min_reserve:
                            while max_length >= 0:
                                Z[m, j] = edge[m, j]
                                m = m + 1
                                max_length = max_length - 1

                    elif where == 90:
                        max_length = 1
                        m = j - 1
                        while max_length < min_reserve:
                            if m >= 0 and round_angle(D[i, m]) == 90:
                                m = m - 1
                                max_length = max_length + 1
                            else:
                                break

                        if max_length >= min_reserve:
                            while max_length >= 0:
                                Z[i, m] = edge[i, m]
                                m = m + 1
                                max_length = max_length - 1

                    elif where == 135:
                        max_length = 1
                        m = i - 1
                        n = j - 1
                        while max_length < min_reserve:
                            if m >= 0 and n >= 0 and round_angle(D[m, n]) == 135:
                                m = m - 1
                                n = n - 1
                                max_length = max_length + 1
                            else:
                                break

                        if max_length >= min_reserve:
                            while max_length >= 0:
                                Z[m, n] = edge[m, n]
                                m = m + 1
                                n = n + 1
                                max_length = max_length - 1

                    elif where == 45:
                        max_length = 1
                        m = i - 1
                        n = j + 1
                        while max_length < min_reserve:
                            if m > 0 and n < N - 1 and round_angle(D[m, m]) == 45:
                                m = m - 1
                                n = n + 1
                                max_length = max_length + 1
                            else:
                                break

                        if max_length >= min_reserve:
                            while max_length >= 0:
                                Z[m, n] = edge[m, n]
                                m = m + 1
                                n = n - 1
                                max_length = max_length - 1
                except IndexError as e:
                    """ Todo: Deal with pixels at the image boundaries. """
                    pass
    return Z


def sharp_curve_remove(img, D, min_pixel, t=50):
    """ Step 4: sharp_curve_remove

    Args:
        img: Numpy ndarray of image to be processed (gradient-intensed image)
        D: Numpy ndarray of gradient directions for each pixel in img
        min_pixel: Num of minimal pixels in a line(with same gradient direction)
        t: Minimal gradient of edge

    Returns:
        ...
    """
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)

    for i in range(M):
        for j in range(N):
            # find neighbour pixels to visit from the gradient directions
            # find 10 pixels without too much curve
            if abs(img[i, j]) > t:
                where = round_angle(D[i, j])
                try:
                    if where == 0:
                        max_length = 1
                        m = i - 1
                        while max_length < min_pixel:
                            if m >= 0 and round_angle(D[m, j]) == 0:
                                m = m - 1
                                max_length = max_length + 1
                            else:
                                break

                        if max_length >= min_pixel:
                            while max_length >= 0:
                                Z[m, j] = img[m, j]
                                m = m + 1
                                max_length = max_length - 1

                    elif where == 90:
                        max_length = 1
                        m = j - 1
                        while max_length < min_pixel:
                            if m >= 0 and round_angle(D[i, m]) == 90:
                                m = m - 1
                                max_length = max_length + 1
                            else:
                                break

                        if max_length >= min_pixel:
                            while max_length >= 0:
                                Z[i, m] = img[i, m]
                                m = m + 1
                                max_length = max_length - 1

                    elif where == 135:
                        max_length = 1
                        m = i - 1
                        n = j - 1
                        while max_length < min_pixel:
                            if m >= 0 and n >= 0 and round_angle(D[m, n]) == 135:
                                m = m - 1
                                n = n - 1
                                max_length = max_length + 1
                            else:
                                break

                        if max_length >= min_pixel:
                            while max_length >= 0:
                                Z[m, n] = img[m, n]
                                m = m + 1
                                n = n + 1
                                max_length = max_length - 1

                    elif where == 45:
                        max_length = 1
                        m = i - 1
                        n = j + 1
                        while max_length < min_pixel:
                            if m > 0 and n < N - 1 and round_angle(D[m, m]) == 45:
                                m = m - 1
                                n = n + 1
                                max_length = max_length + 1
                            else:
                                break

                        if max_length >= min_pixel:
                            while max_length >= 0:
                                Z[m, n] = img[m, n]
                                m = m + 1
                                n = n - 1
                                max_length = max_length - 1
                except IndexError as e:
                    """ Todo: Deal with pixels at the image boundaries. """
                    pass
    return Z


def threshold(img, t=50, T=100):
    """ Step 5: Thresholding
    Iterates through image pixels and marks them as WEAK and STRONG edge
    pixels based on the threshold values.

    Args:
        img: Numpy ndarray of image to be processed (suppressed image)
        t: lower threshold
        T: upper threshold

    Return:
        img: Thresholdes image

    """
    # define gray value of a WEAK and a STRONG pixel
    cf = {
        'WEAK': np.int32(80),
        'STRONG': np.int32(255),
    }

    # get strong pixel indices
    strong_i, strong_j = np.where(img > T)

    # get weak pixel indices
    weak_i, weak_j = np.where((img >= t) & (img <= T))

    # get pixel indices set to be zero
    zero_i, zero_j = np.where(img < t)

    # set values
    img[strong_i, strong_j] = cf.get('STRONG')
    img[weak_i, weak_j] = cf.get('WEAK')
    # img[weak_i, weak_j] = np.int32(0)

    img[zero_i, zero_j] = np.int32(0)

    return img


def tracking(img, weak=80, strong=255):
    """ Step 6:
    Checks if edges marked as weak are connected to strong edges.

    Note that there are better methods (blob analysis) to do this,
    but they are more difficult to understand. This just checks neighbour
    edges.

    Also note that for perfomance reasons you wouldn't do this kind of tracking
    in a seperate loop, you would do it in the loop of the threshold process.
    Since this is an **educational** implementation ment to generate plots
    to help people understand the major steps of the Canny Edge algorithm,
    we exceptionally don't care about performance here.

    Args:
        img: Numpy ndarray of image to be processed (thresholded image)
        weak: Value that was used to mark a weak edge in Step 4
        strong: Value that a strong edge was marked
    Returns:
        final Canny Edge image.
    """

    M, N = img.shape
    for i in range(M):
        for j in range(N):
            if img[i, j] == weak:
                # check if one of the neighbours is strong (=255 by default)
                try:
                    if ((img[i + 1, j] == strong) or (img[i - 1, j] == strong)
                            or (img[i, j + 1] == strong) or (img[i, j - 1] == strong)
                            or (img[i + 1, j + 1] == strong) or (img[i - 1, j - 1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img


def line_break_detector(img, lines):
    for rho, theta in lines[:]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        lx = int(img.shape[1] / 2)
        ly = int(y0 + (lx - x0) / (-b) * a)

        if lx < 0 or ly < 0:
            ly = int(img.shape[0] / 2)
            lx = int(x0 + (ly - y0) / a * (-b))

        print(str(lx) + " " + str(ly))

        print("x1:" + str(x1) + " y1:" + str(y1) + " x2:" + str(x2) + " y2:" + str(y2))
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)

    return False


def edge_detector(img, t, T, max_remove):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('gray.jpg', gray)

    # modified canny edge detector

    gray = gaussian_filter(gray, 1.5)
    cv2.imwrite('gaussian.jpg', gray)

    # calculate gradient matrix and gradient angle matrix
    gradient, D = gradient_intensity(gray)
    cv2.imwrite('sobel.jpg', gradient)
    # select non maximum gradient
    gradient, D = suppression(gradient, D, t)
    cv2.imwrite('suppression.jpg', gradient)

    # remove sharp curve, clear the image
    gradient = sharp_curve_remove(gradient, D, max_remove, t)
    cv2.imwrite('curve_remove.jpg', gradient)

    gradient = threshold(gradient, t, T)

    # tracking edges(figure out the strong edge)
    img1 = tracking(gradient)
    img1 = img1.astype(np.uint8)
    cv2.imwrite('dege.jpg', img1)

    return img1


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
    #                 lines1[index][1] - lines1[compare_index][1]) < 1.0:
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

