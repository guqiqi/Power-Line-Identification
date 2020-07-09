""" Canny Edge Detection is based on the following five steps:

    1. Gaussian filter
    2. Gradient Intensity
    3. Non-maximum suppression
    4. Double threshold
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

    # return the hypothenuse of (Ix, Iy)
    G = np.hypot(Ix, Iy)
    D = np.arctan2(Iy, Ix)
    return G, D


def suppression(img, D, t):
    """ Step 3: Non-maximum suppression(code contains threshold)

    Args:
        img: Numpy ndarray of image to be processed (gradient-intensed image)
        D: Numpy ndarray of gradient directions for each pixel in img

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


def sharp_curve_remove(img, D, min_pixel, t):
    """ Step 3: sharp_curve_remove

    Args:
       img: Numpy ndarray of image to be processed (gradient-intensed image)
        D: Numpy ndarray of gradient directions for each pixel in img
        n: Num of minimal pixels in a line(with same gradient direction)
        min_pixel:
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
                                Z[m, m] = img[m, n]
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


def threshold(img, t, T):
    """ Step 4: Thresholding
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
        'WEAK': np.int32(50),
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


def tracking(img, t, T, strong=255):
    """ Step 5:
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

    Returns:
        final Canny Edge image.
    """

    M, N = img.shape
    for i in range(M):
        for j in range(N):
            if t < img[i, j] < T:
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


# cv2.namedWindow('img', cv2.WINDOW_NORMAL)

img = cv2.imread("test_images/10kV里洋II线内坑支线1-24号.MOV_024942.405.jpg", 0)
# img = cv2.imread("test_image/Snipaste_2020-04-27_17-00-33.png", 0)

# cv2.imshow('img', img)
# cv2.waitKey(0)

img = gaussian_filter(img, 1.5)

# img1 = cv2.Canny(img, 50, 200)
# img1 = img1.astype(np.uint8)
# cv2.imwrite("test_processed/canny库函数.jpg", img1)


img, D = gradient_intensity(img)
# cv2.imshow('img', img)
# cv2.waitKey(0)

img, D = suppression(img, D, 40)
# img4 = img.astype(np.uint8)
#
# cv2.imwrite("test_processed/过滤非极大值-双阈值.jpg", img4)
#
# img3 = tracking(img, 50, 100)
# img3 = img3.astype(np.uint8)
#
# cv2.imwrite("test_processed/手写canny.jpg", img3)
print(D)

img = sharp_curve_remove(img, D, 15, 40)
img = threshold(img, 40, 75)
img1 = tracking(img, 40, 75)
img1 = img1.astype(np.uint8)

cv2.imwrite('t1.jpg', img1)

# cv2.imwrite("test_processed/减少短直线50.jpg", img1)

# img2 = tracking(img, 50)
# img2 = img2.astype(np.uint8)
#
# cv2.imwrite("test_processed/2.jpg", img2)

# 膨胀操作，使得直线连接
# kernel = np.ones((3, 3), np.uint8)
# img1 = cv2.dilate(img1, kernel, iterations=1)


# lines = cv2.HoughLines(img1, 1, np.pi / 180, 500)
# print(lines)
# lines1 = lines[:, 0, :]  # 提取为为二维
# print(lines.shape)
# print(lines1.shape)
# print(lines1)
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
# cv2.imwrite("test_processed/手写canny+膨胀+hough.jpg", img)
