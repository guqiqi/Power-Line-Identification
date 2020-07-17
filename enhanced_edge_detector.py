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
    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)

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
    DZ = np.zeros((M, N), dtype=np.float32)

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


def sharp_curve_remove(img, D, min_pixel, t=50):
    """ Step 4: remove sharp curve

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
                            if m > 0 and n < N - 1 and round_angle(D[m, n]) == 45:
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


def edge_detector(img, t, T, max_remove):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite('gray.jpg', gray)

    # modified canny edge detector
    gray = gaussian_filter(gray, 1.5)
    # cv2.imwrite('gaussian.jpg', gray)

    # calculate gradient matrix and gradient angle matrix
    gradient, D = gradient_intensity(gray)
    # cv2.imwrite('sobel.jpg', gradient)

    # select non maximum gradient
    gradient, D = suppression(gradient, D, t)
    # cv2.imwrite('suppression.jpg', gradient)

    # direction
    # M, N = gradient.shape
    # Z = np.zeros((M, N, 3), dtype=np.int32)
    # for i in range(M):
    #     for j in range(N):
    #         if D[i, j] != 0:
    #             where = round_angle(D[i, j])
    #
    #             if where == 0:
    #                 Z[i, j] = [255, 0, 0]
    #             elif where == 45:
    #                 Z[i, j] = [0, 255, 0]
    #             elif where == 90:
    #                 Z[i, j] = [0, 0, 255]
    #             else:
    #                 Z[i, j] = [100, 100, 100]
    # cv2.imwrite('dir.jpg', Z)
    # np.savetxt(fname="data.csv", X=M, fmt="%.4e", delimiter=",")

    # remove sharp curve, clear the image
    gradient = sharp_curve_remove(gradient, D, max_remove, t)
    # cv2.imwrite('curve_remove.jpg', gradient)

    gradient = threshold(gradient, t, T)

    # tracking edges(figure out the strong edge)
    img1 = tracking(gradient)
    img1 = img1.astype(np.uint8)
    # cv2.imwrite('edge.jpg', img1)

    return img1
