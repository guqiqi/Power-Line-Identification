# Module imports
import time
# Third party imports
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import cv2


def edge_detector(img, t, T):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # modified canny edge detector

    gray = gaussian_filter(gray, 1.5)

    edges = cv2.Canny(gray, t, T)
    edges = edges.astype(np.uint8)
    return edges


def line_detector(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # modified canny edge detector

    gray = gaussian_filter(gray, 1.5)
    edges = cv2.Canny(gray, 50, 100)
    edges = edges.astype(np.uint8)

    cv2.imwrite("test_processed/canny" + str(time.time()) + ".jpg", edges)

    min_pixel = 500
    step = 100

    lines = cv2.HoughLines(edges, 1, np.pi / 180, min_pixel)

    while (lines is None) or (lines.size > 10):
        print(lines)
        if lines is None:
            if min_pixel > 500:
                step = int(step / 2)
            min_pixel = min_pixel - step
        elif lines.size > 10:
            min_pixel = min_pixel + step

        lines = cv2.HoughLines(edges, 1, np.pi / 180, min_pixel)

    lines1 = lines[:, 0, :]  # 提取为为二维
    print(lines1)
    print(lines1.shape)
    print(lines1)
    for rho, theta in lines1[:]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)

    return img


# img = cv2.imread("test_image/10kV白泉支线#4-#2.MOV_000100.061.jpg")
# line_detector(img)