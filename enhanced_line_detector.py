import cv2
import numpy as np
# Third party imports
from scipy.ndimage.filters import gaussian_filter


def line_break_detector(img, lines):
    # 合并直线，对于两条直线差rho<10，theta<10^-2认为同一条直线，取平均值
    line_size = lines.__len__()
    index = 0
    while index < line_size:
        compare_index = index + 1
        while compare_index < line_size:
            if abs(lines[index][0] - lines[compare_index][0]) < 10 and abs(
                    lines[index][1] - lines[compare_index][1]) < 1e-4:
                lines[index] = [(lines[index][0] + lines[compare_index][0]) / 2,
                                (lines[index][1] + lines[compare_index][1]) / 2]
                del lines[compare_index]
                line_size = line_size - 1
                pass
            compare_index = compare_index + 1
        index = index + 1

    for rho, theta in lines[:]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = x0 + 1000 * (-b)
        y1 = y0 + 1000 * (a)
        x2 = x0 - 1000 * (-b)
        y2 = y0 - 1000 * (a)

        # 找到与图像中间相交的点（lx, ly)
        lx = int(img.shape[1] / 2)
        ly = int(y0 + (lx - x0) / (-b) * a)

        if lx < 0 or ly < 0:
            ly = int(img.shape[0] / 2)
            lx = int(x0 + (ly - y0) / a * (-b))

        print(str(lx) + " " + str(ly))

        # 找到线段对应点周围像素最白的点
        # cv2.line(img, (lx - 5, ly - 5), (lx + 5, ly + 5), (255, 0, 0), 1)

        max_pixel = img[lx][ly].astype(np.int32)

        cv2.imwrite("test_processed/jjj.jpg", img[lx - 5:lx + 5, ly - 5:ly + 5])

        x = lx - 5
        y = ly - 5
        for i in range(10):
            for j in range(10):
                if max_pixel < (img[lx + i - 5][ly + j - 5].astype(np.int32)):
                    x = lx + i - 5
                    y = ly + j - 5
                    max_pixel = img[lx + i - 5][ly + j - 5].astype(np.int32)

        cv2.line(img, (x - 5, y + 5), (x + 5, y - 5), (0, 255, 0), 1)

        print("x1:" + str(x1) + " y1:" + str(y1) + " x2:" + str(x2) + " y2:" + str(y2))
        print("x:" + str(x) + " y:" + str(y))

        # cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
        #
    cv2.imwrite("test_processed/hough.jpg", img)

    return False


img = cv2.imread("test_image/10kV白泉支线#4-#2.MOV_000100.061.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# modified canny edge detector

gray = gaussian_filter(gray, 1.5)
edges = cv2.Canny(gray, 50, 150)
kernel = np.ones((5, 5), np.uint8)
edges = cv2.dilate(edges, kernel, iterations=1)
edges = edges.astype(np.uint8)
cv2.imwrite("test_processed/edge.jpg", edges)
kernel = np.ones((5, 5), np.uint8)
closeEdges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
closeEdges = closeEdges.astype(np.uint8)
cv2.imwrite("test_processed/closeedge.jpg", closeEdges)

lines = [[1.8670000e+03, 5.2359879e-02],
         [1.8450000e+03, 8.7266460e-02],
         [1.8270000e+03, 1.2217305e-01],
         [1.8630000e+03, 5.2359879e-02]]

# line_break_detector(closeEdges, lines)
