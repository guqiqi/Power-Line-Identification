import image_line_detector
import line_detector_lib
import cv2

img_name = 'test_images/10kV青洋支线#23-#24.MOV_001827.132.jpg'
img = cv2.imread(img_name)

# edge = image_line_detector.edge_detector(img, 40, 75, 5)
# cv2.imwrite('40-75-5.jpg', edge)
#
# edge = image_line_detector.edge_detector(img, 40, 75, 10)
# cv2.imwrite('40-75-10.jpg', edge)
#
# edge = image_line_detector.edge_detector(img, 40, 75, 15)
# cv2.imwrite('40-75-15.jpg', edge)
#
# edge = image_line_detector.edge_detector(img, 50, 100, 10)
# cv2.imwrite('50-100-10.jpg', edge)
#
# edge = image_line_detector.edge_detector(img, 50, 100, 15)
# cv2.imwrite('50-100-15.jpg', edge)

edge = line_detector_lib.edge_detector(img, 40, 75)
cv2.imwrite('40-75.jpg', edge)

# edge = line_detector_lib.edge_detector(img, 50, 100)
# cv2.imwrite('50-100.jpg', edge)
#
# edge = line_detector_lib.edge_detector(img, 50, 150)
# cv2.imwrite('50-150.jpg', edge)
#
# edge = line_detector_lib.edge_detector(img, 75, 150)
# cv2.imwrite('75-150.jpg', edge)
#
# edge = line_detector_lib.edge_detector(img, 75, 225)
# cv2.imwrite('75-225.jpg', edge)
#
# edge = line_detector_lib.edge_detector(img, 80, 160)
# cv2.imwrite('80-160.jpg', edge)
#
# edge = line_detector_lib.edge_detector(img, 90, 180)
# cv2.imwrite('90-180.jpg', edge)
#
# edge = line_detector_lib.edge_detector(img, 100, 200)
# cv2.imwrite('100-200.jpg', edge)
