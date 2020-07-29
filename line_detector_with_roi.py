import os
import cv2

import enhanced_edge_detector
import image_line_detector
import xml_parser


def get_images_by_dir(dirname):
    img_names = os.listdir(dirname)
    img_paths = [dirname + '/' + img_name for img_name in img_names]

    imgs = [cv2.imread(path) for path in img_paths]

    return imgs, img_names


xml_trees, file_names = xml_parser.xml_tree_by_dir('box')
# tags = xml_parser.get_tags(xml_trees[0])

min_pixels = [
    # 200,
    #
    # 370,
    # 100,
    # 137,
    #
    # 300,
    #
    # 280,

    300,
    200,
    130,

    # 300,
    # 210,
    # 130,

    260,
    300,
    220,
    170,
    270,

    255,
    80,
    175
]

img_num = 0
j = 0
pix_index = 4
for file in file_names:

    if j < pix_index:
        j = j + 1
        continue

    tags = xml_parser.get_tags(xml_trees[j])
    img_f = cv2.imread('new_testset_bbox/' + file + '.jpg')
    for i in range(tags.__len__()):
        print(file + '-' + str(i) + '.jpg')
        img = img_f[int(tags[i]['y1']): int(tags[i]['y2']), int(tags[i]['x1']): int(tags[i]['x2'])]
        # cv2.imwrite("box-0/" + file + '-' + str(i) + '.jpg', img)

        # edge = enhanced_edge_detector.edge_detector(img, file + '.jpg', 40, 75, 5)
        # cv2.imwrite("box-3/" + file + '-' + str(i) + '.jpg', edge)

        edge = cv2.imread('box-3/' + file + '-' + str(i) + '.jpg', 0)

        lines = image_line_detector.fixed_line_detector(edge, min_pixels[img_num])

        # 35kV港隆线16-27号2.MOV_015145.293.jpg
        # lines = [
        #     [-4., 2.9321532],
        #     [-28., 2.760167],
        #     [-38., 2.740167],
        # ]

        edge_more = cv2.imread('box-1/' + file + '-' + str(i) + '.jpg', 0)

        i1, i2 = image_line_detector.find_line_in_image(img, edge_more, lines)

        img_f[int(tags[i]['y1']): int(tags[i]['y2']), int(tags[i]['x1']): int(tags[i]['x2'])] = i1

        # cv2.imwrite('box-n/' + file + '-' + str(i) + '.jpg', i1)
        img_num = img_num + 1

    cv2.imwrite('box-f/' + file + '.jpg', img_f)
    j = j + 1

    # if j > pix_index:
    #     break

#
# img = cv2.imread('new_testset_bbox/10kV兴华线天城支线1-12号.MOV_031157.978.jpg')
#
# imCrop = img[int(tags[0]['y1']): int(tags[0]['y2']), int(tags[0]['x1']): int(tags[0]['x2'])]
#
# img[int(tags[0]['y1']): int(tags[0]['y2']), int(tags[0]['x1']): int(tags[0]['x2'])] = imCrop
#
# cv2.imwrite('t.jpg', img)
