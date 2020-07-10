import os
import xml.etree.ElementTree as ET


def xml_tree_by_dir(dirname):
    file_names = []
    for file in os.listdir(dirname):
        if os.path.splitext(file)[1] == '.xml':
            file_names.append(file)

    xml_trees = []
    for file in file_names:
        xml_trees.append(ET.parse(dirname + '/' + file))
    return xml_trees


def get_tags(tree):
    root = tree.getroot()
    tags = []

    for i in range(6, len(root)):
        obj = root[i]
        name = obj[0].text
        box = obj[4]
        x1 = box[0].text
        y1 = box[1].text
        x2 = box[2].text
        y2 = box[3].text

        tag = 0
        if name == 'DNG plants':
            tag = 1

        tags.append({'tag': tag, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})

    return tags


xml_trees = xml_tree_by_dir('dataset-training')
for tree in xml_trees:
    tags = get_tags(tree)
    print(tags[0]['x1'])
