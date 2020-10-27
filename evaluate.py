from yolo import YOLO
from PIL import Image
import numpy as np

def readfile(path):
    with open(path, 'r') as f:
        test_data = f.readlines()

    annotation = []
    for data in test_data:
        anno = {}
        line = data.split(' ', 1)
        anno["image path"] = line[0].replace('hpdb/', 'data/BIWI')
        anno["infor"] = []
        infors = line[1].replace('\n', '')
        infors = infors.split('\t')
        for infor in infors:
            dict_box = {}
            box_infor = [float(x) for x in infor.split(' ')]
            dict_box["box"] = box_infor[:4] #top left rigth bottom
            dict_box["angle"] = box_infor[4:] #yaw pitch roll
            anno["infor"].append(dict_box)
        annotation.append(anno)
    return annotation


def evaluate(yolo, ground_truth):
    pass

if __name__ == "__main__":
    test_path = 'BIWI_test.txt'
    ground_truth = readfile(test_path)
    # print(ground_truth)

    yolo = YOLO()
    evaluate(yolo, ground_truth)
