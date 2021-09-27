import tensorflow as tf
import cv2,os,sys,argparse, random, yaml
import numpy as np
from tqdm import tqdm
from anchor import generate_default_boxes
from box_utils import decode, compute_nms
from image_utils import ImageVisualizer
from losses import create_losses
from network import create_ssd
from PIL import Image
import xml.etree.ElementTree as ET

from box_utils import compute_target
from image_utils import random_patching, horizontal_flip
from functools import partial
import glob
from matplotlib import pyplot as plt
import PIL.Image

def show(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
def mergeimg(bg, sub_image, pos_h, pos_w, skip):
    for h in range(len(sub_image)):
        for w in range(len(sub_image[0])):
            if sub_image[h][w][0] < skip :
                continue
            if 0 < pos_h + h  and 0 < pos_w + w and pos_h + h < len(bg)  and  pos_w + w< len(bg[0]) :
                bg[pos_h + h ][pos_w + w] = sub_image[h][w]
    return bg
def isHideFace(f_y, f_x, f_y1, f_x1, h_y, h_x, h_y1, h_x1):
    g = int((f_x1 - f_x)/4)
    f_y  += g
    f_x  += g
    f_y1 -= g
    f_x1 -= g
    return  (f_y < h_y and h_y < f_y1 or f_y < h_y1 and h_y1 < f_y1 ) and  (f_x < h_x and h_x < f_x1 or f_x < h_x1 and h_x1 < f_x1)

class Obj:
    def __init__(self, name, xmin, ymin, xmax, ymax, truncated=0, difficult=0, objectBox=None, objectNm=None,
                 objectNmBg=None, parent=None):
        self.name = name
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.truncated = truncated
        self.difficult = difficult
        self.objectBox = objectBox
        self.objectNm = objectNm
        self.objectNmBg = objectNmBg
        self.parent = parent
    def __str__(self):
        return "<{}<{},{} ~ {},{}>_".format(self.name, self.xmin, self.ymin, self.xmax, self.ymax)

def saveVocXml(path_file_name, width, height, object_list):
    fname = os.path.basename(path_file_name)
    xml = []
    xml.append("<annotation>")
    xml.append("    <folder>face</folder>")
    xml.append("    <filename>{}</filename>".format(fname))
    xml.append("    <source>")
    xml.append("        <database>carno</database>")
    xml.append("        <annotation>carno</annotation>")
    xml.append("    </source>")
    xml.append("    <size>")
    xml.append("        <width>{}</width>".format(int(width)))
    xml.append("        <height>{}</height>".format(int(height)))
    xml.append("        <depth>3</depth>")
    xml.append("    </size>")
    xml.append("    <segmented>0</segmented>")

    for obj in object_list:
        if obj.parent != None:
            continue
        xml.append("    <object>")
        xml.append("        <name>{}</name>".format(obj.name))
        xml.append("        <pose>Unspecified</pose>")
        xml.append("        <truncated>{}</truncated>".format(obj.truncated))
        xml.append("        <difficult>{}</difficult>".format(obj.difficult))
        xml.append("        <bndbox>")
        xml.append("            <xmin>{}</xmin>".format((obj.xmin)))
        xml.append("            <ymin>{}</ymin>".format((obj.ymin)))
        xml.append("            <xmax>{}</xmax>".format((obj.xmax)))
        xml.append("            <ymax>{}</ymax>".format((obj.ymax)))
        xml.append("        </bndbox>")
        '''
        part_list = self.getPartList(obj)
        for sobj in part_list:
            xml.append("        <part>")
            xml.append("            <name>{}</name>".format(sobj.name))
            xml.append("            <bndbox>")
            xml.append("                <xmin>{}</xmin>".format((sobj.xmin)))
            xml.append("                <ymin>{}</ymin>".format((sobj.ymin)))
            xml.append("                <xmax>{}</xmax>".format((sobj.xmax)))
            xml.append("                <ymax>{}</ymax>".format((sobj.ymax)))
            xml.append("            </bndbox>")
            xml.append("        </part>")
        '''
        xml.append("    </object>")
    xml.append("</annotation>")
    f = open(path_file_name.replace(".jpg", ".xml"), "w")
    f.write('\n'.join(xml))
    f.close()

bground_list = glob.glob("data/bground/*jpg")
facejpg_list = glob.glob("data/face_jpg/*jpg")
handjpg_list = glob.glob("data/hand_jpg/*jpg")

bground_list = []
for fn in glob.glob("data/bground/*jpg"):
    img = cv2.imread(fn, cv2.IMREAD_COLOR)
    m = min(len(img), len(img[0]))
    crop_img = img[0:m, 0:m]
    bground_list.append(cv2.resize(crop_img, (300, 300)))

    img = cv2.resize(crop_img, (900, 900))
    y = 0
    while (y + 300 <= len(img)):
        x = 0
        while (x + 300 <= len(img[0])):
            crop_img = img[y:y + 300, x:x + 300]
            bground_list.append(crop_img)
            x += 300
        y += 300
facejpg_list = []
for fn in glob.glob("data/face_jpg/*jpg"):
    img = cv2.imread(fn, cv2.IMREAD_COLOR)
    facejpg_list.append(img)

handjpg_list = []
for fn in glob.glob("data/hand_jpg/*jpg"):
    img = cv2.imread(fn, cv2.IMREAD_COLOR)
    handjpg_list.append(img)

print(len(bground_list), len(facejpg_list), len(handjpg_list))


#얼굴 크기 배경 300 pixel중 50 ~ 150 pixel
#손크기 얼굴 크기의 40~60%
#얼굴크기 결정data:
seq = 0
random.shuffle(handjpg_list)

for hand in handjpg_list:
    b_img = bground_list[random.randint(0, len(bground_list) - 1)]
    f_img = facejpg_list[random.randint(0, len(facejpg_list) - 1)]
    b_img = b_img.copy()
    f_w = random.randint(80, 150)
    f_h = int(len(f_img)/len(f_img[0])*f_w)
    # 손크기
    if max(len(hand), len(hand[0])) < min(len(hand), len(hand[0])) * 2:
        r1 = random.uniform(0.4, 0.6)
        h_h = int(f_h  * r1)
        h_w = int(len(hand[0])/len(hand)*h_h)
    else:
        r1 = random.uniform(0.4, 0.5)
        if len(hand) < len(hand[0]):
            h_h = int(f_h * r1)
            h_w = int(len(hand[0])/len(hand)*h_h)
        else:
            h_w = int(f_w  * r1)
            h_h = int(len(hand)/len(hand[0])*h_w)

    f_x =  random.randint(10, 300-f_w)
    f_y =  random.randint(10, 300-f_h)
    fs_img = cv2.resize(f_img, (f_w, f_h))
    b_img = mergeimg(b_img, fs_img, f_y, f_x, -1)
    retry = 0
    if h_w >= 200 or h_h >= 200:
        continue
    while True:
        h_x =  random.randint(10, 300-h_w)
        h_y =  random.randint(10, 300-h_h)
        if not isHideFace(f_y, f_x, f_y+f_h, f_x+f_w, h_y, h_x, h_y+h_h, h_x+h_w):
            break
        retry += 1
        if retry > 100 :
            print('Hand Position ERROR Skip')
            break
    hs_img = cv2.resize(hand, (h_w, h_h))
    b_img = mergeimg(b_img, hs_img, h_y, h_x, 40)
    cv2.imwrite(f"face-voc/trainval/JPEGImages/I{seq}.jpg", b_img)
    print(f"face-voc/trainval/JPEGImages/I{seq}.jpg")

    obj1 = Obj('2', f_x, f_y, f_x+f_w, f_y+f_h)
    obj2 = Obj('1', h_x, h_y, h_x+h_w, h_y+h_h)
    saveVocXml(f"face-voc/trainval/Annotations/I{seq}.xml", 300, 300, [obj1, obj2])
    print(f"face-voc/trainval/Annotations/I{seq}.xml")
    seq += 1