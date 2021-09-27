## -*- coding: utf-8 -*- 
# !pip install tensorflow-gpu==2.0.0-beta1

from datetime import datetime

import tkinter as tk
from   tkinter import *
import tkinter.font as fn
from PIL import Image, ImageTk, ImageDraw
import argparse, ntpath
import os, sys, glob
import numpy as np
import yaml
from tqdm import tqdm
import random
import xml.etree.ElementTree as ET
from box_utils import compute_target
from image_utils import random_patching, horizontal_flip
from functools import partial
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib.pyplot as plt
import face_recognition.api as face_recognition

from anchor import generate_default_boxes
from box_utils import decode, compute_nms
from image_utils import ImageVisualizer
from losses import create_losses
from network import create_ssd
import click, os, glob, re, multiprocessing, sys, itertools, cv2, random, math
import xml.etree.ElementTree as xml_tree
import threading,  time, queue
import pyglet
from playsound import playsound
import contextlib
import tensorflow as tf
import gc


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Global
cancel = False
lock = threading.Lock()
lip_motion = []
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
NUM_CLASSES = 4
colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
face_encoding_list = [None,None,None,None]
is_record = True

win_x = 2560
if len(sys.argv) >= 2 and sys.argv[1] == 'd':
    win_x=0

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
    xml.append("  <folder>carno</folder>")
    xml.append("  <filename>{}</filename>".format(fname))
    xml.append("  <source>")
    xml.append("    <database>carno</database>")
    xml.append("    <annotation>carno</annotation>")
    xml.append("    <image>flickr</image>")
    xml.append("  </source>")
    xml.append("  <size>")
    xml.append("        <width>{}</width>".format(width))
    xml.append("        <height>{}</height>".format(height))
    xml.append("    <depth>3</depth>")
    xml.append("  </size>")
    xml.append("  <segmented>0</segmented>")

    for obj in object_list:
        if obj.parent != None:
            continue
        xml.append("  <object>")
        xml.append("    <name>{}</name>".format(obj.name))
        xml.append("    <pose>Unspecified</pose>")
        xml.append("    <truncated>{}</truncated>".format(obj.truncated))
        xml.append("    <difficult>{}</difficult>".format(obj.difficult))
        xml.append("    <bndbox>")
        xml.append("            <xmin>{}</xmin>".format(obj.xmin))
        xml.append("            <ymin>{}</ymin>".format(obj.ymin))
        xml.append("            <xmax>{}</xmax>".format(obj.xmax))
        xml.append("            <ymax>{}</ymax>".format(obj.ymax))
        xml.append("    </bndbox>")
        part_list = getPartList(obj, object_list)
        for sobj in part_list:
            xml.append("    <part>")
            xml.append("      <name>{}</name>".format(sobj.name))
            xml.append("      <bndbox>")
            xml.append("                <xmin>{}</xmin>".format(sobj.xmin))
            xml.append("                <ymin>{}</ymin>".format(sobj.ymin))
            xml.append("                <xmax>{}</xmax>".format(sobj.xmax))
            xml.append("                <ymax>{}</ymax>".format(sobj.ymax))
            xml.append("      </bndbox>")
            xml.append("    </part>")
        xml.append("  </object>")
    xml.append("</annotation>")

    f = open(path_file_name.replace(".jpg", ".xml"), "w")
    f.write('\n'.join(xml))
    f.close()

def saveImage():
    global org_img, org_img2, boxes, classes, score, save_image_seq
    h = len(org_img)
    w = len(org_img[0])
    cv2.imwrite('save_jpg/' + str(save_image_seq) + '.jpg', org_img)
    cv2.imwrite('save_jpg/' + str(save_image_seq) + '_2.jpg', org_img2)
    object_list = []
    for ind in range(len(classes)):
        b = boxes[ind]
        object_list.append(
            Obj(classes[ind], int(b[0] * w), int(b[1] * h), int(b[2] * w), int(b[3] * h)))
    saveVocXml('save_jpg/' + str(save_image_seq) + '.xml', w, h, object_list)
    save_image_seq += 1

def memoryFace(id):
    global org_img
    no = id[3:4] #기록(1)
    encodng = face_recognition.face_encodings(org_img)
    if len(encodng) > 0:
        face_encoding_list[int(no)] = encodng[0]
        msg(f'정상 !!, 얼굴 등록{no}', 'green')
        mp3_queue.put(f'voice/save{no}.m4a')
    else:
        msg('얼굴 분석실패. ', 'red')
        mp3_queue.put(f'voice/seecamera.m4a')

def findFace():
    global org_img, face_encoding_list
    current_face = face_recognition.face_encodings(org_img)
    if len(current_face) > 0:
        min_result = 100
        min_index  = 0
        for no in range(len(face_encoding_list)):
            memory_face = face_encoding_list[no]
            if isinstance(memory_face, np.ndarray):
                result = face_recognition.face_distance(memory_face, current_face)
                #print(no, 'result', result)
                if min_result > result:
                    min_result = result
                    min_index = no
        if min_index != -1 and min_result < 0.4  :
            msg(f'{min_index}등록 얼굴과 일치합니다. ({int((1-min_result)*100)}%정확도)' , 'green')
            mp3_queue.put(f'voice/same{min_index}.m4a')
            return
        msg('등록된 얼굴을 찾지 못했습니다. ', 'red')
        mp3_queue.put(f'voice/notfound.m4a')
    else:
        msg('카메라를 응시해 주세요. ', 'red')
        mp3_queue.put(f'voice/seecamera.m4a')

def getPartList(obj, object_list):
    part_list = []
    for o in object_list:
        if o.parent == obj:
            part_list.append(o)
    return part_list

def msg(msg_text, color):
    global msg_start_time, status_label
    status_label['text'] = msg_text
    status_label.config(bg=color)
    msg_start_time = time.time()
    status_label.pack()

def parseFace(img):
    try :
        face_landmarks_list = face_recognition.face_landmarks(img)
        #print(face_landmarks_list, len(face_landmarks_list))
        if len(face_landmarks_list) > 0:
            for face_landmarks in face_landmarks_list:
                # 'chin','left_eyebrow','right_eyebrow','nose_bridge','nose_tip','left_eye','right_eye','top_lip','bottom_lip'
                '''
                {'chin': [(454, 81), (457, 106), (462, 129), (468, 153), (477, 175), (492, 194), (511, 209), (534, 220),
                          (558, 223), (581, 219), (602, 209), (620, 195), (634, 176), (640, 153), (641, 129),
                          (642, 106), (642, 83)],
                 'left_eyebrow': [(479, 65), (492, 55), (509, 52), (526, 54), (542, 60)],
                 'right_eyebrow': [(572, 60), (587, 52), (604, 48), (620, 51), (632, 61)],
                 'nose_bridge': [(558, 79), (558, 93), (558, 108), (558, 123)],
                 'nose_tip': [(540, 138), (549, 140), (559, 142), (567, 139), (575, 137)],
                 'left_eye': [(499, 84), (508, 78), (521, 77), (531, 85), (520, 89), (508, 89)],
                 'right_eye': [(581, 83), (592, 74), (604, 74), (614, 80), (605, 85), (593, 86)],
                 'top_lip': [(526, 171), (538, 163), (551, 159), (559, 160), (567, 158), (579, 161), (589, 167),
                             (584, 168), (567, 166), (559, 167), (551, 167), (532, 171)],
                 'bottom_lip': [(589, 167), (581, 178), (569, 183), (561, 185), (552, 185), (539, 181), (526, 171),
                                (532, 171), (552, 173), (560, 173), (568, 171), (584, 168)]}
                '''
                #print(face_landmarks)
                for key in face_landmarks.keys():
                    list = face_landmarks.get(key)
                    for p in list:
                        cv2.circle(img, p, 1, (0,255,0), -1)
                nose_bridge = face_landmarks.get('nose_bridge')
                left_eye = face_landmarks.get('left_eye')
                right_eye = face_landmarks.get('right_eye')
                chin = face_landmarks.get('chin')
                bottom_lip = face_landmarks.get('bottom_lip')
                cv2.circle(img, bottom_lip[0], 2, (0, 0, 255), -1)
                cv2.circle(img, bottom_lip[6], 2, (0, 0, 255), -1)

                r0 = distance(nose_bridge[3], chin[3]) / distance(nose_bridge[3], chin[13])
                r1 = distance(left_eye[0], left_eye[3]) / distance(right_eye[0], right_eye[3])
                return True, r0, r1
    except:
        traceback.print_exc()
    return False, 1, 1

def distance(p1, p2):
    g = np.asarray(p1) - np.asarray(p2)
    return  math.sqrt(math.pow(g[0], 2) + math.pow(g[1], 2))

def isStereoFace(frameL, frameR):
    stat_R, rate0_R, rate1_R = parseFace(frameR)
    stat_L, rate0_L, rate1_L = parseFace(frameL)

    if stat_R and stat_L:
        r0 = round((rate0_R / rate0_L), 5)  # 카메라에 따른 촤우 : 코끝-양쪽 볼 거리 비 달라야함ㄴ
        r1 = round((rate1_R / rate1_L), 5)  # 카메라에 따른 눈크기비  같아야함
        is_real = ((r0 < 0.9 or 1.3 < r0)  and (0.9 < r1 and r1 < 1.2)) or (1.7 < r0 and r1 < 1.4)
        msg =  f'eye-nose:{is_real} R={r0},R={r1}'
        print(msg)
        return True, is_real
    return False, False

def processButton(id):
    global root, image_label, btn_smoney, btn_list, face_count, msg_start_time, status_label, mp3_queue, var_record, is_record
    #print(f'processButton {id} click')

    lock.acquire()

    try:
        if id == 'passwd':
            if face_count > 1:
                msg('고객님 비밀번호가 노출 되지 않도록 조심해주세요', 'red')
                mp3_queue.put('voice/w_passwd.m4a')
            else:
                msg('정상 !!, 감지얼굴 수 1', 'green')
                mp3_queue.put('voice/ok.m4a')
        elif id == 'smoney':
            if usePhone(boxes, classes)  or lipMotionCount() >= 3:
                msg('고객님 혹시 휴대폰 통화중인가요. 보이스피싱이 아닌지 생각해주세요. ', 'red')
                mp3_queue.put('voice/w_call.m4a')
            else:
                msg('정상 !!, 통화 하지 않음 확인', 'green')
                mp3_queue.put('voice/ok_call.m4a')

        elif id == 'omoney':
            if maskedFace(boxes, classes, img300):
                msg('고객 보호를 위해 출금자 신원을 확인합니다. 얼굴을 보여주세요!', 'red')
                mp3_queue.put('voice/w_out.m4a')
            else:
                msg('정상 !! 얼굴확인', 'green')
                mp3_queue.put('voice/ok_face.m4a')

        elif id == 'spoofing':
            face_ind = findMajorFace(boxes, classes)
            if face_ind < 0:
                msg("화면 중앙을 보아주세요.", 'yellow')
                mp3_queue.put('voice/seecenter.m4a')
            else:
                is_in_phote = isMajorFaceRectInPhoto(face_ind, classes, boxes)
                if (is_in_phote):
                    msg(f'사진얼굴 입니다.', 'red')
                    mp3_queue.put('voice/w_pictureface.m4a')
                else:
                    is_findface, is_real = isStereoFace(org_img, org_img2)
                    if not is_findface:
                        msg("화면 중앙을 보아주세요.", 'yellow')
                        mp3_queue.put('voice/seecenter.m4a')
                    elif is_findface and is_real:
                        msg(f'정상 얼굴입니다.', 'green')
                        mp3_queue.put('voice/ok_realface.m4a')
                    else:
                        msg(f'사진얼굴 입니다.', 'red')
                        mp3_queue.put('voice/w_pictureface.m4a')

        elif id == 'record':
            is_record = (var_record.get() == 1)
            if is_record:
                msg('재학습영상 녹화를 시작합니다.', 'yellow')
                mp3_queue.put('voice/record_on.m4a')
            else:
                msg('영상녹화를 중지합니다.', 'green')
                mp3_queue.put('voice/record_off.m4a')

        elif id == 'find':
            findFace()
        elif id[0:2] == '기록':
            memoryFace(id)
    finally:
        lock.release()

def findMajorFace(boxes, classes):
    m_face = [0, 0, 0, 0]
    max_face = 0
    face_ind  = -1
    for i in range(len(classes)):
        b = boxes[i]
        sz = (b[2] - b[0]) * (b[3] - b[1])
        if classes[i] == 2 and max_face < sz:
            max_face = sz
            m_face = b
            face_ind = i
    return face_ind

def findMajorFaceRect(classes, boxes):
    ind = findMajorFace(boxes, classes)
    if ind >= 0:
        b = boxes[ind]
        return b[0], b[1], b[2], b[3]
    return 0, 0, 0, 0

def isMajorFaceRectInPhoto(face_ind, classes, boxes):
    f = boxes[face_ind]
    for i in range(len(classes)):
        if classes[i] == 3:
            p = boxes[i]
            if  f[0] < p[2] and p[0] < f[2] and f[1] < p[3] and p[1] < f[3] :
                return True
    return False

def usePhone(boxes, classes):
    face_ind = findMajorFace(boxes, classes)
    if face_ind < 0:
        return False
    #얼굴 근처 손목 확인
    f = boxes[face_ind]
    for i in range(len(classes)):
        if  classes[i] == 1 :
            h = boxes[i]
            if  f[0] < h[2] and h[0] < f[2] and f[1] < h[3] and h[1] < f[3] :
                return True
    return False

def maskedFace(boxes, classes, img300):
    face_ind = findMajorFace(boxes, classes)
    if face_ind < 0:
        return True
    b = boxes[face_ind]
    sz = len(img300)
    #print('TEST', int(b[0]*sz),int(b[1]*sz),int(b[2]*sz),int(b[3]*sz))

    face_sub_img = img300[int(b[1]*sz):int(b[3]*sz), int(b[0]*sz):int(b[2]*sz)]

    face_landmarks = face_recognition.face_landmarks(face_sub_img)
    if len(face_landmarks) == 0:
         return True
    #print('face_landmarks', type(face_landmarks[0]), face_landmarks[0])
    if 'chin'        in face_landmarks[0] and \
       'nose_bridge' in face_landmarks[0] and \
       'nose_tip'    in face_landmarks[0] and \
       'left_eye'    in face_landmarks[0] and \
       'right_eye'   in face_landmarks[0] and \
       'top_lip'     in face_landmarks[0] and \
       'bottom_lip'  in face_landmarks[0]:
        return False
    #print('face_landmarks_list', 'chin' in face_landmarks)
    return True

def makeFunc(f, x):
  return lambda: f(x)

def quit1():
    global root
    global mp3_queue, mp3_queue_thread
    mp3_queue.put('q')
    root.quit()

def initScr():
    global root, image_label, btn_smoney, btn_omoney, btn_list, status_label, var_record
    root = tk.Tk() #
    root.title("누구나하는 딥러닝 COP")
    #root.geometry(f"1280x960+{win_x}+0")
    #root.geometry(f"1280x960+{win_x}+0")
    root.resizable(width=True, height=True)
    root.bind('<Escape>', lambda e: quit1())
    root.protocol("WM_DELETE_WINDOW", quit1)

    font = fn.Font(family="맑은 고딕", size=18, slant="italic")
    status_label = tk.Label(root, compound=tk.BOTTOM, relief=tk.SUNKEN, font=font, height=2, width=400)
    status_label.pack(side=tk.BOTTOM, anchor=tk.SW)
    buttonPan = tk.Frame(root, height=40)
    buttonPan.pack(side=tk.BOTTOM, anchor=tk.NW)
    leftPan = tk.Frame(root, width=24)
    leftPan.pack(side=tk.RIGHT, anchor=tk.NW)
    var_record = tk.IntVar()
    var_record.set(1)

    btn_height = 4
    btn_width = 22
    btn_passwd = tk.Button(leftPan, text="암호입력\n(비밀번호 엿보기)", font=font, command=makeFunc(processButton, 'passwd'), height=btn_height, width=btn_width)
    btn_passwd.pack(side=tk.TOP)
    btn_smoney = tk.Button(leftPan, text="입금/송금\n(통화여부감지)", font=font, command=makeFunc(processButton, 'smoney'), height=btn_height, width=btn_width)
    btn_smoney.pack(side=tk.TOP)
    btn_omoney = tk.Button(leftPan, text="출금\n(얼굴가림확인)", font=font, command=makeFunc(processButton, 'omoney'), height=btn_height, width=btn_width)
    btn_omoney.pack(side=tk.TOP)
    btn_spoofing = tk.Button(leftPan, text="가짜얼굴분석\nSpoofing", font=font, command=makeFunc(processButton, 'spoofing'), height=btn_height, width=btn_width)
    btn_spoofing.pack(side=tk.TOP)
    btn_memory = tk.Button(leftPan, text="(계좌)\n소유자확인", font=font, command=makeFunc(processButton, 'find'), height=btn_height, width=btn_width)
    btn_memory.pack(side=tk.TOP)
    btn_record = tk.Checkbutton(leftPan,text="재학습영상 자동저장",  variable=var_record, command=makeFunc(processButton, 'record'))
    btn_record.pack(side=tk.TOP)

    btn1_height = 2
    btn1_width = 3
    btn_list = []
    lbl_memory = tk.Label(buttonPan, text='"기록"버튼으로 얼굴 기억후 "소유자확인"으로 비교하세요.', font=font, fg="black", height=btn1_height, width=47)
    lbl_memory.pack(side=tk.LEFT)
    for id in ['기록(3)','기록(2)', '기록(1)', '기록(0)']:
        btn1 = tk.Button(buttonPan, text=id, font=font, height=btn1_height, width=len(id)*2, command=makeFunc(processButton, id))
        btn1.pack(side=tk.RIGHT)
        btn_list.append(btn1)
    image_label = tk.Label(root, compound=tk.LEFT, anchor=tk.NE, relief=tk.RAISED, height=920, width=920)
    image_label.pack(side=tk.TOP)

def predict(imgs, default_boxes):
    timelog("predict  begin" )
    confs, locs = ssd(imgs)
    timelog("predict  ssd" )
    confs = tf.squeeze(confs, 0)
    locs = tf.squeeze(locs, 0)
    confs = tf.math.softmax(confs, axis=-1)
    classes = tf.math.argmax(confs, axis=-1)
    scores = tf.math.reduce_max(confs, axis=-1)
    boxes = decode(default_boxes, locs)

    timelog("predict  decode" )
    out_boxes = []
    out_labels = []
    out_scores = []

    for c in range(1, NUM_CLASSES):
        cls_scores = confs[:, c]

        score_idx = cls_scores > 0.5
        # cls_boxes = tf.boolean_mask(boxes, score_idx)
        # cls_scores = tf.boolean_mask(cls_scores, score_idx)
        cls_boxes = boxes[score_idx]
        cls_scores = cls_scores[score_idx]

        nms_idx = compute_nms(cls_boxes, cls_scores, 0.45, 200)
        cls_boxes = tf.gather(cls_boxes, nms_idx)
        cls_scores = tf.gather(cls_scores, nms_idx)
        cls_labels = [c] * cls_boxes.shape[0]

        out_boxes.append(cls_boxes)
        out_labels.extend(cls_labels)
        out_scores.append(cls_scores)
    timelog("predict  for" )

    out_boxes = tf.concat(out_boxes, axis=0)
    out_scores = tf.concat(out_scores, axis=0)

    boxes = tf.clip_by_value(out_boxes, 0.0, 1.0).numpy()
    classes = np.array(out_labels)
    scores = out_scores.numpy()
    timelog(f"predict  return{boxes}" )
    return boxes, classes, scores

def getGab(p1, p2):
    return int( math.sqrt( (p1[0] - p2[0])* (p1[0] - p2[0]) +  (p1[1] - p2[1]) *  (p1[1] - p2[1]) ))

def addLipMotion(gab):
    global lip_motion
    lip_motion.append(gab)
    if len(lip_motion) >= 35:
        del lip_motion[0]

def lipMotionCount():
    global lip_motion
    cnt = 0
    last = 0
    for curr in  lip_motion:
        if  last > curr:
            last = curr
        elif last + 2 < curr:
            last = curr
            cnt += 1
    return cnt

def addLipOfBigFace(res_img, in_img, b0, b1, b2, b3):
    timelog("addLipOfBigFace begin" )
    gab = -1
    gab = -1
    i_sz = len(in_img)
    o_sz = len(res_img)
    g = o_sz / i_sz
    sub_img = in_img[ int(b1*i_sz) : min(i_sz,int((b3+0.05)*i_sz)),    int(b0*i_sz) : int(b2*i_sz)]
    timelog("face_recognition.face_landmarks imgsz=" + str(len(sub_img) ))
    face_landmarks_list = face_recognition.face_landmarks(sub_img)

    timelog("face_recognition.face_landmarks" )
    for face_landmarks in face_landmarks_list:
        tlip = face_landmarks.get('top_lip')
        blip = face_landmarks.get('bottom_lip')
        tlip1 = [ (int((p[0]+b0*i_sz)*g), int((p[1]+b1*i_sz)*g) ) for p in tlip ]
        blip1 = [ (int((p[0]+b0*i_sz)*g), int((p[1]+b1*i_sz)*g) ) for p in blip ]
        cv2.polylines(res_img, [np.array(tlip1),  np.array(blip1)], True, (0 , 0 , 0), 1 )
        Landscape(res_img, tlip1[9], blip1[9])
        gab = getGab (tlip1[9], blip1[9])
        addLipMotion( gab )
        #print(len (face_landmarks)
    timelog("addLipOfBigFace end" )
    return gab

def Landscape(res_img, p0, p1):
    c = (0, 128, 0)
    l = max(p0[0] - 8, 0)
    r = min(p0[0] + 8, 300)
    cv2.line(res_img, (l, p0[1]) , (r, p0[1]), c, 1)
    cv2.line(res_img, (l, p1[1]) , (r, p1[1]), c, 1)
    cv2.line(res_img, p0, p1, c, 1)

#@#tf.function
def predict1(img1, default_boxes):
    return predict([img1], default_boxes)

def processImage1():
    global org_img,org_img2, img300, boxes, classes, scores, cancel, image_label, face_count, msg_start_time, status_label, text_font, is_first,img1, is_record, cnt

    lock.acquire()
    try:
        timelog("try" )

        if len(status_label['text']) > 0 and time.time() - msg_start_time > 5:
            status_label['text'] = ''
            status_label.config(bg="gray")
        _, org_img = cap.read()
        _, org_img2 = cap2.read()
        timelog("cap.read" )
        img300 = cv2.resize(org_img, (300, 300))
        timelog("cv2.resize" )
        img1 = cv2.cvtColor(img300, cv2.COLOR_BGR2RGB)
        timelog("cv2.cvtColor" )
        img1 = (img1 / 127.0) - 1.0
        timelog("img1 / 127.0" )
        boxes, classes, scores = predict([img1], default_boxes)
        timelog("fpredict1 classes={len(classes)}" )
        #print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'),  boxes, classes, scores, usePhone(boxes, classes))

        # 얼굴에 사각형 표시
        res_img = img300
        for i in range(len(classes)):
            b = [int(b * 300) for b in boxes[i]]
            cv2.rectangle(res_img, (b[0], b[1]), (b[2], min(300,b[3]+10)), colors[classes[i]], 2)

        timelog("cv2.rectangle" )
        face_count = list(classes).count(2)

        # 제일큰얼굴 찿기
        b0, b1, b2, b3 = findMajorFaceRect(classes, boxes)
        b0 = 0
        timelog("bigFace" )
        # 제일큰얼굴에서 입술만 찾기 face_lip recognition
        gab = -1
        if b0 > 0 and False:
            img900 = cv2.resize(org_img, (600, 600))
            gab = addLipOfBigFace(res_img, img900, b0, b1, b2, b3)
        else:
            addLipMotion(0)
        timelog("addLipOfBigFace" )

        # 영상 좌우반전, 확대,
        res_img = cv2.flip(res_img, 1)
        timelog("cv2.flip" )
        res_img = cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR)
        timelog("cv2.cvtColor" )
        res_img = Image.fromarray(res_img)
        timelog("Image.fromarray" )
        res_img = res_img.resize((900, 900))
        timelog("Image.resize 900")

        # 우측상단 통화 여부 입술 움직임 count
        draw = ImageDraw.Draw(res_img)
        draw.text((700,10), '입모양 변화(5초) : ' + str(lipMotionCount()), font=text_font, filee=(255,0, 255))

        # 우측상단 작은이미지
        img300_2 = cv2.resize(org_img2, (150, 150))
        res_img2 = cv2.flip(img300_2, 1)
        res_img.paste( Image.fromarray(cv2.cvtColor(res_img2, cv2.COLOR_RGB2BGR)), (0, 0))

        # Dialog 화면출력
        imgtk = ImageTk.PhotoImage(image=res_img)
        timelog("ImageTk.PhotoImage" )
        image_label.imgtk = imgtk
        image_label.configure(image=imgtk)
        timelog("image_label.configure" )
        cnt += 1
        if is_record:
            img300_2 = cv2.resize(org_img2, (300, 300))
            res_img2 = cv2.flip(img300_2, 1)
        #if cnt % 3 == 0 and face_count > 0 and b0 > 0 and gab >= 0 and is_record:
        #    saveImage()
        if not cancel:
            image_label.after(1, processImage1)
    finally:
        lock.release()
    #gc.collect()

save_image_seq = 0
cnt = 0
def loadsImageSeq():
    global save_image_seq
    save_image_seq = 0
    for f in glob.glob("save_jpg/*.jpg"):
        try:
            _, fname = ntpath.split(f)
            s = fname.replace('.jpg', '')
            if save_image_seq < int(s):
                save_image_seq = int(s)
        except:
            print('maxsave_image_seq err')
    save_image_seq += 1

class Mp3Player(threading.Thread): # Thread Extends
    def __init__(self, mp3_queue):
        threading.Thread.__init__(self)
        self.mp3_queue = mp3_queue # data queue
    def run(self):
        sound= {}
        for s  in glob.glob("voice/*.m4a"):
            sound[s] = pyglet.resource.media(s, streaming=False)

        while True: # thread task
            try :
                mp3_file = self.mp3_queue.get() # queue element get
                if mp3_file == 'q':
                    break
                self.mp3_queue.queue.clear()
                #sound[mp3_file].play()
                playsound(mp3_file)
            except:
                print('Thead ERROR ' + mp3_file)
                time.sleep(3)
        self.mp3_queue.task_done()

def timelog(s):
    #print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'), s)
    None

def timelog2(s):
    #print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'), s)
    None

if __name__ == '__main__':
    #tf.debugging.set_log_device_placement(True)
    fontpath = "font.ttc"
    text_font = ImageFont.truetype(fontpath, 20)

    loadsImageSeq()
    np.set_printoptions(precision=3)
    msg_start_time = time.time()
    with open('config.yml') as f:
        cfg = yaml.load(f)
    try:
        config = cfg['SSD300']
    except AttributeError:
        raise ValueError('Unknown architecture: {}'.format('SSD300'))
    default_boxes = generate_default_boxes(config)
    timelog("create_ssd")
    ssd = create_ssd(NUM_CLASSES, 'ssd300', 'latest', 'checkpoints', 'checkpoints')
    timelog("create_ssd ok")

    mp3_queue = queue.Queue()
    mp3_queue_thread = Mp3Player(mp3_queue)
    mp3_queue_thread.start()

    print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'), "cv2.VideoCapture try...")
    ########################################################################################
    cap  = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(1)
    ########################################################################################
    print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'), "cv2.VideoCapture ok")
    #cap.set(3,640*4)
    #cap.set(3,640*4)
    #cap2.set(4,480*4)
    #cap2.set(4,480*4)


    initScr()
    processImage1()
    root.mainloop()
