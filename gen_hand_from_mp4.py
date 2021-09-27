import cv2

from PIL import ImageFont, ImageDraw, Image
import numpy
import imutils
import os
import sys

def filter(contours):
    r_contours = []
    x=y=w=h=max_sz=0
    for contour in contours:
        area = cv2.contourArea(contour)
        (x1, y1, w1, h1) = cv2.boundingRect(contour)
        if w1*h1 > max_sz:
            max_sz = w1 * h1
            (x, y, w, h) = (x1, y1, w1, h1)
            r_contours.append(contour)
    return (x, y, w, h), r_contours

def black2dark(c):
    if c[0] == 0:
        c[0] = 1
    if c[1] == 0:
        c[1] = 1
    if c[2] == 0:
        c[2] = 1
    return c

def merge_frame(org_frame, blue_frame):
    h = len(org_frame)
    w = len(org_frame[0])
    ret_frame = numpy.zeros((h,w, 3), numpy.uint8)
    for r in range(h):
        for c in range(w):
            if blue_frame[r][c][2] != 0 :
                ret_frame[r][c] = org_frame[r][c]
            elif org_frame[r][c][0] > 50 and  org_frame[r][c][0] > org_frame[r][c][1] * 1.2 and org_frame[r][c][0] > org_frame[r][c][2] * 1.2:
                ret_frame[r][c] = [0,0, 0]#black2dark(org_frame[r][c])
            else:
                ret_frame[r][c] = [1, 1, 1]
    return ret_frame

def conv(fpath):
    i = 0
    head, fname = os.path.split(fpath)
    cap_i = cv2.VideoCapture(fpath +".mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    seq = 0
    while True:
        try:
            ret, frame = cap_i.read()
            if not ret:
                break
            seq += 1
            if fname.find("20200717_202720") and seq % 10 != 0:
                continue

            frame0 = frame[:, :, :]
            frame1 = frame0[:, :, 2]
            ret, frame2 = cv2.threshold(frame1, 120, 255, cv2.THRESH_BINARY)
            frame3 = cv2.cvtColor(frame2, cv2.COLOR_GRAY2BGR)
            print(fpath, frame2.shape, frame3.shape)
            contours, hierarchy = cv2.findContours(frame2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            frame4 = frame3 & frame
            r, r_contours = filter(contours)
            if r[3] <= 0 or r[2] <= 0:
                continue
            '''
            if r_contours is not None:
                for r_contour in r_contours:
                    epsilon = 0.002 * cv2.arcLength(r_contour, True)
                    approx = cv2.approxPolyDP(r_contour, epsilon, True)
                    cv2.drawContours(frame4, [approx], -1, (0, 256, 0), 0)
            '''
            cv2.rectangle(frame4, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), (256, 0, 0), 2)
            g = 5
            frame_org = frame[r[1]-g:r[1] + r[3]+g, r[0]-g:r[0] + r[2]+g]
            frame5 = frame4[r[1]-g:r[1] + r[3]+g, r[0]-g:r[0] + r[2]+g]
            frame6 = merge_frame(frame_org, frame5)
            cv2.imwrite('data/hand_jpg/'+fname + f"_{i}.jpg", frame6)
            print('data/hand_jpg/'+fname + f"_{i}.jpg")
            i+=1
        except Exception as err:
            print(f"error {err}")
    cap_i.release()
if __name__ == "__main__":
    for filename in sys.argv[1:]:
        print(filename)
        if filename.index('.mp4'):
            conv(filename.replace('.mp4', ''))

cv2.destroyAllWindows()