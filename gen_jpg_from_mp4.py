import cv2

from PIL import ImageFont, ImageDraw, Image
import numpy
import os
import sys


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
            if seq % 10 != 0:
                continue
            '''
            frame0 = frame[:, :, :]
            frame1 = frame0[:, :, 2]
            ret, frame2 = cv2.threshold(frame1, 120, 255, cv2.THRESH_BINARY)
            frame3 = cv2.cvtColor(frame2, cv2.COLOR_GRAY2BGR)
            print(fpath, frame2.shape, frame3.shape)
            contours, hierarchy = cv2.findContours(frame2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            frame4 = frame3 & frame
            '''
            cv2.imwrite('data/facehand_jpg/'+fname + f"_{i}.jpg", frame)
            print('data/facehand_jpg/'+fname + f"_{i}.jpg")
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