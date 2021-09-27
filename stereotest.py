import cv2, math, sys, traceback
import numpy as np
import face_recognition.api as face_recognition

def distance(p1, p2):
    g = np.asarray(p1) - np.asarray(p2)
    return  math.sqrt(math.pow(g[0], 2) + math.pow(g[1], 2))



def parseFace(img):
    try :
        face_landmarks_list = face_recognition.face_landmarks(img)
        #print(face_landmarks_list, len(face_landmarks_list))
        if len(face_landmarks_list) > 0:
            for face_landmarks in face_landmarks_list:
                # 'chin','left_eyebrow','right_eyebrow','nose_bridge','nose_tip','left_eye','right_eye','top_lip','bottom_lip'
                for key in face_landmarks.keys():
                    list = face_landmarks.get(key)
                    for p in list:
                        cv2.circle(img, p, 1, (0,255,0), -1)
                nose_bridge = face_landmarks.get('nose_bridge')
                chin = face_landmarks.get('chin')
                left_eye = face_landmarks.get('left_eye')
                right_eye = face_landmarks.get('right_eye')
                bottom_lip = face_landmarks.get('bottom_lip')
                cv2.circle(img, nose_bridge[3], 2, (0, 0, 255), -1)
                cv2.circle(img, left_eye[0], 2, (0, 0, 255), -1)
                cv2.circle(img, right_eye[3], 2, (0, 0, 255), -1)
                cv2.circle(img, bottom_lip[0], 2, (0, 0, 255), -1)
                cv2.circle(img, bottom_lip[6], 2, (0, 0, 255), -1)

                cv2.line(img, right_eye[0], right_eye[3], (255, 255, 0), 1)
                cv2.line(img, left_eye[0], left_eye[3], (255, 255, 0), 1)
                cv2.line(img, nose_bridge[3], chin[3], (0, 255, 255), 1)
                cv2.line(img, nose_bridge[3], chin[13],(0, 255, 255), 1)

                r0 = distance(nose_bridge[3], chin[3]) / distance(nose_bridge[3], chin[13])
                r1 = distance(left_eye[0], left_eye[3]) / distance(right_eye[0], right_eye[3])
                return True, r0,  r1
    except:
        traceback.print_exc()
    return False, 1, 1

b = [4, 40]
frameL = cv2.imread(f'{sys.argv[1]}.jpg')
frameR = cv2.imread(f'{sys.argv[1]}_2.jpg')

cv2.circle(frameR, (int(640/2), int(480/2)), 3, (255, 0, 0), -1)
cv2.circle(frameL, (int(640/2) + b[0], int(480/2)+b[1]), 3, (255, 0, 0), -1)
stat_R, rate0_R, rate1_R  = parseFace(frameR)
stat_L, rate0_L, rate1_L  = parseFace(frameL)

if stat_R and stat_L:
    r0 = round((rate0_R / rate0_L), 5)  # 카메라에 따른 촤우 : 코끝-양쪽 볼 거리 비 달라야함ㄴ
    r1 = round((rate1_R / rate1_L), 5)  # 카메라에 따른 눈크기비  같아야함
    is_real = not (0.9 < r0 and r0 < 1.2)
    msg = f'eye-nose:{is_real} R0={r0}  R1={r1}   '
    print(r0, r1,  msg)
    print(r0, r1,  msg)

    cv2.imshow("VideoFrameR", frameR)
    cv2.imshow("VideoFrameL", frameL)
    cv2.waitKey(60000)

cv2.imshow("VideoFrameR", frameR)
cv2.imshow("VideoFrameL", frameL)

