import cv2, math
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
                #cv2.circle(img, chin[3], 3, (0, 0, 255), -1)
                #cv2.circle(img, chin[13], 3, (0, 0, 255), -1)
                #return True, np.asarray(nose_bridge[3]), np.asarray(chin[0]), np.asarray(chin[16])
                #return True, np.asarray(chin[8]),np.asarray(nose_bridge[3]), np.asarray(left_eye[0]), np.asarray(right_eye[3])
                r = distance(nose_bridge[3], chin[3]) /  distance(nose_bridge[3], chin[13])

                # A = arccos((b ^ 2 + c ^ 2 - a ^ 2) / (2bc))
                return True, r
    except:
        None
    return False, [0,0], [0,0], [0,0]

captureR = cv2.VideoCapture(0)
captureL = cv2.VideoCapture(1)

b = [4, 40]
while True:
    retR, frameR = captureR.read()
    retL, frameL = captureL.read()

    cv2.circle(frameR, (int(640/2), int(480/2)), 3, (255, 0, 0), -1)
    cv2.circle(frameL, (int(640/2) + b[0], int(480/2)+b[1]), 3, (255, 0, 0), -1)
    stat_R, rate_R  = parseFace(frameR)
    stat_L, rate_L  = parseFace(frameL)

    if stat_R and stat_L:
        r = round((stat_R / rate_L), 5)  # 카메라에 따른 코끝-양쪽 볼 거리 비

        is_real = not (0.9 < r and r < 1.15)
        msg = f'eye-nose:{is_real} R={r} '
        print(r, msg)


    cv2.imshow("VideoFrameR", frameR)
    cv2.imshow("VideoFrameL", frameL)

    if cv2.waitKey(1) > 0: break

captureR.release()
captureL.release()
cv2.destroyAllWindows()