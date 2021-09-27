import cv2

captureR = cv2.VideoCapture(0)
captureL = cv2.VideoCapture(1)

while True:
    retR, frameR = captureR.read()
    retL, frameL = captureL.read()
    cv2.imshow("VideoFrameR", frameR)
    cv2.imshow("VideoFrameL", frameL)
    if cv2.waitKey(1) > 0: break

captureR.release()
captureL.release()
cv2.destroyAllWindows()