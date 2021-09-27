import cv2, sys

capture = cv2.VideoCapture(int(sys.argv[1]))

while True:
    ret, frame = capture.read()
    print("VideoFrame", frame.shape)
    cv2.imshow("VideoFrame", frame)
    if cv2.waitKey(1) > 0: break

capture.release()
cv2.destroyAllWindows()