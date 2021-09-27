import cv2, math
import numpy as np

f = 500
rotXval = 90
rotYval = 90
rotZval = 90
distXval = 500
distYval = 500
distZval = 500

def onFchange(val):
    global f
    f = val
def onRotXChange(val):
    global rotXval
    rotXval = val
def onRotYChange(val):
    global rotYval
    rotYval = val
def onRotZChange(val):
    global rotZval
    rotZval = val
def onDistXChange(val):
    global distXval
    distXval = val
def onDistYChange(val):
    global distYval
    distYval = val
def onDistZChange(val):
    global distZval
    distZval = val

def genImg(src, f, rotXval, rotYval, rotZval, distXval, distYval, distZval):
    dst = np.zeros_like(src)
    h, w = src.shape[:2]

    rotX = (rotXval - 90) * np.pi / 180
    rotY = (rotYval - 90) * np.pi / 180
    rotZ = (rotZval - 90) * np.pi / 180
    distX = distXval - 500
    distY = distYval - 500
    distZ = distZval - 500

    # Camera intrinsic matrix
    K = np.array([[f, 0, w / 2, 0],
                  [0, f, h / 2, 0],
                  [0, 0, 1, 0]])

    # K inverse
    Kinv = np.zeros((4, 3))
    Kinv[:3, :3] = np.linalg.inv(K[:3, :3]) * f
    Kinv[-1, :] = [0, 0, 1]

    # Rotation matrices around the X,Y,Z axis
    RX = np.array([[1, 0, 0, 0],
                   [0, np.cos(rotX), -np.sin(rotX), 0],
                   [0, np.sin(rotX), np.cos(rotX), 0],
                   [0, 0, 0, 1]])

    RY = np.array([[np.cos(rotY), 0, np.sin(rotY), 0],
                   [0, 1, 0, 0],
                   [-np.sin(rotY), 0, np.cos(rotY), 0],
                   [0, 0, 0, 1]])

    RZ = np.array([[np.cos(rotZ), -np.sin(rotZ), 0, 0],
                   [np.sin(rotZ), np.cos(rotZ), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    # Composed rotation matrix with (RX,RY,RZ)
    R = np.linalg.multi_dot([RX, RY, RZ])

    # Translation matrix
    T = np.array([[1, 0, 0, distX],
                  [0, 1, 0, distY],
                  [0, 0, 1, distZ],
                  [0, 0, 0, 1]])

    # Overall homography matrix
    H = np.linalg.multi_dot([K, R, T, Kinv])

    # Apply matrix transformation
    cv2.warpPerspective(src, H, (w, h), dst, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT, 0)

    return dst

if __name__ == '__main__':

    #Read input image, and create output image
    src = cv2.imread('1.jpg')
    src = cv2.resize(src,(640,480))

    #Create user interface with trackbars that will allow to modify the parameters of the transformation
    wndnameP = "WarpPerspective:"
    cv2.namedWindow(wndnameP, 1)

    cv2.createTrackbar("f", wndnameP, f, 1000, onFchange)
    cv2.createTrackbar("Rotation X", wndnameP, rotXval, 180, onRotXChange)
    cv2.createTrackbar("Rotation Y", wndnameP, rotYval, 180, onRotYChange)
    cv2.createTrackbar("Rotation Z", wndnameP, rotZval, 180, onRotZChange)
    cv2.createTrackbar("Distance X", wndnameP, distXval, 1000, onDistXChange)
    cv2.createTrackbar("Distance Y", wndnameP, distYval, 1000, onDistYChange)
    cv2.createTrackbar("Distance Z", wndnameP, distZval, 10000, onDistZChange)
    #Show original image
    cv2.imshow(wndnameP, src)

    k = -1
    while k != 27:
        cam_distance = distZval
        cam_gab = 8
        calc_rotYval = math.atan(cam_gab/max(cam_distance, 1)) * 180
        #print(calc_rotYval)
        dstL = genImg(src, f, rotXval, rotYval, rotZval, distXval, distYval, distZval)
        dstR = genImg(src, f, rotXval, rotYval+calc_rotYval, rotZval, distXval+cam_gab, distYval, distZval)

        # Show the image
        cv2.imshow("camL",  dstL)
        cv2.imshow('camR"', dstR)
        k = cv2.waitKey(1)