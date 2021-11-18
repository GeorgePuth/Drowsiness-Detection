from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    EAR = (A + B) / (2.0 * C)

    return EAR


def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    EAR = (leftEAR + rightEAR) / 2.0
    return (EAR, leftEye, rightEye)



def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance


print("-> Loading the predictor and detector...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
print("-> Starting Video Stream")
cap = cv2.VideoCapture(0)


EYE_AR_THRESH = 0.28
EYE_AR_CONSEC_FRAMES = 3
YAWN_THRESH = 20
# initialize the frame counters and the total number of blinks
COUNTER = 0
yawns = 0
counter = 0
TOTAL = 0
Total = 0
blip = 0
flip = 0
Wr = 0
Wt = 0
Wb = 0
Wy = 0
framerate = 0

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:

        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        eye = final_ear(shape)
        EAR = eye[0]
        leftEye = eye[1]
        rightEye = eye[2]

        distance = lip_distance(shape)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)


        if EAR > 0:
            framerate += 1


        if EAR < EYE_AR_THRESH:
            COUNTER += 1
            blip += 1


        else:

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1

            COUNTER = 0

        if (distance > YAWN_THRESH):
            yawns += 1



        else:

            if yawns >= 10:
                Total += 1

            yawns = 0

        if framerate <= 900 :
            if blip > 270 :
                Wr = 1
            if framerate <= 900 and TOTAL > 25:
                Wb = 1
            if COUNTER > 30 and distance < 15:
                Wt = 1
            if Total >= 2:
                Wy = 1
            T = Wr + Wb + Wt + Wy
            if T >= 2.0:
                cv2.putText(frame, "DROWSY", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                print(Wr,Wb,Wt,Wy)
            else:
                print(Wr,Wb,Wt,Wy)
        else:
            #totalframeswitheyeandfacedetected
            framerate = 0
            Wr = 0
            Wb = 0
            Wt = 0
            Wy = 0
            #eyeclosedframes
            blip = 0
            #yawn
            Total = 0
            #blinks
            TOTAL = 0
            T = 0
        cv2.putText(frame, "EAR: {:.2f}".format(EAR), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "Yawn: {}".format(Total), (40, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "Frames: {}".format(framerate), (30, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Are you Sleepy", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()





