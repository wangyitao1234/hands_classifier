import cv2
import mediapipe as mp
import time
import numpy as np
import math
import time

from cvzone.ClassificationModule import Classifier

class handDetector():
    def __init__(self,mode =False,maxHands =2,complexity =1,detectionCon =0.5,trackCon=0.5):
        self.mode =mode
        self.maxHands =maxHands
        self.complexity =complexity
        self.detectionCon =detectionCon
        self.trackCon =trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self,img,draw =True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allhands=[]
        myHand = {}

        if self.results.multi_hand_landmarks:

            h,w,c =img.shape#高，宽，位深

            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):

                ## lmList
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    xList.append(px)
                    yList.append(py)

                ## bbox
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH

                myHand["bbox"] = bbox

                #point,line
                cz0 = handLms.landmark[0].z
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                # for i in range(21):
                #     cx = int(handLms.landmark[i].x *w)
                #     cy = int(handLms.landmark[i].y *h)
                #     cz = handLms.landmark[i].z
                #
                #     depth_z =cz0-cz
                #     radius =int(6* (1+depth_z))
                #
                #     if i ==0:
                #         cv2.circle(img,(cx,cy),radius*2,(255,0,0),cv2.FILLED)
                #     if i ==8:
                #         cv2.circle(img, (cx, cy), radius * 2, (255, 0, 0), cv2.FILLED)
                #     if i ==1 or i ==5 or i == 9 or i ==13 or i ==17:
                #         cv2.circle(img, (cx, cy), radius, (16, 144, 247), cv2.FILLED)
                #     if i ==2 or i ==6 or i ==10 or i ==14 or i ==18:
                #         cv2.circle(img, (cx, cy), radius, (1, 240, 255), cv2.FILLED)
                #     if i ==3 or i ==7 or i ==11 or i ==15 or i ==19:
                #         cv2.circle(img, (cx, cy), radius, (140, 47, 240), cv2.FILLED)
                #     if i ==4 or i ==8 or i ==12 or i ==16 or i ==20:
                #         cv2.circle(img, (cx, cy), radius, (233, 155, 60), cv2.FILLED)
                #handType
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Right"
                    else:
                        myHand["type"] = "left"

                    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                                  (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                  (255, 0, 255), 2)
                    cv2.putText(img, myHand["type"], (bbox[0] - 60, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 0, 255), 2)
                allhands.append(myHand)

        return img, allhands

    def findPosition(self,img,HandNumber =0):
        Lmlist=[]
        if self.results.multi_hand_landmarks:
            MyHand= self.results.multi_hand_landmarks[HandNumber]

            for id, lm in enumerate(MyHand.landmark):
                h, w, c = img.shape  # 高，宽，通道
                c_x, c_y = int(lm.x * w), int(lm.y * h)
                Lmlist.append([id,c_x,c_y])

        return Lmlist

def main():
    cap = cv2.VideoCapture(0)

    offset =20

    pTime = 0
    cTime = 0
    color = (0, 255, 0)
    imagesize =300

    counter =0

    FilePath ='./DATA/2'

    classifier =Classifier('./model/keras_model.h5','./model/labels.txt')
    labels=['0','1','2']

    detector =handDetector()
    while True:
        success, img = cap.read()#type(img)=ndarray
        #img =cv2.imread("C:/Users/15634/hand.jpg")
        #img = cv2.imread("C:/Users/15634/two_hand.jpg")

        img =cv2.flip(img,1)
        img,myHands=detector.findHands(img,draw=False)
        imgout = img.copy()


        try:
            if myHands:
                myHand =myHands[0]
                x,y,w,h =myHand['bbox']

                imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
                imgWhite = np.ones((imagesize, imagesize, 3), np.uint8) * 255

                imgCropshape = imgCrop.shape

                aspectratio = h/ w
                if aspectratio > 1:
                    k=imagesize/h
                    wCal = math.ceil(k*w)
                    imgResize = cv2.resize(imgCrop, (wCal,imagesize))
                    wGap =math.ceil((imagesize-wCal)/2)
                    imgWhite[:,wGap:wGap+wCal] =imgResize
                    prediction,idx =classifier.getPrediction(imgWhite)
                else:
                    k=imagesize/w
                    hCal = math.ceil(k*h)
                    imgResize = cv2.resize(imgCrop, (imagesize,hCal))
                    hGap =math.ceil((imagesize-hCal)/2)
                    imgWhite[hGap:hGap+hCal,:] =imgResize
                    prediction,idx =classifier.getPrediction(imgWhite)

                print(prediction, idx)
                cv2.rectangle(imgout,(x-20,y-70),(x+70,y-20),(255,0,255),cv2.FILLED)
                cv2.putText(imgout,labels[idx],(x,y-26),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                cv2.imshow('imgCrop', imgCrop)
                cv2.imshow("imageWhite", imgWhite)
        except:
                pass

    # Lmlist = detector.findPosition(img)
    # if len(Lmlist) != 0:
    #     print(Lmlist[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, 'fps:' + str(int(fps)), (10, 70), cv2.FONT_ITALIC, 1, color, 3)
        cv2.imshow('imageout', imgout)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()