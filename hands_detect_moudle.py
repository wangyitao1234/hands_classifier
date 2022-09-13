import cv2
import mediapipe as mp
import time

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
        #print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self,img,HandNumber =0,draw =True):
        Lmlist=[]
        if self.results.multi_hand_landmarks:
            MyHand= self.results.multi_hand_landmarks[HandNumber]
            for id, lm in enumerate(MyHand.landmark):
                h, w, c = img.shape  # 高，宽，通道
                c_x, c_y = int(lm.x * w), int(lm.y * h)
                Lmlist.append([id,c_x,c_y])
                if draw:
                    cv2.circle(img,(c_x,c_y),15,(255,0,255),cv2.FILLED)

        return Lmlist

def main():
    cap = cv2.VideoCapture(0)

    pTime = 0
    cTime = 0
    color = (0, 256, 0)

    detector =handDetector()

    while True:
        success, img = cap.read()

        img =detector.findHands(img)
        Lmlist =detector.findPosition(img)
        if len(Lmlist)!=0:
            print(Lmlist)



        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, 'fps:' + str(int(fps)), (10, 70), cv2.FONT_ITALIC, 1, color, 3)
        cv2.imshow('image', img)
        cv2.waitKey(1)



if __name__ == '__main__':
    main()