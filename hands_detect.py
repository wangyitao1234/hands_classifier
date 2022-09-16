import cv2
import mediapipe as mp
import time

cap =cv2.VideoCapture(0)

mpHands =mp.solutions.hands
hands =mpHands.Hands()
mpDraw =mp.solutions.drawing_utils

pTime =0
cTime =0
color =(0,256,0)

while True:
    success, img= cap.read()
    imgRGB =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results =hands.process(imgRGB)
    print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id,lm in enumerate(handLms.landmark):
                h,w,c =img.shape#高，宽，通道
                c_x,c_y =int(lm.x*w),int(lm.y*h)
                print(id,c_x,c_y)


            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)

    cTime =time.time()
    fps = 1/(cTime-pTime)
    pTime =cTime

    cv2.putText(img,'fps:'+str(int(fps)),(10,70),cv2.FONT_ITALIC,1,color,3)
    cv2.imshow('image',img)
    cv2.waitKey(1)