import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

capture = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

folder = "Dataset/D"
counter = 0

while True:
    success,img = capture.read()
    hands,img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255 # 0-255 ayarlıyor beyaz ekran 300pikselli
        imgCrop = img[y-offset:y + h+offset, x-offset:x + w+offset]

        imgCropShape = imgCrop.shape
        

        aspectRatio = h/w

        if aspectRatio >1:
            k = imgSize/h
            wCal = math.ceil(k*w) # yukarı yuvarlıyo sayıyı kesirli olmasın diye
            imgResize = cv2.resize(imgCrop,(wCal,imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2) # image centerlamak için
            imgWhite[:,wGap:wCal+wGap] = imgResize# heigth and weight
        else:
            k = imgSize/w
            hCal = math.ceil(k*h) # yukarı yuvarlıyo sayıyı kesirli olmasın diye
            imgResize = cv2.resize(imgCrop,(imgSize,hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize-hCal)/2) # image centerlamak için
            imgWhite[hGap:hCal+hGap,:] = imgResize# heigth and weight


        cv2.imshow("ImageCrop",imgCrop)
        cv2.imshow("ImgWhitee",imgWhite)

    cv2.imshow("Image",img)
    key = cv2.waitKey(1)

    if key == ord("s"):
        counter +=1
        cv2.imwrite(f"{folder}/Image_{time.time()}.jpg",imgWhite)
        print(counter)
        