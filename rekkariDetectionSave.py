"""
returns rectangles containing possible licence plates from a given image
for parameters see
http://docs.opencv.org/3.0-beta/modules/objdetect/doc/cascade_classification.html

python3 rekkariDetectionSave.py 0-751.jpg
(there must be trained classifiier file, assumption is 'rekkari.xml' )
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from glob import glob
import sys

class DetectPlate():

    def __init__(self, trainedHaarFileName='./rekkari.xml', imageFileName=None,
                detectFactor=5, scaleFactor=1.03, minSize=(5,18),
                colorConversion=cv2.COLOR_BGR2GRAY):
        import os

        if not os.path.isfile(trainedHaarFileName):
            raise FileNotFoundError('you need cascade training file for cv2.CascadeClassifier')
        self.trainedHaarFileName = trainedHaarFileName
        self.cascade = cv2.CascadeClassifier(trainedHaarFileName)
        self.imageFileName = imageFileName
        self.detectFactor = detectFactor
        self.scaleFactor = scaleFactor
        self.minSize = minSize
        self.colorConversion = colorConversion
        self.img = None
        self.gray = None

    def image2Plate(self):
        self.img = cv2.imread(self.imageFileName)
        self.gray = cv2.cvtColor(self.img.copy(), self.colorConversion)
        rectangles = self.cascade.detectMultiScale(self.gray, self.scaleFactor, self.detectFactor, minSize=self.minSize)
        plates = []
        for [x,y,w,h] in rectangles:
            roi_gray = self.gray[y:y+h, x:x+w]
            plates.append([x,y,w,h])
        return plates

    def getGray(self):
        return self.gray


if __name__ == '__main__':
    import sys
    app = DetectPlate(imageFileName=sys.argv[1])
    plates = app.image2Plate()
    clone = app.getGray().copy()
    for i, [x,y,w,h] in enumerate(plates):
            print("xywh",x,y,w,h)
            cv2.rectangle(clone,(x,y),(x+w,y+h),(0,255,0),5)
            roi_gray = clone[y:y+h, x:x+w]
            #roi_color = img[y:y+h, x:x+w]
            cv2.imwrite(str(i)+'-'+sys.argv[1]+'.jpg', roi_gray)

    #cv2.imshow('img', clone)


