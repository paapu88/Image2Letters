"""
INPUT: image file and  list of tuples of six rectangles (many times six tuples possible)
OUTPUT: list of recognized characters and their probabilities and total probability of a plate


python3 myTesseract.py 0-vkz-825.jpg.tif
"""

from __future__ import print_function
from tesserocr import PyTessBaseAPI, PSM, Justification
import cv2
from PIL import Image



class MyTesseract():

    def __init__(self, imageFileName=None):
        """ gray would be best for tesseract """

        if imageFileName is not None:
            self.img = Image.open(imageFileName)
        #self.rectangles = None
        self.plateStrings = []
        self.charecterConfidences = []
        self.plateConfidences = []
        self.finalString = None   # final plate string
        self.finalPlateConfidence = None  # confidence on final plate string
        self.finalCharacterConfidences = None  # in final plate string, confidence on individual characters

    def setImage(self, image):
        """
        """
        import numpy as np
        import matplotlib.pyplot
        self.img = Image.fromarray(image)


    def showImage(self):
        self.img.show()


    def defineSixPlateCharacters(self, listOfListofRectangles, lang='fin'):
        """ loops over plates and characters in a plate
        gets character of the rectangle and
        confidence for character recognition and
        confidence for overall plate recognition
        gives the six-character which has highest probability
        """
        import numpy as np

        #self.rectangles = listOfListofRectangles
        self.plateStrings = []
        self.characterConfidences = []
        self.plateConfidences = []

        # get letters first
        with PyTessBaseAPI(psm=PSM.SINGLE_CHAR, lang=lang) as apiL:
            with PyTessBaseAPI(psm=PSM.SINGLE_CHAR, lang=lang) as apiD:
                apiL.SetVariable("tessedit_char_whitelist","ABCDEFGHIJKLMNOPQRSTUVXYZÅÄÖ")
                apiD.SetVariable("tessedit_char_whitelist","0123456789")
                apiL.SetImage(self.img)
                apiD.SetImage(self.img)
                for plate in listOfListofRectangles:
                    if len(plate) != 6:
                        raise RuntimeError('only six character plates allowed in getSixPlateCharacters')
                    string=''
                    confidenceOnCharacter = []
                    # alphabets
                    for [x,y,w,h] in plate[0:3]:
                        apiL.SetRectangle(x, y, w, h)
                        #print("rectangle:", x,y,w,h)
                        string = string + apiL.GetUTF8Text().strip()
                        #print("current string: ", apiL.GetUTF8Text())
                        confidenceOnCharacter.append(round(apiL.AllWordConfidences()[0]/100,2))
                    # digits
                    for [x,y,w,h] in plate[3:6]:
                        apiD.SetRectangle(x, y, w, h)
                        string = string + apiD.GetUTF8Text().strip()
                        confidenceOnCharacter.append(round(apiD.AllWordConfidences()[0]/100,2))
                    self.plateStrings.append(string[0:3]+'-'+string[3:6])
                    self.characterConfidences.append(confidenceOnCharacter)
                    #print ("confidence on characters, ", confidenceOnCharacter)
                    self.plateConfidences.append(round(np.prod(np.array(confidenceOnCharacter)),2))
                    #print ("confidence on plate ", self.plateConfidences[-1])
                    #print("PLATE IS: ", self.plateStrings[-1])

        if len(self.plateStrings)> 0:
            #get plate with largest confidence
            sorted_idx = np.argsort(self.plateConfidences)[::-1]
            largest_i = sorted_idx[0]
            self.finalString = self.plateStrings[largest_i]
            self.finalPlateConfidence = self.plateConfidences[largest_i]
            self.finalCharacterConfidences = self.characterConfidences[largest_i]
        #print("PLATE, confidence, char conf:", self.finalString, self.finalPlateConfidence, self.finalCharacterConfidences)


    def getFinalString(self):
        return self.finalString

    def getFinalPlateConfidence(self):
        return self.finalPlateConfidence

    def getFinalCharacterConfidences(self):
        return self.finalCharacterConfidences


    def defineSingleCharacter(self, lang='fin'):
        """ for testing: get one character in a single image"""
        with PyTessBaseAPI(psm=PSM.SINGLE_CHAR, lang=lang) as api:
            api.SetVariable("tessedit_char_whitelist","ABCDEFGHIJKLMNOPQRSTUVXYZÅÄÖ0123456789")
            api.SetImage(self.img)
            print("character is: ",api.GetUTF8Text())
            print("by certainty of ", api.AllWordConfidences())


if __name__ == '__main__':
    import sys
    app = MyTesseract(imageFileName=sys.argv[1])
    app.defineSingleCharacter(lang='fin')

