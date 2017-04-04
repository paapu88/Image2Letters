"""
INPUT: image file and  list of tuples of six rectangles (many times six tuples possible)
OUTPUT: list of recognized characters and their probablilities and total prbability of a plate

python3
"""

from __future__ import print_function
from tesserocr import PyTessBaseAPI, PSM, Justification
#import cv2
from PIL import Image

class MyTesseract():

    def __init__(self, imageFileName):
        """ gray would be best for tesseract """

        self.img = Image.open(imageFileName)
        #self.rectangles = None
        self.plateStrings = []
        self.charecterConfidences = []
        self.plateConfidences = []

    def getSixPlateCharacters(self, listOfListofRectangles, lang='fin'):
        """ loops over plates and characters in a plate
        gets character of the rectangle and
        confidence for character recognition and
        confidence for overall plate recognition """
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
                        string = string + apiL.GetUTF8Text()
                        confidenceOnCharacter.append(apiL.AllWordConfidences())
                    # digits
                    for [x,y,w,h] in plate[3:6]:
                        apiD.SetRectangle(x, y, w, h)
                        string = string + apiD.GetUTF8Text()
                        confidenceOnCharacter.append(apiD.AllWordConfidences())
                    self.plateStrings.append(string)
                    self.characterConfidences.append(confidenceOnCharacter)
                    self.plateConfidences.append(np.prod(np.array(confidenceOnCharacter)))


    def getSingleCharacter(self, lang='fin'):
        """ for testing: get one character in a single image"""
        with PyTessBaseAPI(psm=PSM.SINGLE_CHAR, lang=lang) as api:
            api.SetVariable("tessedit_char_whitelist","ABCDEFGHIJKLMNOPQRSTUVXYZÅÄÖ0123456789")
            api.SetImage(self.img)
            print("character is: ",api.GetUTF8Text())
            print("by certainty of ", api.AllWordConfidences())


if __name__ == '__main__':
    import sys
    app = MyTesseract(imageFileName=sys.argv[1])
    app.getSingleCharacter()

