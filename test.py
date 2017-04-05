"""
test getting string of characters of a licence plate
"""
import sys

from rekkariDetectionSave import DetectPlate
from getCharacterRegions import GetCharacterRegions
from myTesseract import MyTesseract
import glob

# image to plate(s)
files=glob.glob(sys.argv[1])
print(files)

for file in files:
    # 1 image to possible plates
    app1 = DetectPlate(imageFileName=file)
    plates = app1.getNpPlates()
    #print(plates)
    for plate in plates:
        # from a plate image to list of six-rectangles
        app2 = GetCharacterRegions(npImage=plate)
        platesWithCharacterRegions = app2.imageToPlatesWithCharacterRegions()

        # for character recognition, we take original image in gray
        # and tesseract studies rectangles in that original image in gray.
        app3 = MyTesseract()
        app3.setImage(plate)
        #app3.showImage()
        app3.defineSixPlateCharacters(platesWithCharacterRegions)
        print("PLATE, confidence, char conf:", \
              app3.getFinalString(), \
              app3.getFinalPlateConfidence(), \
              app3.getFinalCharacterConfidences() )

