"""
test getting string of characters of a licence plate
"""
import sys

from rekkariDetectionSave import DetectPlate
from getCharacterRegions import GetCharacterRegions
from myTesseract import MyTesseract

# image to plate(s)
app1 = DetectPlate(imageFileName=sys.argv[1])
app1.image2PlateNumpyArrays()  # get plate(s) as array of numpy array(s)
#app1.image2Plate() # for debugging only
#app1.showPlates()  # for debuggin, press 'q'
#plate to regtangles
plates = app1.getNpPlates()
#print(plates)
for plate in plates:
    app2 = GetCharacterRegions(npImage=plate)
    app2.getInitialRegions()
    #app2.showIntermediateRectangles()
    app2.checkHeight()
    #print("11: all RECTANGLEs ",app2.regions)
    app2.checkWidth()
    #print("22: all RECTANGLEs ",app2.regions)
    app2.checkArea()
    #print("33: all RECTANGLEs ",app2.regions)
    app2.checkSameness()
    #print("44: all RECTANGLEs ",app2.regions)
    app2.determineSetsOfSix()
    #print("AFTER SETS OF SIX: ", app2.getCurrentSixLists())
    #app2.showIntermediateRectangles()
    app2.sortSetsAndToList()
    print(app2.getCurrentSixLists())
    app2.checkSixXcloseness()
    print("SHowing graphics window, final results")
    app2.showFinalRectangles()
    print("FINAL:",app2.getFinalSixLists())

