"""
test getting string of characters of a licence plate
python3 /home/mka/PycharmProjects/Image2Letters/test.py "*.jpg"
"""
import sys

from rekkariDetectionSave import DetectPlate
from filterImage import FilterImage
from filterCharacterRegions import FilterCharacterRegions
from initialCharacterRegions import InitialCharacterRegions
from noQTpicture2rectangle import MouseRectangle, Rectangles
import glob

# image to plate(s)
files=glob.glob(sys.argv[1])
print(files)
if len(files)==0:
    raise FileNotFoundError('no files with search term: '+sys.argv[1])

for file in files:
    # 1 image to possible plates
    # app1 = DetectPlate(imageFileName=file)
    #plates = app1.getNpPlates()
    #app1.showPlates()
    #print(file+' number of plates found '+ str(len(plates)))
    #for plate in plates:
    #    # from a plate image to list of six-rectangles
    #    app2 = FilterImage(npImage=plate)
    #    plate = app2.filterOtsu()
    app3 = FilterCharacterRegions()
    app3.setImageFromFile(imageFileName=file)
    platesWithCharacterRegions = app3.plateChars2CharacterRegions()
    app5 = Rectangles()
    for plate in platesWithCharacterRegions:
        for rectangle in plate:
            app5.set_refPts((rectangle[0], rectangle[1]), (rectangle[0]+rectangle[2],rectangle[1]+rectangle[3]) )
            app5.

    app3.showFinalRectangles()



