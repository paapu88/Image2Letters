"""
test getting string of characters of a licence plate

cd  /home/mka/PycharmProjects/Image2Letters/test.py
python3 /home/mka/PycharmProjects/Image2Letters/test.py "*.jpg"
"""
import sys

from rekkariDetectionSave import DetectPlate
from filterImage import FilterImage
from filterCharacterRegions import FilterCharacterRegions
from initialCharacterRegions import InitialCharacterRegions
from myTesseract import MyTesseract
import glob

# image to plate(s)
files=glob.glob(sys.argv[1])
print(files)
if len(files)==0:
    raise FileNotFoundError('no files with search term: '+sys.argv[1])

for file in files:
    # 1 image to possible plates
    app1 = DetectPlate(imageFileName=file)
    plates = app1.getNpPlates()
    app1.showPlates()
    print(file+' number of plates found '+ str(len(plates)))

    for plate in plates:
        # from a plate image to list of six-rectangles
        #app2 = FilterImage(npImage=plate)
        #plate = app2.filterOtsu()
        app3 = FilterCharacterRegions(npImage=plate)
        platesWithCharacterRegions = app3.imageToPlatesWithCharacterRegions()

        app5 = MyTesseract()
        app5.setImage(plate)
        #app3.showImage()
        app5.defineSixPlateCharacters(platesWithCharacterRegions,lang='eng')
        print("PLATE, confidence, char conf:", \
              app5.getFinalString(), \
              app5.getFinalPlateConfidence(), \
              app5.getFinalCharacterConfidences() )

