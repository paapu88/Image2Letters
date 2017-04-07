"""
python3 getCharacterRegions.py image.jpq


Inherits: InitialCharacterRegions which containts methods that produce lots of initial rectangles

INPUT: lots of rectangles
OUTPUT: list of list of rectangles that are suitable for a licence plate
"""

import cv2
import sys
from matplotlib import pyplot as plt
import numpy as np
from initialCharacterRegions import InitialCharacterRegions


class FilterCharacterRegions(InitialCharacterRegions):
    """
    from one image representing a possible plate get the rectangles in which letters/number are located
    INPUT: gray scale image as numpy array
    OUTPUT: list of possible plates in this image
            each possible plate is a list of six rectangles
    """

    def __init__(self, npImage=None):
        # give the initial rectangle produces the current image
        InitialCharacterRegions.__init__(self, npImage)

        self.img = npImage  # image as numpy array
        self.mser = cv2.MSER_create(_max_variation=10)
        self.regions = None
        if npImage is not None:
            self.imageY = self.img.shape[0]
            self.imageX = self.img.shape[1]
        self.listOfSixSets = []   # list of sets, each set has 6 rectangles
        self.listOfSixLists = None   # list of lists, each set has 6 rectangles
        #print("x y : ",self.imageX, self.imageY)


    def setImageFromFile(self, imageFileName, colorConversion=cv2.COLOR_BGR2GRAY):
        """ for debuggin image can be read from file also"""
        self.img = cv2.imread(imageFileName)
        self.img = cv2.cvtColor(self.img, colorConversion)
        self.imageY = self.img.shape[0]
        self.imageX = self.img.shape[1]


    def getClone(self):
        return self.img.copy()

    def checkHeight(self, minHeight=0.2):
        """ to short letters are killed"""
        ok = []
        for (x,y,w,h) in self.regions:
            if h > minHeight * self.imageY:
                ok.append((x,y,w,h))
        self.regions = ok

    def checkWidth(self, minWidth=0.02, maxWidth=0.16):
        """ to broad or narrow letters are killed"""
        ok = []
        for (x,y,w,h) in self.regions:
            if (w > minWidth * self.imageX) and (w < maxWidth * self.imageX):
                ok.append((x,y,w,h))
        self.regions = ok

    def checkArea(self, minArea=0.005, maxArea=0.2):
        """letters with too small/large area are killed"""
        ok = []
        imageArea = self.imageX * self.imageY
        for (x,y,w,h) in self.regions:
            area=w*h
            if (area > minArea * imageArea) and (area < maxArea * imageArea):
                ok.append((x,y,w,h))
        self.regions = ok

    def checkSameness(self, toleranceInPixel=None):
        """ if we have almost the same rectange, kill one of them
            the remaining rectangle is the bigger one.
        """
        if toleranceInPixel is None:
            toleranceInPixel = int(round((self.imageX / 7) / 5))  # default tolerance one fifth of one character space
        deletes=[]
        # we must change list of tuples to list of lists in order to modify an individual rectangle
        # (tuples are needed for items in sets)
        regionsAsLists = list(map(list, self.regions))
        for i, region_i in enumerate(self.regions):
            delete_i = False
            #print("A", i, region_i,len(self.regions))
            start=i+1
            for j in range(start, len(self.regions)):
                #print("B",j,self.regions[j])
                [x1,y1,w1,h1] = region_i
                [x2,y2,w2,h2] = self.regions[j]
                if ((abs(x2-x1) < toleranceInPixel) and \
                    (abs(y2-y1) < toleranceInPixel) and \
                            ((w1*h1)<(w2*h2))):
                    delete_i = True
            deletes.append(delete_i)
        ok = []
        for delete, region in zip(deletes, regionsAsLists):
            if not delete:
                ok.append(tuple(region))
        self.regions = ok

    def determineSetsOfSix(self, heightCriterium=0.12):
        """ get all possible 6-sets of letters that are close each other in height"""
        import itertools
        self.listOfSixSets = []
        mymin=1-heightCriterium
        mymax=1+heightCriterium
        #print("regions: ", self.regions)
        for i in range(len(self.regions)):
            heightI=self.regions[i][3]
            sixSet=set([self.regions[i]])
            #print('ini:', sixSet)
            for j in range(len(self.regions)):
                heightJ=self.regions[j][3]
                if (mymin < heightJ/heightI) and (mymax > heightJ/heightI):
                    sixSet.add(self.regions[j])
                    #print(i, j, heightJ / heightI)
            # check that number of rectangles is six and this set of rectangles is new
            #print("I, LEN SET",i, len(sixSet))
            if len(sixSet) == 6 and not(sixSet in self.listOfSixSets):
                #print("sixset: ", sixSet)
                self.listOfSixSets.append(sixSet)
                # if we have a longer set, take all 6-permutations..
            elif len(sixSet) > 6:
                #print("adding >6 set:", sixSet)
                myFixedSet = frozenset(sixSet)
                sixSetTrials= itertools.permutations(myFixedSet, 6)
                #print("sixSetTrials: ",sixSetTrials)
                for addSet in sixSetTrials:
                    if not(set(addSet) in self.listOfSixSets):
                        self.listOfSixSets.append(set(addSet))


    def sortSetsAndToList(self):
        """sort charactar regions in each plate by x coordinate"""
        import numpy as np
        ok = []
        self.listOfSixLists = []

        for currentSet in self.listOfSixSets:
            mykeys = []
            listCurrentSet = list(currentSet)
            for rectangle in listCurrentSet:
                mykeys.append(rectangle[0])
            np_mykeys=np.array(mykeys)
            sorted_idx = np.argsort(np_mykeys)
            #listCurrentSet = listCurrentSet[sorted_idx]
            #self.listOfSixLists.append(list(currentSet)[sorted_idx])
            #print("SORT:",sorted_idx)
            sorted=[]
            for isort in sorted_idx:
                sorted.append(listCurrentSet[isort])

            self.listOfSixLists.append(sorted)

    def checkSixXcloseness(self, minFraction=0.25, maxFraction=0.7):
        """check that subsequent rectangles are close/far enought in x-direction
            if NOT remove the 6-rectangle
            we compare the difference in x-direction of the characters
            to the avereage height of the characters"""
        import numpy as np

        deletePlate = []
        for plate in self.listOfSixLists:
            averageHeight=np.average(np.asarray(plate[:][3]))
            #print("AVERAGE Height:", averageHeight, plate)
            if ((plate[1][0] - plate[0][0]) < (minFraction * averageHeight) or \
                (plate[2][0] - plate[1][0]) < (minFraction * averageHeight) or \
                (plate[3][0] - plate[2][0]) < (minFraction * averageHeight) or \
                (plate[4][0] - plate[3][0]) < (minFraction * averageHeight) or \
                (plate[5][0] - plate[4][0]) < (minFraction * averageHeight) or \
                (plate[1][0] - plate[0][0]) > (maxFraction * averageHeight) or \
                (plate[2][0] - plate[1][0]) > (maxFraction * averageHeight) or \
                #(plate[3][0] - plate[2][0]) > (maxFraction*averageHeight) or \ not for '-'
                (plate[4][0] - plate[3][0]) > (maxFraction * averageHeight) or \
                (plate[5][0] - plate[4][0]) > (maxFraction * averageHeight)):
                deletePlate.append(True)
            else:
                deletePlate.append(False)
        accepted = []
        for i, delete in enumerate(deletePlate):
            if not delete:
                accepted.append(self.listOfSixLists[i])
        self.listOfSixLists = accepted

    def getCurrentSixLists(self):
        """ current candidates for character regions"""
        #print("current list of list(s)/set(s)")
        if self.listOfSixLists is None:
            return self.listOfSixSets
        else:
            return self.listOfSixLists

    def writeIntermediateRectangles(self):
        """ write all current rectangles to disk individually """
        clone = self.getClone()
        for i, (x,y,w,h) in enumerate(self.regions):
            roi_gray = clone[y:y+h, x:x+w]
            cv2.imwrite(str(i)+'-'+sys.argv[1]+'.tif', roi_gray)


    def showIntermediateRectangles(self):
        """ show image with current rectangles on it"""

        clone = self.getClone()

        for (x,y,w,h) in self.regions:
            cv2.rectangle(clone,(x,y),(x+w,y+h),(255,255,255),1)
        plt.imshow(clone, cmap = 'gray', interpolation = 'bicubic')
        #plt.imshow(clone)
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()

        #cv2.imshow('clone', clone)
        #while(cv2.waitKey()!=ord('q')):
        #    continue

    def writeFinalRectangles(self):
        """ write final rectangles to disk individually """
        clone = self.getClone()
        i=0
        for candidatePlate in self.listOfSixLists:
            for (x,y,w,h) in candidatePlate:
                roi_gray = clone[y:y+h, x:x+w]
                cv2.imwrite(str(i)+'-'+sys.argv[1]+'.tif', roi_gray)
                i=i+1

    def showFinalRectangles(self):
        """ show image with final rectangles on it"""
        clone = self.getClone()
        for candidatePlate in self.listOfSixLists:
            for (x,y,w,h) in candidatePlate:
                cv2.rectangle(clone,(x,y),(x+w,y+h),(0,255,0),5)
        cv2.imshow('clone', clone)
        while(cv2.waitKey()!=ord('q')):
            continue

    def getFinalListSixLists(self):
        """ give Final result of character regions for a plate ordered from left to right"""
        return self.listOfSixLists

    def imageToPlatesWithCharacterRegions(self):
        """from a given numpy array representing the image possibly containing licence plate(s),
        returns a list of possible plates
                each possible plate is a list of six rectangles"""
        self.getInitialRegionsMser()
        self.showIntermediateRectangles()
        self.checkHeight()
        self.checkWidth()
        self.checkArea()
        self.checkSameness()
        self.determineSetsOfSix()
        self.sortSetsAndToList()
        self.checkSixXcloseness()
        #self.writeFinalRectangles()
        return self.getFinalListSixLists()



if __name__ == '__main__':
    import sys
    app = FilterCharacterRegions()
    app.setImageFromFile(imageFileName=sys.argv[1])
    app.cleanImage()
    #app.getInitialRegionsMser()
    app.getInitialRegionsByContours()
    app.showIntermediateRectangles()
    sys.exit()
    app.checkHeight()
    app.checkWidth()
    app.checkArea()
    print("all RECTANGLEs ",app.regions)
    print("end of all RECTANGLES")
    app.checkSameness()
    print("22: all RECTANGLEs ",app.regions)
    print("22: end of all RECTANGLES")
    #sys.exit()
    app.determineSetsOfSix()

    print("LIST OF RECTANGLE CANDIDATES",app.listOfSixSets)
    print("end of candidates")
    app.sortSetsAndToList()
    app.showRectangles()
    print("LIST OF SORTED RECTANGLE CANDIDATES", app.listOfSixLists)
    print("end of SORTED candidates")
    app.showRectangles()
    app.checkSixXcloseness()
    print("FINAL LIST OF SORTED RECTANGLE CANDIDATES",app.listOfSixLists)
    print("end of SORTED candidates")

    #for region in app.regions:
    #    print(region)

    clone = app.getClone()
    #for six in app.listOfSets:
    #    for rectangle in six:

    for sixList in app.listOfSixLists:
        for i, [x,y,w,h] in enumerate(sixList):
            cv2.rectangle(clone,(x,y),(x+w,y+h),(0,255,0),5)
            roi_gray = clone[y:y+h, x:x+w]
    #        #roi_color = img[y:y+h, x:x+w]
    #        cv2.imwrite(str(i)+'-'+sys.argv[1]+'.jpg', roi_gray)

    cv2.imshow('img', clone)



    #    cv2.namedWindow('img', 0)
    #    cv2.imshow('img', vis)
    while(cv2.waitKey()!=ord('q')):
        continue
    cv2.destroyAllWindows()
