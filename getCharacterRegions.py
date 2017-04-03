"""
python3 getCharacterRegions.py image.jpq
"""

import cv2
import sys


class GetCharacterRegions():
    """typically within one plate.
    getInitialRegions returns the canditate-rectangles for all letters in the image

    """

    def __init__(self, imageFileName, colorConversion=cv2.COLOR_BGR2GRAY):

        self.img = cv2.imread(imageFileName)
        if colorConversion is not None:
            self.img = cv2.cvtColor(self.img, colorConversion)
        self.mser = cv2.MSER_create()
        self.regions = None
        self.imageY = self.img.shape[0]
        self.imageX = self.img.shape[1]
        self.listOfSixSets = []   # list of sets, each set has 6 rectangles
        self.listOfSixLists = []   # list of lists, each set has 6 rectangles
        #print("x y : ",self.imageX, self.imageY)


    def getInitialRegions(self):
        """ get rectangles of possible characters"""
        # tuple is used instead of list because we need make sets of tuples
        self.regions = tuple(self.mser.detectRegions(self.img)[-1])
        #print(self.regions)
        #hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]
        #cv2.polylines(vis, hulls, 1, (0,255,0))
        #return regions

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

    def checkArea(self, minArea=0.02, maxArea=0.2):
        """letters with too small areas are killed"""
        ok = []
        imageArea = self.imageX * self.imageY
        for (x,y,w,h) in self.regions:
            area=w*h
            if (area > minArea * imageArea) and (area < maxArea * imageArea):
                ok.append((x,y,w,h))
        self.regions = ok

    def checkSameness(self, toleranceInPixel=3):
        """ if we have almost the same rectange, kill one of them
            tthe remaining rectangle is as big as possible
        """
        deletes=[]
        for i, region_i in enumerate(self.regions):
            delete_i = False
            print("A", i, region_i)
            start=i+1
            for j in range(start, len(self.regions)):
                [x1,y1,w1,h1] = region_i
                [x2,y2,w2,h2] = self.regions[j]
                if ((abs(x2-x1) < toleranceInPixel) and \
                    (abs(y2-y1) < toleranceInPixel) and \
                    (abs(w2-w1) < toleranceInPixel) and \
                    (abs(h2-h1) < toleranceInPixel)):
                    delete_i = True
                    print(self.regions[j][0], min(x1,x2))
                    self.regions[j][0]=min(x1,x2)
                    self.regions[j][1]=min(y1,y2)
                    self.regions[j][2]=max(w1,w2)
                    self.regions[j][3]=max(h1,h2)
            deletes.append(delete_i)
        ok = []
        for delete, region in zip(deletes,self.regions):
            if not delete:
                ok.append(region)
        self.regions = ok

    def getSetsOfSix(self, heightCriterium=0.05):
        """ get all possible 6-sets of letters that are close in height"""
        import itertools
        self.listOfSixSets = []
        mymin=1-heightCriterium
        mymax=1+heightCriterium
        for i in range(len(self.regions)):
            heightI=self.regions[i][3]
            sixSet=set(self.regions[i])
            #print('ini:', sixSet)
            for j in range(len(self.regions)):
                heightJ=self.regions[j][3]
                #print(i, j, heightJ/heightI)
                if (mymin < heightJ/heightI) and (mymax > heightJ/heightI):
                    sixSet.add(self.regions[j])
            # check that number of rectangles is six and this set of rectangles is new
            #print("I, LEN SET",i, len(sixSet))
        if len(sixSet) == 6 and not(sixSet in self.listOfSixSets):
            #print("sixset: ", sixSet)
            self.listOfSixSets.append(sixSet)
            # if we have a longer set, take all 6-permutations..
        elif len(sixSet) > 6:
            myFixedSet = frozenset(sixSet)
            sixSetTrials= itertools.permutations(myFixedSet, 6)
            for addSet in sixSetTrials:
                if not(set(addSet) in self.listOfSixSets):
                    self.listOfSixSets.append(set(addSet))


    def sortSetsAndToList(self):
        """sort sets by x coordinate"""
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

    def checkSixXcloseness(self, minFraction=1.0, maxFraction=1.5):
        """check that subsegueent rectangles are close enought
            if NOT remove the 6-rectangle """
        import numpy as np

        deleteList = []
        for rectangles in self.listOfSixLists:
            averageWidth=np.average(np.asarray(rectangles[:][0]))
            if ((rectangles[1][0] - rectangles[0][0]) < (minFraction * averageWidth) or \
                (rectangles[2][0] - rectangles[1][0]) < (minFraction * averageWidth) or \
                (rectangles[3][0] - rectangles[2][0]) < (minFraction * averageWidth) or \
                (rectangles[4][0] - rectangles[3][0]) < (minFraction * averageWidth) or \
                (rectangles[5][0] - rectangles[4][0]) < (minFraction * averageWidth) or \
                (rectangles[1][0] - rectangles[0][0]) > (maxFraction * averageWidth) or \
                (rectangles[2][0] - rectangles[1][0]) > (maxFraction * averageWidth) or \
                #(rectangles[3][0] - rectangles[2][0]) > (maxFraction*averageWidth) or \ not for '-'
                (rectangles[4][0] - rectangles[3][0]) > (maxFraction * averageWidth) or \
                (rectangles[5][0] - rectangles[4][0]) > (maxFraction * averageWidth)):
                deleteList.append(True)
            else:
                deleteList.append(False)
        accepted = []
        for i, delete in enumerate(deleteList):
            if not delete:
                accepted.append(self.listOfSixLists[i])

    def showRectangles(self):
        clone = self.getClone()
        for (x,y,w,h) in self.regions:
            cv2.rectangle(clone,(x,y),(x+w,y+h),(0,255,0),5)
            roi_gray = clone[y:y+h, x:x+w]
        cv2.imshow('clone', clone)
        while(cv2.waitKey()!=ord('q')):
            continue

if __name__ == '__main__':
    import sys
    app = GetCharacterRegions(imageFileName=sys.argv[1], colorConversion=None)
    app.getInitialRegions()
    app.checkHeight()
    app.checkWidth()
    app.checkArea()
    print("all RECTANGLEs ",app.regions)
    print("end of all RECTANGLES")
    app.checkSameness()
    print("22: all RECTANGLEs ",app.regions)
    print("22: end of all RECTANGLES")
    sys.exit()
    app.getSetsOfSix()
    #print("LIST OF RECTANGLE CANDIDATES",app.listOfSixSets)
    #print("end of candidates")
    app.sortSetsAndToList()
    print("LIST OF SORTED RECTANGLE CANDIDATES",app.listOfSixLists)
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
