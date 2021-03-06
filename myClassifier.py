"""
predict a single character with SVM
inspired by
http://docs.opencv.org/trunk/dd/d3b/tutorial_py_svm_opencv.html

At the moment (4/2017) letter and digit recognation works ok,
but binary classification NOT (whether we have a character in the box or not)

"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

class Classifier():
    def __init__(self, npImage=None, svmFileName=None, dictionaryFile=None, sizeX=12, sizeY=18):
        self.asciiDict = {}
        if svmFileName is not None:
            self.setSvmTrainedFile(svmFileName=svmFileName)
        if dictionaryFile is not None:
            self.setSvmDictionary(dictionaryFile=dictionaryFile)
        self.img = npImage  # image as numpy array
        self.sizeX = sizeX
        self.sizeY = sizeY
        self.plateString = None
        self.plateStrings = []

    def setNumpyImage(self, image):
        """
        set image from numpy array
        """
        self.img = image

    def setCharacter(self, rectangle=None):
        if rectangle is None:
            self.char = self.img.copy()
        else:
            (x,y,w,h) = rectangle
            self.char = self.img.copy()[y:y+h,x:x+w]

    def showCharacter(self):
        plt.imshow(self.char, cmap = 'gray', interpolation = 'bicubic')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()

    def setImageFromFile(self, imageFileName, colorConversion=cv2.COLOR_BGR2GRAY):
        """ for debuggin image can be read from file also"""
        self.img = cv2.imread(imageFileName)
        self.img = cv2.cvtColor(self.img, colorConversion)

    def setSvmTrainedFile(self, svmFileName):
        """load trained svm classifier"""
        self.svm = cv2.ml.SVM_load(svmFileName)

    def setSvmDictionary(self, dictionaryFile):
        """A dictionary containing mapping from labels of svm to ascii codes of letters or digits"""
        self.dictionaryFile = dictionaryFile
        with open(dictionaryFile, 'r') as f:
            lines=f.readlines()
        for line in lines:
            value, key = line.split()
            key = line.split()[1]
            value = int(line.split()[0])
            self.asciiDict[key] = value

    def deskew(self, img):
        """ descew from
        http://codingexodus.blogspot.fi/2013/06/moment-based-de-skewing.html
        """
        SZ=max(self.sizeX, self.sizeY)
        SZ2=int(round(SZ))
        resized = cv2.resize(img,(SZ, SZ))
        m = cv2.moments(resized)
        if abs(m['mu02']) < 1e-2:
            return img.copy()
        skew = m['mu11']/m['mu02']
        M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
        resized = cv2.warpAffine(resized,M,(SZ2, SZ2),flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR)
        rotatedImg=cv2.resize(resized,(self.sizeX, self.sizeY))

        plt.imshow(rotatedImg, cmap = 'gray', interpolation = 'bicubic')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()
        return rotatedImg

    def preprocess_simple(self):
        self.sample = None
        resized = cv2.resize(self.char,(self.sizeX, self.sizeY))
        self.sample = np.reshape(resized, (-1, self.sizeX*self.sizeY)).astype(np.float32)/255.0

    def preprocess_hog(self):
        """picking right features, if used this must also be present when generating imput file for SVM"""
        self.sample = None
        resized = cv2.resize(self.char,(self.sizeX, self.sizeY))
        gx = cv2.Sobel(resized, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(resized, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n*ang/(2*np.pi))
        bin_cells = bin[:10,:10], bin[10:,:10], bin[:10,10:], bin[10:,10:]
        mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)
        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= np.linalg.norm(hist) + eps
        self.sample = np.reshape(hist, (-1, len(hist))).astype(np.float32)


    def get_character_by_SVM(self, binary=False):
        self.preprocess_hog()
        ret, resp = self.svm.predict(self.sample)
        #print (ret, resp)
        #print("prob:", resp)
        label = int(round(resp.flatten()[0]))
        if binary:
            return label
        else:
            mychar = str(chr(self.asciiDict[str(label)]))
            return mychar

    def defineSixPlateCharacters(self, listOfListofRectangles,
                                 lettersSvmFile='/home/mka/PycharmProjects/TrainSVM/Letters/SvmDir/digits_svm.dat',
                                 lettersDictionaryFile='/home/mka/PycharmProjects/TrainSVM/Letters/SvmDir/allSVM.txt.dict',
                                 digitsSvmFile='/home/mka/PycharmProjects/TrainSVM/Digits/SvmDir/digits_svm.dat',
                                 digitsDictionaryFile='/home/mka/PycharmProjects/TrainSVM/Digits/SvmDir/allSVM.txt.dict',
                                 binarySvmFile='/home/mka/PycharmProjects/TrainSVM/Binary/SvmDir/digits_svm.dat'):
        """check all plates and in each plate go through every set of 6-rectangles
        give a result for each 6-rectange, for instance ABC-123 """


        # if there are more thatn one candidate for 6-chars, we predict them all...
        for plate in listOfListofRectangles:
            if len(plate) != 6:
                raise RuntimeError('only six character plates allowed in getSixPlateCharacters')
            string=''
            # alphabets
            self.setSvmTrainedFile(svmFileName=lettersSvmFile)
            self.setSvmDictionary(dictionaryFile=lettersDictionaryFile)
            for rectangle in plate[0:3]:
                self.setCharacter(rectangle=rectangle)
                string = string + self.get_character_by_SVM()
            # digits
            self.setSvmTrainedFile(svmFileName=digitsSvmFile)
            self.setSvmDictionary(dictionaryFile=digitsDictionaryFile)
            for rectangle in plate[3:6]:
                self.setCharacter(rectangle=rectangle)
                string = string + self.get_character_by_SVM()
            self.plateString = (string[0:3]+'-'+string[3:6])
            #print(self.plateString)
            self.plateStrings.append(self.plateString)

    def getFinalStrings(self):
        return self.plateStrings


if __name__ == '__main__':
    import sys
    app = Classifier(svmFileName='/home/mka/PycharmProjects/TrainSVM/Letters/SvmDir/digits_svm.dat',
                     dictionaryFile='/home/mka/PycharmProjects/TrainSVM/Letters/SvmDir/allSVM.txt.dict')
    app.setImageFromFile(imageFileName=sys.argv[1])
    app.setCharacter()
    print("result:",app.get_character_by_SVM())

