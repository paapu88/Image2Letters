"""
predict a single character with various methods
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

class Classifier():
    def __init__(self, npImage=None, svmFileName=None, sizeX=12, sizeY=18):
        if svmFileName is not None:
            self.svm = cv2.ml.SVM_load(svmFileName)
        self.img = npImage  # image as numpy array
        self.sizeX = sizeX
        self.sizeY = sizeY
        self.asciiDict = {}
        self.plateString = None

    def setImage(self, image):
        """
        set image from numpy array
        """
        from PIL import Image
        self.img = Image.fromarray(image)

    def setCharacter(self, rectangle):
        (x,y,w,h) = rectangle
        self.char = self.img.copy()[y:y+h,x:x+w]

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

        m = cv2.moments(img)
        if abs(m['mu02']) < 1e-2:
            return img.copy()
        skew = m['mu11']/m['mu02']
        M = np.float32([[1, skew, -0.5*self.sizeX*skew], [0, 1, 0]])
        img = cv2.warpAffine(img, M, (self.sizeX, self.sizeY), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
        plt.imshow(img)
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()

        return img


    def preprocess_simple(self):
        self.sample = None
        resized = cv2.resize(self.char,(self.sizeX, self.sizeY))
        self.sample = np.reshape(resized, (-1, self.sizeX*self.sizeY)).astype(np.float32)/255.0

    def preprocess_hog(self):
        self.sample = None
        resized = cv2.resize(self.char,(self.sizeX, self.sizeY))
        resized = self.deskew(resized)
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
        print("after HOG:",self.sample.shape, self.sample.dtype)



    def get_character_by_SVM(self):
        self.preprocess_hog()
        ret, resp = self.svm.predict(self.sample)
        print (ret, resp)
        label = str(int(round(resp)))
        mychar = str(chr(self.asciiDict[label]))
        return mychar

    def defineSixPlateCharacters(self, listOfListofRectangles,
                                 lettersSvmFile='/home/mka/PycharmProjects/TrainSVM/Letters/SvmDir/digits_svm.dat',
                                 lettersDictionaryFile='/home/mka/PycharmProjects/TrainSVM/Letters/SvmDir/allSVM.txt.dict',
                                 digitsSvmFile='/home/mka/PycharmProjects/TrainSVM/Digits/SvmDir/digits_svm.dat',
                                 digitsDictionaryFile='/home/mka/PycharmProjects/TrainSVM/Digits/SvmDir/allSVM.txt.dict'):

        if len(listOfListofRectangles) > 1:
            raise NotImplementedError("in Classifier, defineSixPlateCharacters, only one set of six characters allowed")

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
            print(self.plateString)

    def getFinalString(self):
        return self.plateString

if __name__ == '__main__':
    import sys
    #app = Classifier(svmFileName='/home/mka/PycharmProjects/TrainSVM/Binary/SvmDir/digits_svm.dat')
    app = Classifier(svmFileName=sys.argv[2])
    app.setImageFromFile(imageFileName=sys.argv[1])
    #app.preprocess_hog()
    app.get_character_by_SVM()

