"""
predict a single character with various methods
"""
import cv2
import numpy as np

class Classifier():
    def __init__(self, npImage=None, svmFileName=None):
        if svmFileName is not None:
            self.svm = cv2.ml.SVM_load(svmFileName)
        self.img = npImage  # image as numpy array

    def preprocess_hog(self, sizeX=12, sizeY=18):
        self.sample = None
        resized = cv2.resize(self.img,(sizeX, sizeY))
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

    def setImageFromFile(self, imageFileName, colorConversion=cv2.COLOR_BGR2GRAY):
        """ for debuggin image can be read from file also"""
        self.img = cv2.imread(imageFileName)
        self.img = cv2.cvtColor(self.img, colorConversion)

    def get_character_by_SVM(self):

        ret, resp = self.svm.predict(self.sample)
        print (ret, resp)

if __name__ == '__main__':
    import sys
    app = Classifier(svmFileName='/home/mka/PycharmProjects/TrainSVM/Binary/SvmDir/digits_svm.dat')
    app.setImageFromFile(imageFileName=sys.argv[1])
    app.preprocess_hog()
    app.get_character_by_SVM()

