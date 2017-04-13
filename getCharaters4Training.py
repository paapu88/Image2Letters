
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from scipy import ndimage
import cv2

class GetCharatersForTraining():
    """
    produce positive training set for haar cascade to recognize a character in a finnisn licence plate
    """

    def __init__(self):
        self.font_height = 32
        self.output_height = 48
        self.chars = 'ABCDEFGHIJKLMNOPRSTUVXYZ0123456789'
        self.char_shift_y = -5


    def getMinAndMaxY(self, a, thr=0.5):
        """find the value in Y where image starts"""
        minY = None
        maxY = None
        for iy in range(a.shape[0]):
            for ix in range(a.shape[1]):
                if a[iy,ix]> thr:
                    minY = iy
                    break
        for iy in reversed(range(a.shape[0])):
            for ix in range(a.shape[1]):
                if a[iy,ix]> thr:
                    maxY = iy
                    break
        return minY, maxY

    def getMinAndMaxX(self, a, thr=0.5):
        """find the value in Y where image starts"""
        minX = None
        maxX = None
        for ix in range(a.shape[1]):
            for iy in range(a.shape[0]):
                if a[iy,ix]> thr:
                    minX = ix
                    break
        for ix in reversed(range(a.shape[1])):
            for iy in range(a.shape[0]):
                if a[iy,ix]> thr:
                    maxX = ix
                    break
        return minX, maxX



    def make_char_ims(self, font_file):
        """ get characters as numpy arrays"""

        font_size = self.output_height * 4

        font = ImageFont.truetype(font_file, font_size)

        height = max(font.getsize(c)[1] for c in self.chars)
        width =  max(font.getsize(c)[0] for c in self.chars)
        for c in self.chars:

            im = Image.new("RGBA", (width, height), (0, 0, 0))

            draw = ImageDraw.Draw(im)
            draw.text((0, 0), c, (255, 255, 255), font=font)
            scale = float(self.output_height) / height
            im = im.resize((int(width * scale), self.output_height), Image.ANTIALIAS)
            not_moved = np.array(im)[:, :, 0].astype(np.float32) / 255.
            minx,maxx = self.getMinAndMaxX(not_moved)
            cmx=np.average([minx,maxx])
            miny,maxy = self.getMinAndMaxY(not_moved)
            cmy=np.average([miny,maxy])

            cm = ndimage.measurements.center_of_mass(not_moved)
            rows,cols = not_moved.shape
            dy = rows/2 - cmy
            dx = cols/2 - cmx
            M = np.float32([[1,0,dx],[0,1,dy]])
            dst = cv2.warpAffine(not_moved,M,(cols,rows))
            yield c, dst

            #cm = ndimage.measurements.center_of_mass(not_moved)
            #rows,cols = not_moved.shape
            #dy = rows/2 - cm[0]
            #dx = cols/2 - cm[1]
            #print(c, cm, dx, dy)
            #M = np.float32([[1,0,dy],[0,1,dx]])
            ##dst = cv2.warpAffine(not_moved,M,(cols,rows))
            ##print("CM", cm,not_moved.shape)
            #yield c, not_moved



if __name__ == '__main__':
    import sys, glob
    from matplotlib import pyplot as plt

    app1 = GetCharatersForTraining()
    font_file = sys.argv[1]
    font_char_ims = dict(app1.make_char_ims(font_file=font_file))
    for mychar, img in font_char_ims.items():
        plt.imshow(img)
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()
