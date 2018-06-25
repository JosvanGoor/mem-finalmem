import cv2

import preprocessing as pp
import linesegment as ls

class Character:

    def __init__(self, rect, prob, name):
        self.rect = rect
        self.prob = prob
        self.name = name

    #print funcs
    def __str__(self):
        return "{}: {}, p={}".format(self.name, self.rect, self.prob)
    def __repr__(self):
        return self.__str__()

class Image:
    
    def __init__(self, folder, filename):
        self.folder = folder
        self.filename = filename

        self.image = None
        self.line_names = []
        self.lines = []

    def load_processed(self):
        print("Loading {}/{}... ".format(self.folder, self.filename), end="")
        filename = "{}/{}".format(self.folder, self.filename)
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        image2 = pp.get_gauss_otsu(image)

        rect, mask = pp.get_page_rect_mask(image2)

        image = (image * (mask // 255))
        image = pp.subimage(image, rect)
        image2 = pp.get_gauss_otsu(image)
        self.image = pp.fill_white(image2)
        print("done!")
    
    def fill_gaps(self):
        contours = pp.get_all_rects(self.image)
        
        avg_size = 0

        for x in contours:
            avg_size += cv2.contourArea(x[4])
        
        avg_size /= len(contours)
        for x in contours:
            contour_size = cv2.contourArea(x[4])
            if contour_size > (avg_size * 10):
                self.image = cv2.drawContours(self.image, [x[4]], 0, 128, -1)
        cv2.imshow("contours", pp.resize(self.image, 0.5))
        cv2.waitKey(0)

    def segment_lines(self):
        print("Segmenting {}/{}... ".format(self.folder, self.filename), end="")
        self.lines = ls.segmentLine(self.image)
        print("done!")

    def output_annotation(self, linedata):
        with open("{}/{}.txt".format(self.folder, self.filename), "w+") as file:
            for line in self.line_names:
                for char in linedata[line]:
                    file.write("{} ".format(char.name))
                file.write("\r\n")
