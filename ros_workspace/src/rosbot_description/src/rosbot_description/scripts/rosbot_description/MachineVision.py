import cv2
import numpy as np

class MachineVision:
    """
    A class that implements machine vision

    ...
    Attributes
    ----------
    __thresholdRange : int[]
        a table containing information about thresholding
    __minArea : int
        minimum area of objects that are to be detected
    __maxArea : int
        miximum area of objects that are to be detected
    __currentFrame : int
        current processed frame
    __kernel : np.ones
        filter of given size
    __textScale : double
        size of a font
    __textThickness : double
        thickness of a font
    __CIRCLE_MIN_EDGES : int
        the number of edges greater than this represents a circle

    Methods
    -------
    addThresholdRange(minTupleRGB, maxTupleRGB)
        add the frame thresholding range. User give tuple (R, G, B) R-red, G-green, B-blue
    uploadImage(image)
        add image to the object.
    uploadImageFromComputer(path)
        add image to the object from computer
    __drawText(frame, string, center)
        generates image with centered text
    runDetection()
        finds objects and describes them by the numebr of edges
    """
    def __init__(self, minimumArea, maximumArea, textScale = 1.5, textThickness = 3, kernelSize = 3):
        """

        :param minimumArea: dsd
            dsdas
        :param maximumArea:
        :param textScale:
        :param textThickness:
        :param kernelSize:
        """
        self.__thresholdRange = []
        self.__minArea = minimumArea
        self.__maxArea = maximumArea
        self.__currentFrame = None
        self.__kernel = np.ones((kernelSize, kernelSize), np.uint8)
        self.__textScale = textScale
        self.__textThickness = textThickness
        self.__CIRCLE_MIN_EDGES = 8

    def addThresholdRange(self, minTupleRGB, maxTupleRGB):
        """add the frame thresholding range. User give tuple (R, G, B) R-red, G-green, B-blue

        Parameters
        ----------
        minTupleRGB : tuple (R, G, B)
            minimum of the range of the threshold range (int R, int G, int B) R-red, G-green, B-blue
        maxTupleRGB: tuple (R, G, B)
            minimum of the range of the threshold range (int R, int G, int B) R-red, G-green, B-blue
        """
        self.__thresholdRange.append(((minTupleRGB[2], minTupleRGB[1], minTupleRGB[0]), (maxTupleRGB[2], maxTupleRGB[1], maxTupleRGB[0])))

    def uploadImage(self, image):
        """ add image to the object

        Parameters
        ----------
        image : cv::Mat
            frame to update
        """
        self.__currentFrame = image

    def uploadImageFromComputer(self, path):
        """ add image to the object from computer

        Parameters
        ----------
        path : string
            path to the file that is to be loaded
        """
        self.__currentFrame = cv2.imread(path)

    def __drawText(self, frame, string, center):
        """ generates image with centered text

        Parameters
        ----------
        frame : cv::Mat
            frame to draw text on
        string : string
            text to write
        center : (int, int)
            position of the center to write text on
        """
        textSize, _ = cv2.getTextSize(string, cv2.FONT_HERSHEY_SIMPLEX, self.__textScale, self.__textThickness)
        textOrigin = (int(center[0] - textSize[0] / 2), int(center[1] + textSize[1] / 2))
        frame = cv2.putText(frame, string, textOrigin, cv2.FONT_HERSHEY_SIMPLEX, self.__textScale, (255, 0, 0), self.__textThickness)

    def runDetection(self):
        """finds objects and describes them by the numebr of edges

        Returns
        -------
        image with highlighted contours and text
        """
        modifiedFrame = np.zeros(self.__currentFrame.shape[:2], np.uint8)

        for threshold in self.__thresholdRange:
            tmpFrame = cv2.inRange(self.__currentFrame, threshold[0], threshold[1])
            modifiedFrame = cv2.bitwise_or(modifiedFrame, tmpFrame)

        modifiedFrame = cv2.morphologyEx(modifiedFrame, cv2.MORPH_OPEN, self.__kernel)
        modifiedFrame = cv2.Canny(modifiedFrame, 100, 200)

        contours, _ = cv2.findContours(modifiedFrame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        returnFrame = self.__currentFrame.copy()

        i = 0
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if (i % 2 == 0 and cv2.contourArea(contour) >= self.__minArea and cv2.contourArea(contour) <= self.__maxArea):
                returnFrame = cv2.drawContours(returnFrame, contours, i, (255, 255, 0), 2)
                moment = cv2.moments(contour, False)
                centerPoint = (int(moment["m10"] / moment["m00"]), int(moment["m01"] / moment["m00"]))
                if len(approx) == 3:
                    self.__drawText(returnFrame, 'trojkat', centerPoint)
                elif len(approx) == 4:
                    self.__drawText(returnFrame, 'czworokat', centerPoint)
                elif len(approx) == 5:
                    self.__drawText(returnFrame, 'pieciokat', centerPoint)
                elif len(approx) == 6:
                    self.__drawText(returnFrame, 'szesciokat', centerPoint)
                elif len(approx) >= self.__CIRCLE_MIN_EDGES:
                    self.__drawText(returnFrame, 'kolo', centerPoint)
                else:
                    self.__drawText(returnFrame, str(len(approx)), centerPoint)

                
            i += 1

        return returnFrame

