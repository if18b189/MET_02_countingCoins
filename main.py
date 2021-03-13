"""
useful links and sources:

    https://java2blog.com/cv2-imread-python/#cv2imread_Method_example # good source to look up basic cv2 functionalities

    https://docs.opencv.org/4.5.1/db/d8e/tutorial_threshold.html # documentation and examples for thresholding operations
    https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html

    https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html # opencv erosion and dilatation
    https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#gaeb1e0c1033e3f6b891a25d0511362aeb # functions



"""
from tkinter import ttk, messagebox

import cv2
import numpy as np
import glob
import tkinter as tk
from PIL import Image, ImageTk
import os

# printing bigger np matrix
# # a = np.arange(127 * 127).reshape(127, 127)
np.set_printoptions(edgeitems=127)  # this line sets the amount you want to print


class ImagePaths:
    """
    Finds all images and summarizes their paths.
    """

    def __init__(self, path=os.getcwd() + "\\coins", imageType="jpg"):
        """
        Constructor
        """
        self.imagePaths = glob.glob(os.path.join(path, '*.' + imageType))  # searching for all .jpg files

        # print(self.imagePaths)
        # print(os.getcwd())

    def fillListBox(self, listBoxObject):
        """
        Fills the listbox(GUI) with image names.
        """
        for imagePath in self.imagePaths:
            imageName = imagePath.split("\\")[-1]  # splitting all the .pdf up
            listBoxObject.insert('end', imageName)  # inserting each word into tk listbox

    def getPath(self, listBoxIndex):
        """
        Returns image path according to the given index.
        """
        return self.imagePaths[listBoxIndex]


class ImageClass:
    """
    Shows image on a window as a tkinter label. Summarizes operations for images.
    """

    def __init__(self, frame, imageArray, colorType="rgb", title=""):
        """
        Constructor
        """

        self.title = title
        self.originalImage = ImageTk.PhotoImage(image=Image.fromarray(imageArray))
        self.originalImageArray = imageArray

        self.imageArray = imageArray
        self.colorType = colorType

        self.newSize = (400, 300)  # default size for all images displayed in the program
        resized = cv2.resize(self.imageArray, self.newSize)
        self.image = ImageTk.PhotoImage(image=Image.fromarray(resized))

        self.imageLabel = tk.Label(frame, image=self.image, compound="top", text=title)
        self.imageLabel.pack(side="left", padx=10, pady=10)

    def setImage(self, imagePath):
        """
        Changes the current image and updates with updateImage().
        """
        global convertedImageArray
        imageArray = cv2.imread(imagePath, cv2.IMREAD_COLOR)

        if self.colorType == "rgb":
            convertedImageArray = cv2.cvtColor(imageArray, cv2.COLOR_BGR2RGB)

        if self.colorType == "gray":
            convertedImageArray = cv2.cvtColor(imageArray, cv2.COLOR_BGR2GRAY)

        # add more if statements here for additional color options

        # print(self.title + ": " + imagePath)

        self.image = ImageTk.PhotoImage(image=Image.fromarray(convertedImageArray))
        self.imageArray = convertedImageArray
        self.originalImage = ImageTk.PhotoImage(image=Image.fromarray(convertedImageArray))
        self.originalImageArray = convertedImageArray
        self.updateImage()

    def threshold(self, thresholdValue=127, maxValue=255, thresholdingTechnique="binary"):
        """
        Applies threshold operation with given values and updates the image with updateImage().
        """
        # ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)            #0: Binary
        # ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)        #1: Binary Inverted
        # ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)             #2: Threshold Truncated
        # ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)            #3: Threshold to Zero
        # ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)        #4: Threshold to Zero Inverted

        global thresh1

        if thresholdingTechnique == "binary":
            ret, thresh1 = cv2.threshold(self.imageArray, thresholdValue, maxValue,
                                         cv2.THRESH_BINARY)  # threshold binary operation

        if thresholdingTechnique == "binary_inv":
            ret, thresh1 = cv2.threshold(self.imageArray, thresholdValue, maxValue,
                                         cv2.THRESH_BINARY_INV)  # threshold binary inverse operation

        # add more if statements here for additional thresholding technique options

        self.imageArray = thresh1
        self.updateImage()

    def erode(self, iterationsArg=1):
        """
        Morphological erode function.
        """

        kernel = np.array([[0, -1, 0],
                           [-1, 4, -1],
                           [0, -1, 0]], np.uint8)  # laplacian operator/mask

        imageErosion = cv2.erode(self.imageArray, kernel, iterations=iterationsArg)
        # tweaking the iterations could lead to better results

        self.imageArray = imageErosion
        self.updateImage()

        """
        morphological transformations
        source: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
                http://datahacker.rs/006-morphological-transformations-with-opencv-in-python/
                
        additional information about convolution filters:
        https://www.l3harrisgeospatial.com/docs/ConvolutionMorphologyFilters.html

        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        # shorter example for opening ( erosion followed by dilitation)
        # useful in removing noise

        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        # shorter example for closing ( erosion followed by dilitation)
        # useful in closing small holes inside the foreground objects

        # Rectangular Kernel
        >> > cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        array([[1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1]], dtype=uint8)

        # Elliptical Kernel
        >> > cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        array([[0, 0, 1, 0, 0],
               [1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1],
               [0, 0, 1, 0, 0]], dtype=uint8)

        # Cross-shaped Kernel
        >> > cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        array([[0, 0, 1, 0, 0],
               [0, 0, 1, 0, 0],
               [1, 1, 1, 1, 1],
               [0, 0, 1, 0, 0],
               [0, 0, 1, 0, 0]], dtype=uint8)
               
        """

    def dilate(self, iterationsArg=1):
        """
        Morphological dilate function.
        """

        kernel = np.array([[0, -1, 0],
                           [-1, 4, -1],
                           [0, -1, 0]], np.uint8)  # laplacian operator/mask

        imageDilation = cv2.dilate(self.imageArray, kernel, iterations=iterationsArg)
        # tweaking the iterations could lead to better results

        self.imageArray = imageDilation
        self.updateImage()

    def distance(self, operationArg="DIST_C"):

        if operationArg == "DIST_C":   operation = cv2.DIST_C
        if operationArg == "DIST_L1":   operation = cv2.DIST_L1
        if operationArg == "DIST_L2":   operation = cv2.DIST_L2
        if operationArg == "DIST_LABEL_PIXEL":  operation = cv2.DIST_LABEL_PIXEL
        if operationArg == "DIST_MASK_3":   operation = cv2.DIST_MASK_3

        # examples of distance transformations: https://www.tutorialspoint.com/opencv/opencv_distance_transformation.htm

        imageDistance = cv2.distanceTransform(self.imageArray, operation, 3)  # applying distance transformation
        imageDistance = cv2.normalize(imageDistance, None, 0, 255,
                                      cv2.NORM_MINMAX)  # normalizing values, better results/visibility/peak values

        # cv2.normalize(imageDistance, imageDistance, 0, 255, cv2.NORM_MINMAX)

        # ret, thresh1 = cv2.threshold(imageDistance, 0, 255, cv2.THRESH_TOZERO)

        self.imageArray = imageDistance

        # print(imageDistance)

        # a = np.arange(127 * 127).reshape(127, 127)

        self.updateImage()

    def count(self):
        """
        Last few operations to count the number of coins.
        """

        ret, thresh1 = cv2.threshold(self.imageArray, 160, 255, cv2.THRESH_BINARY)

        s = (3, 3)
        kernel1 = np.ones(s)
        dilated = cv2.dilate(thresh1, kernel1)

        # dilated = cv2.cvtColor(dilated, cv2.COLOR_RGB2GRAY)

        # dilated = cv2.cvtColor(dilated, cv2.CV_32S)

        sure_fg = np.uint8(dilated) # converting to uint8/CV_8U

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sure_fg, connectivity=8,

                                                                                ltype=cv2.CV_32S) # counting

        # ret, labels = cv2.connectedComponents(sure_fg)
        # vals, counts = np.unique(np.hstack([labels[0], labels[-1], labels[:, 0], labels[:, -1]]),
        #                          return_counts=True)

        cv2.imshow("Counting Image", dilated)   # showing image

        print(num_labels - 1)

        messagebox.showinfo('Counter: ', num_labels - 1)

        # (numLabels, labels, stats, centroids) = output

    def updateImage(self):
        """
        Resizes and updates the currently displayed image with the given image array.
        """

        resized = cv2.resize(self.imageArray, self.newSize)  # takes image array and resizes it, returns new image array
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(resized))

        self.image = imgtk

        self.imageLabel['image'] = imgtk  # updating the label to show the new image
        self.imageLabel.photo = imgtk

    def reset(self):
        """
        Returns the image and imageArray to their original values (the values, they were initialized with).
        """
        self.imageArray = self.originalImageArray
        self.image = self.originalImage

    def getImage(self):
        """
        Returns the image currently set in the class object.
        """
        return self.image

    def getImageArray(self):
        """
        Returns the image array currently set in the class object.
        """
        return self.imageArray


class ImagePlot:
    """
    Opens a new window with plots and histograms.
    *Currently not working
    """

    def __init__(self):
        # self.image = image

        print("not yet implemented")

    def showPlot(self, imageArray):
        """
        Opens a new window with plots and histograms.
        *Currently not working
        """

        self.imageArray = imageArray

        # Toplevel object which will
        # be treated as a new window
        newWindow = tk.Toplevel(master)

        # sets the title of the
        # Toplevel widget
        newWindow.title("GrayscalePlot")  # add filename - Info

        # sets the geometry of toplevel
        newWindow.geometry("")  # "" means it will automatically resize

        # f = Figure(figsize=(5, 5), dpi=100)
        # a = f.add_subplot(111)
        # a.plot([1, 2, 3, 4, 5, 6, 7, 8], [5, 6, 1, 3, 8, 9, 3, 5])
        #
        # canvas = FigureCanvasTkAgg(f, self)
        # canvas.show()
        # canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        #
        # toolbar = NavigationToolbar2TkAgg(canvas, self)
        # toolbar.update()
        # canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # TODO: implement histogram view


def updateParameter(event):
    """
    Applies the value from the threshold slider on following image objects.
    """
    # updating binaryImage
    binaryImage.reset()
    binaryImage.threshold(int(thresholdBinarySlider.get()))

    # updating erodeImage
    erodeImage.reset()
    erodeImage.threshold(int(thresholdBinarySlider.get()))
    erodeImage.erode(int(erodeIterationSlider.get()))

    # updating dilateImage
    dilateImage.reset()
    dilateImage.threshold(int(thresholdBinarySlider.get()))
    dilateImage.erode(int(erodeIterationSlider.get()))
    dilateImage.dilate(int(dilateIterationSlider.get()))

    # updating distanceImage
    distanceImage.reset()
    distanceImage.threshold(int(thresholdBinarySlider.get()))
    distanceImage.erode(int(erodeIterationSlider.get()))
    distanceImage.dilate(int(dilateIterationSlider.get()))
    distanceImage.distance(distanceTypeCombo.get())


def callbackFileSelection(event):
    """
    Gets called everytime an image is selected from the listbox
    Changes images according to selection.
    Applies additional functions depending on what kind of operation you want to show.

    Note: inefficiency, creating new objects for each image and reapplying operations instead of inheritance ( from previous images/steps).
    """
    selection = event.widget.curselection()
    selectedImagePath = lbImagePaths.getPath(selection[0])

    # updating originalImage
    originalImage.setImage(selectedImagePath)

    # updating grayImage
    grayImage.setImage(selectedImagePath)

    # updating binaryImage
    binaryImage.setImage(selectedImagePath)
    binaryImage.threshold(int(thresholdBinarySlider.get()))

    # updating erodeImage
    erodeImage.setImage(selectedImagePath)
    erodeImage.threshold(int(thresholdBinarySlider.get()))
    erodeImage.erode(int(erodeIterationSlider.get()))

    # updating dilateImage
    dilateImage.setImage(selectedImagePath)
    dilateImage.threshold(int(thresholdBinarySlider.get()))
    dilateImage.erode(int(erodeIterationSlider.get()))
    dilateImage.dilate(int(dilateIterationSlider.get()))

    # updating distanceImage
    distanceImage.setImage(selectedImagePath)
    distanceImage.threshold(int(thresholdBinarySlider.get()))
    distanceImage.erode(int(erodeIterationSlider.get()))
    distanceImage.dilate(int(dilateIterationSlider.get()))
    distanceImage.distance(distanceTypeCombo.get())


def openPlot():
    print("not yet implemented")


def countCoins():
    distanceImage.count()


if __name__ == '__main__':
    # main application window
    master = tk.Tk()  # creating a tk application+
    master.title('countingCoins')  # title of the program window
    master.geometry("")  # defining the window size, blank means it will "self adjust"

    # subframes, structuring the alignment of GUI objects

    rightFrame = tk.Frame(master)
    rightFrame.pack(side='right', fill=tk.BOTH, expand=True)

    rightTopFrame = tk.Frame(rightFrame)
    rightTopFrame.pack(side='top', fill=tk.BOTH, expand=True)

    rightBottomFrame = tk.Frame(rightFrame)
    rightBottomFrame.pack(side='bottom', fill=tk.BOTH, expand=True)

    # initial image
    initImagePath = '.\\coins\\coinb_01.JPG'  # imagepath for the initial image ... when program is started
    initImage = cv2.imread(initImagePath, cv2.IMREAD_COLOR)

    # initializing the image objects/ different views, used in this program
    originalImage = ImageClass(rightTopFrame, initImage, "rgb", "ORIGINAL")  # creating image object in rgb(default)
    grayImage = ImageClass(rightTopFrame, initImage, "gray", "GRAYSCALE")  # creating image object in grayscale
    binaryImage = ImageClass(rightTopFrame, initImage, "gray", "BINARY THRESHOLD")
    erodeImage = ImageClass(rightBottomFrame, binaryImage.getImageArray(), "gray", "ERODE")
    dilateImage = ImageClass(rightBottomFrame, erodeImage.getImageArray(), "gray", "DILATE")
    distanceImage = ImageClass(rightBottomFrame, dilateImage.getImageArray(), "gray", "DISTANCE")

    # initialization of all images, copied from callbackFileSelection function

    # updating originalImage
    originalImage.setImage(initImagePath)

    # updating grayImage
    grayImage.setImage(initImagePath)

    # updating binaryImage
    binaryImage.setImage(initImagePath)
    binaryImage.threshold()

    # updating erodeImage
    erodeImage.setImage(initImagePath)
    erodeImage.threshold()
    erodeImage.erode()

    # updating dilateImage
    dilateImage.setImage(initImagePath)
    dilateImage.threshold()
    dilateImage.erode()
    dilateImage.dilate()

    # updating distanceImage
    distanceImage.setImage(initImagePath)
    distanceImage.threshold()
    distanceImage.erode()
    distanceImage.dilate()
    distanceImage.distance()

    # initialization of GUI objects

    lbFileSelection = tk.Listbox(master, width=30)  # creating a listbox
    lbFileSelection.bind("<<ListboxSelect>>",
                         callbackFileSelection)  # callback function for listbox ... executes when you select an entry
    lbFileSelection.pack(side="top", fill=tk.BOTH, expand=True, padx=10, pady=10,
                         ipady=6)  # outer padding for the listbox/listview

    lbImagePaths = ImagePaths()
    lbImagePaths.fillListBox(lbFileSelection)

    countCoinsButton = tk.Button(master, text='Count Coins', width=15, height=2, command=countCoins)
    countCoinsButton.pack(side="bottom", padx=10, pady=10)

    openPlotButton = tk.Button(master, text='Plot', width=15, height=2, command=openPlot)
    openPlotButton.pack(side="bottom", padx=10, pady=10)

    # thresholdBinarySlider
    thresholdBinarySlider = tk.Scale(master, from_=0, to=255, orient=tk.HORIZONTAL,
                                     label="Binary Threshold:", command=updateParameter)
    thresholdBinarySlider.pack(side="top", fill=tk.X, padx=10, pady=2)
    thresholdBinarySlider.set(127)  # setting to 127, 127 = start/default value for image objects threshold

    # erodeSLider
    erodeIterationSlider = tk.Scale(master, from_=0, to=20, orient=tk.HORIZONTAL,
                                    label="Erode Iterations:", command=updateParameter)
    erodeIterationSlider.pack(side="top", fill=tk.X, padx=10, pady=2)
    erodeIterationSlider.set(1)  # preset value

    # dilateSlider
    dilateIterationSlider = tk.Scale(master, from_=0, to=20, orient=tk.HORIZONTAL,
                                     label="Dilate Iterations:", command=updateParameter)
    dilateIterationSlider.pack(side="top", fill=tk.X, padx=10, pady=2)
    dilateIterationSlider.set(1)  # preset value

    # distance operation type combobox
    distanceTypes = ["DIST_C", "DIST_L1", "DIST_L2",
                     "DIST_LABEL_PIXEL", "DIST_MASK_3"]
    distanceTypeLabel = tk.Label(text="Distance Operation:")
    distanceTypeLabel.pack(side="left", fill=tk.X, padx=10, pady=2)
    distanceTypeCombo = ttk.Combobox(master, values=distanceTypes)
    distanceTypeCombo.bind('<<ComboboxSelected>>', updateParameter)
    distanceTypeCombo.set("DIST_C")
    distanceTypeCombo.pack(side="top", fill=tk.X, padx=10, pady=2)

    master.mainloop()  # window mainloop

    cv2.destroyAllWindows()
