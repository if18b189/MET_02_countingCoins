"""
useful links:

    https://java2blog.com/cv2-imread-python/#cv2imread_Method_example # good source to look up basic cv2 functionalities

    https://docs.opencv.org/4.5.1/db/d8e/tutorial_threshold.html # documentation and examples for thresholding operations
    https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html



"""

import cv2
import matplotlib
import numpy as np
import glob
import pathlib
import tkinter as tk
from PIL import Image, ImageTk
import os


class ImagePaths:
    def __init__(self, path=os.getcwd() + "\\coins", imageType="jpg"):
        self.imagePaths = glob.glob(os.path.join(path, '*.' + imageType))  # searching for all .jpg files

        # print(self.imagePaths)
        # print(os.getcwd())

    def fillListBox(self, listBoxObject):
        for imagePath in self.imagePaths:
            imageName = imagePath.split("\\")[-1]  # splitting all the .pdf up
            listBoxObject.insert('end', imageName)  # inserting each word into tk listbox

    def getPath(self, listBoxIndex):
        return self.imagePaths[listBoxIndex]


class ImageClass:

    def __init__(self, frame, imageArray, colorType="rgb", title=""):

        self.title = title
        self.originalImage = ImageTk.PhotoImage(image=Image.fromarray(imageArray))

        self.imageArray = imageArray
        self.colorType = colorType

        self.newSize = (300, 200)
        resized = cv2.resize(self.imageArray, self.newSize)
        self.image = ImageTk.PhotoImage(image=Image.fromarray(resized))

        self.imageLabel = tk.Label(frame, image=self.image, compound="top", text=title)
        self.imageLabel.pack(side="left", padx=10, pady=10)

    def setColorType(self):
        if self.colorType == "rgb":
            self.imageArray = cv2.cvtColor(self.imageArray, cv2.COLOR_BGR2RGB)

        if self.colorType == "gray":
            self.imageArray = cv2.cvtColor(self.imageArray, cv2.COLOR_BGR2GRAY)

    def setImage(self, imagePath):
        self.imageArray = cv2.imread(imagePath, cv2.IMREAD_COLOR)

        self.setColorType()

        print(self.title + ": " + imagePath)

        resized = cv2.resize(self.imageArray, self.newSize)  # takes image array and resizes it, returns new image array
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(resized))  # turns imagearray into photoimage

        self.image = imgtk

        self.imageLabel['image'] = imgtk  # updating the label to show the new image
        self.imageLabel.photo = imgtk

    def setThreshold(self, thresholdValue=127, maxValue=255, thresholdingTechnique="binary"):
        # ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)            #0: Binary
        # ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)        #1: Binary Inverted
        # ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)             #2: Threshold Truncated
        # ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)            #3: Threshold to Zero
        # ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)        #4: Threshold to Zero Inverted


        if thresholdingTechnique == "binary":
            ret, thresh1 = cv2.threshold(self.imageArray, thresholdValue, maxValue,
                                         cv2.THRESH_BINARY)  # threshold binary operation

        if thresholdingTechnique == "binary_inv":
            ret, thresh1 = cv2.threshold(self.imageArray, thresholdValue, maxValue,
                                         cv2.THRESH_BINARY_INV)  # threshold binary inverse operation

        # add more if statements here for additional thresholding technique options

        resized = cv2.resize(thresh1, self.newSize)  # takes image array and resizes it, returns new image array
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(resized))

        self.image = imgtk

        self.imageLabel['image'] = imgtk  # updating the label to show the new image
        self.imageLabel.photo = imgtk

    def getImage(self):
        return self.image

    def getImageArray(self):
        return self.imageArray


class ImagePlot:
    def __init__(self):
        # self.image = image
        print("placeholder")

    def showPlot(self, imageArray):
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


def thresholdSliderCallback(var):
    binaryImage.setThreshold(int(var))


def callbackFileSelection(event):
    selection = event.widget.curselection()
    selectedImagePath = lbImagePaths.getPath(selection[0])

    # for images in imageObjectList:
    #     images.setImage(lbImagePaths.getPath(selection[0]))

    # updating originalImage
    originalImage.setImage(selectedImagePath)

    # updating grayImage
    grayImage.setImage(selectedImagePath)

    # updating binaryImage
    binaryImage.setImage(selectedImagePath)
    binaryImage.setThreshold(int(thresholdSlider.get()))

    # updating grayImage3
    grayImage3.setImage(selectedImagePath)


def openPlot():
    grayImagePlot.showPlot(grayImage.getImageArray())


def countCoins():


    # coinLabel['image'] = imgtk
    # coinLabel.photo = imgtk

    threshold = ImageWindow(master, grayImage.getImageArray(), "hello")


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

    initImagePath = '.\\coins\\coinb_01.JPG'  # imagepath for the initial image ... when program is started
    initImage = cv2.imread(initImagePath, cv2.IMREAD_COLOR)

    # initializing the image objects/ different views, used in this program
    originalImage = ImageClass(rightTopFrame, initImage, "rgb", "ORIGINAL")  # creating image object in rgb(default)
    grayImage = ImageClass(rightTopFrame, initImage, "gray", "GRAYSCALE")  # creating image object in grayscale
    binaryImage = ImageClass(rightTopFrame, initImage, "gray", "BINARY THRESHOLD")
    grayImage3 = ImageClass(rightBottomFrame, initImage, "gray", "GRAYSCALE3")

    # initialization of all images, copied from callbackFileSelection function

    # updating originalImage
    originalImage.setImage(initImagePath)

    # updating grayImage
    grayImage.setImage(initImagePath)

    # updating binaryImage
    binaryImage.setImage(initImagePath)
    binaryImage.setThreshold()

    # updating grayImage3
    grayImage3.setImage(initImagePath)

    grayImagePlot = ImagePlot()

    # initialization of GUI objects

    lbFileSelection = tk.Listbox(master, width=30)  # creating a listbox
    lbFileSelection.bind("<<ListboxSelect>>",
                         callbackFileSelection)  # callback function for listbox ... executes when you select an entry
    lbFileSelection.pack(side="top", fill=tk.BOTH, expand=True, padx=10, pady=10,
                         ipady=6)  # outer padding for the listbox/listview

    lbImagePaths = ImagePaths()
    lbImagePaths.fillListBox(lbFileSelection)

    countCoins = tk.Button(master, text='Count', width=15, height=2, command=countCoins)
    countCoins.pack(side="bottom", padx=10, pady=10)

    openPlot = tk.Button(master, text='Plot', width=15, height=2, command=openPlot)
    openPlot.pack(side="bottom", padx=10, pady=10)

    thresholdSlider = tk.Scale(master, from_=0, to=255, orient=tk.HORIZONTAL,
                               label="Threshold value:", command=thresholdSliderCallback)
    thresholdSlider.pack(side="bottom", padx=10, pady=10)
    thresholdSlider.set(127) # setting to 127, 127 = start/default value for image objects threshold

    master.mainloop()  # window mainloop

    cv2.destroyAllWindows()
