from tkinter import messagebox

import cv2
import numpy as np
import glob
import pathlib
import tkinter as tk
from PIL import Image, ImageTk
import os


class ImagePaths:
    def __init__(self, path=os.getcwd() + "\\coins", imageType="jpg"):
        self.imagePaths = glob.glob(os.path.join(path, '*.' + imageType))  # searching for all .jpg files

        print(self.imagePaths)
        print(os.getcwd())

    def fillListBox(self, listBoxObject):
        for imagePath in self.imagePaths:
            imageName = imagePath.split("\\")[-1]  # splitting all the .pdf up
            listBoxObject.insert('end', imageName)  # inserting each word into tk listbox

    def getPath(self, listBoxIndex):
        return self.imagePaths[listBoxIndex]


class ImageClass:

    def __init__(self, frame, image, colorType="rgb"):
        self.imageLabel = tk.Label(frame, image=imgtk, compound="top")
        self.imageLabel.pack(side="right")
        self.image = image
        self.colorType = colorType

    def setColorType(self, type="none"):
        if type != "none":
            self.colorType == type
        else:
            if self.colorType == "rgb":
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

            if self.colorType == "gray":
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def setImage(self, imagePath):
        self.image = cv2.imread(imagePath, cv2.IMREAD_COLOR)

        self.setColorType()

        print(imagePath)

        im = Image.fromarray(self.image)
        imgtk = ImageTk.PhotoImage(image=im)

        self.imageLabel['image'] = imgtk
        self.imageLabel.photo = imgtk

    def getCurrentImage(self):
        # TODO: returns the current image
        print("todo")


def callbackFileSelection(event):
    selection = event.widget.curselection()

    originalImage.setImage(lbImagePaths.getPath(selection[0]))
    grayImage.setImage(lbImagePaths.getPath(selection[0]))

def countCoins():
    print("hello")


if __name__ == '__main__':
    master = tk.Tk()  # creating a tk application+

    master.title('countingCoins')  # title of the program window

    master.geometry("")  # defining the window size

    rightFrame = tk.Frame(master)
    rightFrame.pack(side='right', fill=tk.BOTH, expand=True)

    image = cv2.imread('.\\coins\\coinb_05.JPG', cv2.IMREAD_COLOR)

    imgtk = ImageTk.PhotoImage(image=Image.fromarray(image))

    originalImage = ImageClass(rightFrame, imgtk)
    grayImage = ImageClass(rightFrame, imgtk, "gray")



    lbFileSelection = tk.Listbox(master, width=30)  # creating a listbox
    lbFileSelection.bind("<<ListboxSelect>>",
                         callbackFileSelection)  # callback function for listbox ... executes when you select an entry
    lbFileSelection.pack(side="top", fill=tk.BOTH, expand=True, padx=10, pady=10,
                         ipady=6)  # outer padding for the listbox/listview

    lbImagePaths = ImagePaths()
    lbImagePaths.fillListBox(lbFileSelection)

    countCoins = tk.Button(master, text='count', width=15, height=2, command=countCoins)
    countCoins.pack(side="bottom", padx=10, pady=10)

    master.mainloop()  # window mainloop

    cv2.destroyAllWindows()
