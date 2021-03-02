import PIL.Image
import PIL.ImageTk
import cv2
import numpy as np
import glob
import os

import tkinter as tk

global imagePath


def selectImage(imageObject):
    newImagePath = filedialog.askopenfilename(initialdir="/", title="Select file",
                                              filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
    imageObject.setImage(newImagePath)


def callbackFileSelection(event):
    global imagePath

    if len(filePaths) == 0:
        messagebox.showinfo(title="Result", message="Please select a folder containing .pdf files.")

    else:

        selection = event.widget.curselection()
        print(selection)
        imagePath = filePaths[selection[0]]


class MouseCoordinate:
    position = []

    def __init__(self):
        self.position = []

    def select_point(self, event, x, y, flags, param):  # int event, int x, int y, ...
        if len(self.position) > 2:
            self.position = []
            print("No selection. Please select a point on the image.")

        # if event == cv2.EVENT_LBUTTONDBLCLK:  # left mouse button DOUBLE click in this case
        if event == cv2.EVENT_LBUTTONDOWN:  # single click
            self.position.append([x, y])

    def getPointOne(self):
        return self.position[0]

    def getPointTwo(self):
        return self.position[1]


if __name__ == '__main__':
    image = cv2.imread('paint.jpg', cv2.IMREAD_COLOR)

    imageGrey = cv2.imread('paint.jpg', cv2.IMREAD_GRAYSCALE)

    master = tk.Tk()  # creating a tk application+

    master.title('Banane')  # title of the program window

    master.geometry("")  # defining the window size

    height, width, depth = image.shape

    canvas = tk.Canvas(master, width=width, height=height)  # getting dimensions for tkinter canvas from opencv image
    canvas.pack(side="right", padx=10, pady=10, ipady=6)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # converting BGR into RGB

    photo = PIL.ImageTk.PhotoImage(
        image=PIL.Image.fromarray(imageGrey))  # converting opencv numpy array into photoimage (takes RGB)
    canvas.create_image(0, 0, image=photo, anchor=tk.NW)

    lbFileSelection = tk.Listbox(master, width=30)  # creating a listbox
    lbFileSelection.bind("<<ListboxSelect>>",
                         callbackFileSelection)  # callback function for listbox ... executes when you select an entry
    lbFileSelection.pack(side="top", fill=tk.BOTH, expand=True, padx=10, pady=10, ipady=6)  # outer padding for the listbox/listview

    allImagePaths = glob.glob('*/*.jpg',
                              recursive=True)  # searching for all .jpg files recursively, returns an array of files with their absoulte paths

    for imagePath in allImagePaths:
        imageName = imagePath.split("\\")[-1]  # splitting all the .pdf up
        lbFileSelection.insert('end', imageName)  # inserting each word into tk listbox

    selectImageButton = tk.Button(master, text='Select New Image', width=15, height=2, command=selectImage)
    selectImageButton.pack(side="bottom", padx=10, pady=10)

    master.mainloop()  # window mainloop

    cv2.destroyAllWindows()
