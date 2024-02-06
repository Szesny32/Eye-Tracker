import tkinter as tk
import cv2 as cv
from PIL import Image, ImageTk
from copy import copy

IMAGE_SIZE  = (720, 480)
BUTTON_PANEL_SIZE = (720, 100)

ROOT = tk.Tk()
ROOT.title("Eye Tracker")
ROOT.rowconfigure(0, minsize=IMAGE_SIZE[1])
ROOT.rowconfigure(1, minsize=BUTTON_PANEL_SIZE[1])
ROOT.columnconfigure(0, minsize=IMAGE_SIZE[0]) 

IMAGE_FRAME = tk.Label(ROOT)
IMAGE_FRAME.grid(row=0, column=0)

MODES = ['NORMAL', 'REGISTER', 'TRAIN', 'DETECT']
MODE = MODES[0]

BUTTON_FRAME = tk.Frame(ROOT)
BUTTON_FRAME.grid(row=1, column=0)  # W drugiej kolumnie

CV_IMAGE = None

def switch_mode(mode):
    global MODE, LEFT, RIGHT
    MODE = mode
    LEFT = (-1, -1)
    RIGHT = (-1, -1)

    if(MODE == 'NORMAL'):
        normal_mode()
    if(MODE == 'REGISTER'):
        register_mode()

for i, mode in enumerate(MODES):
    BUTTON = tk.Button(BUTTON_FRAME, text=mode, command=lambda mode = mode: switch_mode(mode), width=10, height=2)
    BUTTON.grid(row=1, column=i, sticky="nsew")



def refreshImage(imgArray):
    image = Image.fromarray(imgArray)
    photoImage = ImageTk.PhotoImage(image)
    IMAGE_FRAME.config(image=photoImage)
    IMAGE_FRAME.image = photoImage   


CAMERA = cv.VideoCapture(0)
def normal_mode():
    global CV_IMAGE
    ret, frame = CAMERA.read()
    if(ret):
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image = cv.resize(image, IMAGE_SIZE)
        CV_IMAGE = copy(image)

        cv.putText(image, f'mode: normal', org = (10, 30), color = (0, 0, 255), thickness = 2, fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale = 1)
        cv.putText(image, 'REGISTER (ENTER)', org = (550, 30), color = (0, 0, 255), thickness = 2, fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale = 0.5)
        refreshImage(image)

    if(MODE == 'NORMAL'):
        ROOT.after(10, normal_mode) 


LEFT = (-1, -1)
RIGHT = (-1, -1)

def register_mode():
    image = copy(CV_IMAGE)

    (X, Y) = getMouseXY()

    # Current mouse pointer
    cv.circle(image, (X, Y), 5, (0, 255, 0), 2) 

    # Mode
    cv.putText(image, f'mode: register ', org = (10, 30), color = (0, 0, 255), thickness = 2, fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale = 1)
    cv.putText(image, f'{(X, Y)}', org = (10, 60), color = (0, 0, 255), thickness = 1, fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale = 0.5)

    # Register info
    if(LEFT != (-1, -1)):
        cv.circle(image, LEFT, 5, (200, 0, 200), cv.FILLED) 
        cv.circle(image, LEFT, 5, (255, 255, 255), 2) 
        cv.putText(image, f'LEFT (LMB): {LEFT}', org = (500, 30), color = (200, 0, 200), thickness = 1, fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale = 0.5)
    else:
        cv.putText(image, f'LEFT (LMB): {LEFT}', org = (500, 30), color = (255, 0, 0), thickness = 1, fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale = 0.5)

    if(RIGHT != (-1, -1)):
        cv.circle(image, RIGHT, 5, (0, 0, 255), cv.FILLED) 
        cv.circle(image, RIGHT, 5, (255, 255, 255), 2) 
        cv.putText(image, f'RIGHT (RMB): {RIGHT}', org = (500, 60), color = (0, 0, 255), thickness = 1, fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale = 0.5)
    else:
        cv.putText(image, f'RIGHT (RMB): {RIGHT}', org = (500, 60), color = (255, 0, 0), thickness = 1, fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale = 0.5)

    if(LEFT != (-1, -1) and RIGHT != (-1, -1)):
        cv.putText(image, 'SAVE (ENTER)', org = (500, 90), color = (0, 255, 0), thickness = 1, fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale = 0.5)

    refreshImage(image)

    if(MODE == 'REGISTER'):
        ROOT.after(10, register_mode) 

def getMouseXY():
    x, y = ROOT.winfo_pointerxy()
    window_x = ROOT.winfo_rootx()
    window_y = ROOT.winfo_rooty()
    relative_x = x - window_x
    relative_y = y - window_y
    return (relative_x, relative_y)

def LMB(event):
    global MODE, LEFT
    if(MODE == 'REGISTER'):
        LEFT = getMouseXY()

def RMB(event):
    global MODE, RIGHT 
    if(MODE == 'REGISTER'):
        RIGHT = getMouseXY()
def ENTER(event):
    print("enter")


ROOT.bind("<Button-1>", LMB)
ROOT.bind("<Button-3>", RMB)
ROOT.bind("<Return>", ENTER)







switch_mode(MODE)
ROOT.mainloop()











# image = cv.imread(image_path)
# image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
# image = cv.resize(image, IMAGE_FRAME)
# image = Image.fromarray(image)

# #image = image.resize(IMAGE_FRAME,  Image.Resampling.NEAREST)  # Dopasuj rozmiar obrazka
# photo = ImageTk.PhotoImage(image)