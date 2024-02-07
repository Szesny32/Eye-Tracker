import tkinter as tk
import cv2 as cv
from PIL import Image, ImageTk
from copy import copy
from time import time 
import numpy as np
from tensorflow.keras.models import Sequential, save_model, load_model

IMAGE_SIZE  = (720, 480)
BUTTON_PANEL_SIZE = (720, 100)

RECORDING = False

SAVE_FILE = 'dataset/s4'
ROOT = tk.Tk()
ROOT.title("Eye Tracker")
ROOT.rowconfigure(0, minsize=IMAGE_SIZE[1])
ROOT.rowconfigure(1, minsize=BUTTON_PANEL_SIZE[1])
ROOT.columnconfigure(0, minsize=IMAGE_SIZE[0]) 

IMAGE_FRAME = tk.Label(ROOT)
IMAGE_FRAME.grid(row=0, column=0)

MODES = ['NORMAL', 'REGISTER PHOTO', 'REGISTER VIDEO', 'TRAIN', 'TRACK']
MODE = MODES[0]

BUTTON_FRAME = tk.Frame(ROOT)
BUTTON_FRAME.grid(row=1, column=0)  # W drugiej kolumnie

CV_IMAGE = None
VIDEO = []

MODEL = load_model('model.h5')


def switch_mode(mode):
    global MODE, LEFT, RIGHT
    if(mode != MODE):
        MODE = mode
        

        if(MODE == 'NORMAL' or MODE == 'TRACK'):
            normal_mode()
            LEFT = (-1, -1)
            RIGHT = (-1, -1)
            VIDEO.clear()
        if(MODE == 'REGISTER PHOTO' or MODE == 'REGISTER VIDEO'):
            register_mode()

for i, mode in enumerate(MODES):
    BUTTON = tk.Button(BUTTON_FRAME, text=mode, command=lambda mode = mode: switch_mode(mode), width=15, height=2)
    BUTTON.grid(row=1, column=i, sticky="nsew")



def refreshImage(imgArray):
    image = Image.fromarray(imgArray)
    photoImage = ImageTk.PhotoImage(image)
    IMAGE_FRAME.config(image=photoImage)
    IMAGE_FRAME.image = photoImage   

import matplotlib.pyplot as plt
CAMERA = cv.VideoCapture(0)
def normal_mode():
    global CV_IMAGE, VIDEO, MODEL, RECORDING
    ret, frame = CAMERA.read()
    if(ret):
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image = cv.resize(image, IMAGE_SIZE)
        CV_IMAGE = copy(image)
        
        if(MODE == 'NORMAL' and RECORDING == True):
            VIDEO.append(copy(image))
            if len(VIDEO) > 100:
                switch_mode('REGISTER VIDEO')
                RECORDING = False
                #VIDEO.pop(0)  
            cv.putText(image, f'mode: {MODE} [{len(VIDEO)}]', org = (10, 30), color = (255, 0, 0), thickness = 2, fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale = 1)
        else:
            cv.putText(image, f'mode: {MODE} ', org = (10, 30), color = (0, 0, 255), thickness = 2, fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale = 1)
        if(MODE == 'TRACK'):
            height, width = CV_IMAGE.shape[:2]
            img = copy(CV_IMAGE)    
            img = cv.resize(img, (256, 256))
            img = img / 255.0
            predicted_points = MODEL.predict(np.array([img]), verbose=0)
     

            # fig, axes = plt.subplots(1, 2, figsize=(16, 16))
            
            # axes[0].imshow(img)
            # axes[0].scatter(predicted_points[0][0], predicted_points[0][1], c='red', marker='o', label='Predicted Left Point')
            # axes[0].scatter(predicted_points[0][2], predicted_points[0][3], c='blue', marker='o', label='Predicted Right Point')
            # axes[0].axis('off')
                    
            x_scale = 256.0 / float(width) 
            y_scale = 256.0 / float(height) 

            L = (int((predicted_points[0][0] / x_scale)), int((predicted_points[0][1] / y_scale)))
            R = (int((predicted_points[0][2] / x_scale)), int((predicted_points[0][3] / y_scale)))



            # axes[1].imshow(CV_IMAGE)
            # axes[1].scatter(L[0], L[1], c='red', marker='o', label='Predicted Left Point')
            # axes[1].scatter(R[0], R[1], c='blue', marker='o', label='Predicted Right Point')
            # axes[1].axis('off')

            # plt.tight_layout()
            # plt.show()


            cv.putText(image, f'{L} : {R}', org = (500, 30), color = (0, 0, 255), thickness = 2, fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale = 0.5)
            cv.circle(image, L, 5, (255, 255, 255), 2) 
            cv.circle(image, R, 5, (255, 255, 255), 2) 


        if(MODE == 'NORMAL'):
            cv.putText(image, 'REGISTER (ENTER)', org = (550, 30), color = (0, 0, 255), thickness = 2, fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale = 0.5)
        refreshImage(image)

    if(MODE == 'NORMAL' or MODE=='TRACK'):
        ROOT.after(10, normal_mode) 





LEFT = (-1, -1)
RIGHT = (-1, -1)




def register_mode():
    global VIDEO, CV_IMAGE
    
    if(MODE == 'REGISTER VIDEO'):
        if(len(VIDEO) > 1):
            CV_IMAGE = copy(VIDEO[0])
        else:
            switch_mode('REGISTER PHOTO')

    image = copy(CV_IMAGE)

    (X, Y) = getMouseXY()

    # Current mouse pointer
    cv.circle(image, (X, Y), 5, (0, 255, 0), 2) 

    # Mode
    cv.putText(image, f'mode: {MODE} : {len(VIDEO)}', org = (10, 30), color = (0, 0, 255), thickness = 2, fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale = 1)
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

    if(MODE == 'REGISTER PHOTO' or MODE == 'REGISTER VIDEO'):
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
    if(MODE == 'REGISTER PHOTO' or MODE == 'REGISTER VIDEO' or MODE == 'NORMAL'):
        LEFT = getMouseXY()
        if(MODE == 'NORMAL'):
            switch_mode('REGISTER PHOTO')

def RMB(event):
    global MODE, RIGHT 
    if(MODE == 'REGISTER PHOTO' or MODE == 'REGISTER VIDEO' or MODE == 'NORMAL'):
        RIGHT = getMouseXY()
        if(MODE == 'NORMAL'):
            switch_mode('REGISTER PHOTO')

def ENTER(event):
    global MODE, LEFT, RIGHT, VIDEO, SAVE_FILE
    if(MODE == 'REGISTER PHOTO' or MODE =='REGISTER VIDEO'):
        if(LEFT != (-1, -1) and RIGHT != (-1, -1)):
            newsize = 256.0 
            height, width = CV_IMAGE.shape[:2]
            x_scale = newsize / width
            y_scale = newsize / height
            print(LEFT, RIGHT)
            print(x_scale, y_scale)
            img = cv.resize(CV_IMAGE, (int(newsize), int(newsize)))
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            left_x, left_y = round(LEFT[0] * x_scale, 2), round(LEFT[1] * y_scale, 2)
            right_x, right_y = round(RIGHT[0]  * x_scale, 2), round(RIGHT[1] * y_scale, 2)

            filename = f'{SAVE_FILE}/{int(time())}_{left_x:.2f}_{left_y:.2f}_{right_x:.2f}_{right_y:.2f}.jpg'
            cv.imwrite(filename, img)
            print(f'Saved in: {filename}')
        if(MODE == 'REGISTER PHOTO'):
            switch_mode('NORMAL')
        else:
            LEFT = (-1, -1)
            RIGHT = (-1, -1)
            VIDEO.pop(0)

            
    elif(MODE =='NORMAL'):
        switch_mode('REGISTER PHOTO')

def SKIP(event):
    global MODE, VIDEO
    if(MODE == 'REGISTER VIDEO' and len(VIDEO) > 1):
        VIDEO.pop(0)

def RECORD(event):
    if(MODE =='NORMAL'):
        global RECORDING, VIDEO
        RECORDING = not RECORDING
        VIDEO.clear()

    
ROOT.bind("<Button-1>", LMB)
ROOT.bind("<Button-3>", RMB)
ROOT.bind("<Return>", ENTER)
ROOT.bind("s", SKIP)
ROOT.bind("r", RECORD)


normal_mode()
ROOT.mainloop()

