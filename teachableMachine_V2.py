import math
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import json
import requests
import os
from tkinter.filedialog import askopenfilename
import cv2
from tkinter import *

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model - This model can be changed freely to another
model = tensorflow.keras.models.load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)



def fileLoader():
    # Forces user to chose a valid .jpg file
    while True:
        # File selector der returner path til den valgte fil
        file = askopenfilename(filetypes =[("jpg","*.jpg")])
        # Splits path by "/" and gets the last index in the list
        fileToLoad = os.path.split(file)[-1]
        # Try to save the chosen .jpg file
        try:
            # saves the chosen image as a variable
            image = Image.open(fileToLoad)
            break
        # Doesnt break out while-loop if it cant save the image - Reopens the file choser
        except:
            print("No jpg chosen")

    # resize the image to a 224x224
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Display the resized image
    image.show()

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)

    listTest = (prediction*100).tolist()

    listToString = str(listTest[0])
    finalList = json.loads(listToString)

    # Creates a list of every line in the desired document
    labels = open("labels.txt").readlines()

    string = labels[finalList.index(max(finalList))][2:-1]

    # Prints the prediction and rounds this down to the nearst integer
    # Prints the index in the labels-list corresponding to the highest value in the finalList
    # Prints without the number on the label
    print("Prediction is ", math.floor(max(finalList)), "%", labels[finalList.index(max(finalList))][2:-1])
    print("\nDrinks containing this ingredient:")

    r_temp = requests.get(url=("https://www.thecocktaildb.com/api/json/v1/1/filter.php?i=" + string))
    drinks = r_temp.json()
    print(drinks["drinks"][0]["strDrink"])

    print("\nIngredients in this drink")
    r_temp2 = requests.get(
        url=("https://www.thecocktaildb.com/api/json/v1/1/lookup.php?i=" + drinks["drinks"][0]["idDrink"]))
    drinks = r_temp2.json()

    for i in range(1, 16):
        ingredients = drinks["drinks"][0]["strIngredient" + f'{i}']
        if ingredients is None:
            break
        else:
            print(ingredients)


def camera():
    # Argument is the index of the camera used. Only needed if multiple cameras are connected to the system
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("test")

    while True:
        # "frame" is the next frame in the camera and "ret" is a boolean representing the value of the frame.
        # False if it cant find a frame
        ret, frame = cam.read()
        # If ret == False
        if not ret:
            print("Couldnt find frame")
            break
        # Display frame in a window
        cv2.imshow("test", frame)

        # waitkey(1) is the number of ms between each frame - This is basically how often it checks for key input
        k = cv2.waitKey(1)
        # Checks if the "escape" key has been pressed on the keyboard - 27 is the value of escape
        if k % 256 == 27:
            print("Closing...")
            break
        # Checks for spacebar input
        elif k % 256 == 32:
            # Saves the jpg in a variable
            img_name = "opencv_frame_{}.jpg".format()
            # Saves the frame in the working directory with a specific name.jpg
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            break
    # closes the video stream and the windows
    cam.release()
    cv2.destroyAllWindows()

    # resize the image to a 224x224 with the same strategy as in TM2:
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(img_name, size, Image.ANTIALIAS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # display the resized image
    image.show()

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)

    # print(prediction)
    listTest = (prediction * 100).tolist()

    listToString = str(listTest[0])
    finalList = json.loads(listToString)

    # Creates a list of every line in the desired document
    labels = open("labels.txt").readlines()

    string = labels[finalList.index(max(finalList))][2:-1]

    # Prints the prediction and rounds this down to the nearest integer
    # Prints the index in the labels-list corresponding to the highest value in the finalList
    # Prints without the number on the label
    print("Prediction is ", math.floor(max(finalList)), "%", labels[finalList.index(max(finalList))][2:-1])
    print("\nDrinks containing this ingredient:")

    r_temp = requests.get(url=("https://www.thecocktaildb.com/api/json/v1/1/filter.php?i=" + string))
    drinks = r_temp.json()
    print(drinks["drinks"][0]["strDrink"])

    print("\nIngredients in this drink")
    r_temp2 = requests.get(url=("https://www.thecocktaildb.com/api/json/v1/1/lookup.php?i=" + drinks["drinks"][0]["idDrink"]))
    drinks = r_temp2.json()

    for i in range(1, 16):
        ingredients = drinks["drinks"][0]["strIngredient" + f'{i}']
        if ingredients is None:
            break
        else:
            print(ingredients)


# Creates the window with the size 400px x 400px
root = Tk()
root.geometry("400x400")
# Creates the two buttons in the window, and which method should be run when pressed
fileButton = Button(root, text="File", height=10, width=20, command = fileLoader)
cameraButton = Button(root, text="Camera", height=10, width=20, command = camera)
# Sets a preset location for the buttons
fileButton.pack(side = RIGHT)
cameraButton.pack(side = LEFT)
# mainloop is like a while-loop, telling the window to keep running - Until closed by root.destroy
root.mainloop()
