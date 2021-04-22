import math
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import json
import requests
import glob,os
from pathlib import Path
from tkinter import *
from tkinter.filedialog import askopenfilename
import cv2
import tkinter

i = 0

def fileLoader():
    i = 1

def useCamera():
    i = -1

top = Tk()

top.geometry("400x300")
camera = Button(top,text = "Use camera", command = useCamera())
camera.pack(side = RIGHT)
fileLoader = Button(top, text ="Choose file", command = fileLoader())
fileLoader.pack(side = LEFT)
top.mainloop()

# Forces user to chose a valid .jpg file
while i is 1:
    # File selecter der returner path til den valgte fil
    yas = askopenfilename(filetypes=[("jpg", "*.jpg")])
    # Splits path by "/"
    yas2 = os.path.split(yas)[-1]
    # En lappeløsning, men fjerne hele den første del af path hen til filen
    imageToLoad = yas2

    try:
        # Replace this with the path to your image
        image = Image.open(yas2)
        break
    except:
        print("No jpg chosen")

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

while i is -1:
    ret, frame = cam.read()
    if not ret:
        print("Couldnt find frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.jpg".format()
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        break

cam.release()

cv2.destroyAllWindows()

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

#resize the image to a 224x224 with the same strategy as in TM2:
#resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.ANTIALIAS)

#turn the image into a numpy array
image_array = np.asarray(image)

# display the resized image
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


#Creates a list of every line in the desired document
labels = open("labels.txt").readlines()

string = labels[finalList.index(max(finalList))][2:-1]

#Prints the prediction and rounds this down to the nearst integer
#Prints the index in the labels-list corresponding to the highest value in the finalList - Prints without the number on the label
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

#Skal vi kun printe den første drink og dens ingredienser ud? Og så måske se om vi kan udvide modellen en lille smule






