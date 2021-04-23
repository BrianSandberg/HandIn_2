import math
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import json
import requests
import os
from tkinter.filedialog import askopenfilename

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model - This model can be changed freely to another
model = tensorflow.keras.models.load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model.
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

#Forces user to chose a valid .jpg file
while True:
    # File selector der returner path til den valgte fil
    file = askopenfilename(filetypes=[("jpg", "*.jpg")])
    # Splits path by "/" and gets the last index in the list
    fileToLoad = os.path.split(file)[-1]
    # Try to save the chosen .jpg file
    try:
        # saves the chosen image as a variable.
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

listTest = (prediction * 100).tolist()

listToString = str(listTest[0])
finalList = json.loads(listToString)

#Creates a list of every line in the desired document
#Its important that every line in the labels list can be used in the API, for valid response
labels = open("labels.txt").readlines()
#Creates a string for the API, based on the highest number in the labels list
#[2:-1] doesnt account for index 0 and 1
string = labels[finalList.index(max(finalList))][2:-1]

# Prints the prediction and rounds this down to the nearst integer
# Prints the index in the labels-list corresponding to the highest value in the finalList
# Prints without the number on the label
print("Prediction is ", math.floor(max(finalList)), "%", labels[finalList.index(max(finalList))][2:-1])
print("\nDrinks containing this ingredient:")

#Sets up the API request with the string created earlier
r_temp = requests.get(url=("https://www.thecocktaildb.com/api/json/v1/1/filter.php?i=" + string))
drinks = r_temp.json()
#Prints from the API response
print(drinks["drinks"][0]["strDrink"])

#Creates a new API request based on the previous request
print("\nIngredients in this drink")
r_temp2 = requests.get(url=("https://www.thecocktaildb.com/api/json/v1/1/lookup.php?i=" + drinks["drinks"][0]["idDrink"]))
drinks = r_temp2.json()

#Prints the ingredients for the requested drink, if the value of the ingredient is not "None"
for i in range(1, 16):
    ingredients = drinks["drinks"][0]["strIngredient" + f'{i}']
    if ingredients is None:
        break
    else:
        print(ingredients)
