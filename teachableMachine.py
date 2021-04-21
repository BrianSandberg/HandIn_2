import math
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import json
import requests


# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
image = Image.open('whiskey.jpg')

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

#processedPrediction = []
# run the inference
prediction = model.predict(data)

#print(prediction)
listTest = (prediction*100).tolist()


#print(listTest[0])

listToString = str(listTest[0])
finalList = json.loads(listToString)

fastTest = open("labels.txt", "rt")
data = fastTest.read()
data.replace("2", "")
fastTest.close()


#Prints the prediction and rounds this down to the nearst integer
print("Prediction is ", math.floor(max(finalList)), "%")

#Creates a list of every line in the desired document
labels = open("labels.txt").readlines()

string = labels[finalList.index(max(finalList))][2:-1]




#Prints the index in the labels-list corresponding to the highest value in the finalList
print(labels[finalList.index(max(finalList))][2:-1])



r_temp = requests.get(url=("https://www.thecocktaildb.com/api/json/v1/1/filter.php?i=" + string))
drinks = r_temp.json()
print(drinks["drinks"][0]["strDrink"])


