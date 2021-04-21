import math
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import json

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
image = Image.open('test_hop.jpg')

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

#Prints the prediction and rounds this down to the nearst integer
print("Prediction is ", math.floor(max(finalList)), "%")

#Creates a list of every line in the desired document
labels = open("labels.txt").readlines()

#Prints the index in the labels-list corresponding to the highest value in the finalList
print(labels[finalList.index(max(finalList))])





