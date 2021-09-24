import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import PIL
import cv2


# Now we have to load the npz image files
# This X variable will be used as our input(training)
X = np.load("image.npz")["arr_0"]
print(X)  # returns a 2-dimensonal array

# This Y Variable will be our result set
Y = pd.read_csv("data.csv")["labels"]
print(Y.head())  # prints the labels

# This value_counts() will show the number of occurances of each label in the data
print(pd.Series(Y).value_counts())

classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
           "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
num_classes = len(classes)

# Now we have to split the data into train and test sets
train_X, test_X, train_Y, test_Y = train_test_split(
    X, Y, train_size=7500, test_size=2500)

# Now we have to scale the train_X and test_X
# so to scale the images we cannot use StandardScaler()
# so to scale the data we have to divide the train_X and test_X each by 255.0
train_X = train_X/255.0
test_X = test_X/255.0

# print(train_X)
# print(test_X)

# now its time to train our model
model = LogisticRegression(solver='saga', multi_class='multinomial')
model.fit(train_X, train_Y)

# now our model will predict the test_Y by taking the test_X(input factors) as input
prediction = model.predict(test_X)
# print(prediction)  # prints an array of predictions

# # accuracy
Accuracy = accuracy_score(test_Y, prediction)
print(f"Accuracy :- {Accuracy}")  # Good 99% accuracy

cameraObj = cv2.VideoCapture(0)

while(True):
    try:
        _ret, frame = cameraObj.read()

        # first we will convert the frame into a grayscale image
        grayscaled_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # get the height and width of the whole video btw this will return a tuple
        height, width = grayscaled_image.shape()

        # we will draw a box on the center of the video that will be our region of interest
        upper_left = (int(width / 2 - 56), int(height / 2 - 56))
        bottom_right = (int(width / 2 + 56), int(height / 2 + 56))

        # now we will draw or plot the rectangle
        cv2.rectangle(grayscaled_image, upper_left,
                      bottom_right, (0, 255, 0), 2)

        #  we are creating a box in the video because we want the user to keep the digit inside the box
        #  our model will give more priority to the digit which is inside the box
        # it is also called roi = region of interest
        roi = grayscaled_image[upper_left[1]:bottom_right[1],
                               upper_left[0]:bottom_right[0]]

        # Converting cv2 image to pil format
        image_pil = Image.fromarray(roi)

        # converting to grayscale image - 'L' format means each pixel is
        # represented by a single value from 0 to 255
        image_bw = image_pil.convert('L')

        # resizing the image and also pixalating the image
        image_bw_resized = image_bw.resize((28, 28), Image.ANTIALIAS)

        # generally in Images and also in mirrors "the mirror efect" occurs
        # so we are inverting the resized image here
        image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)

        # applying a pixel filter to filter the pixalated image
        pixel_filter = 20
        min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter)

        # clipping the image here (cutting or taking a particular part of the image)
        image_bw_resized_inverted_scaled = np.clip(
            image_bw_resized_inverted-min_pixel, 0, 255)

        # taking the maximum pixel from the inverted image
        max_pixel = np.max(image_bw_resized_inverted)

        image_bw_resized_inverted_scaled = np.asarray(
            image_bw_resized_inverted_scaled)/max_pixel

        # reshape() function returns the same array but with a new shape
        test_sample = np.array(
            image_bw_resized_inverted_scaled).reshape(1, 784)

        # our model is predicting by taking the test_sample
        test_pred = model.predict(test_sample)

        print(f"Predicted Class is :- {test_pred}")

        cv2.imshow('frame', grayscaled_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        pass

cameraObj.release()
cv2.destroyAllWindows()


# !!! its not working , Its not giving any error message and I am fustrated
# !!!  the problem is that it is not showing the camera window in line 118
