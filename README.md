# Human-Activity-Recognition
Human Activity Recognition interprets and classifies human movements, postures and activities using CNN

 ```
# Model Creation

This file contains the code to create and train a machine learning model. The model is a convolutional neural network (CNN) that can be used to classify images. The model is created using the Keras deep learning library.

## Step 1: Import the necessary libraries

The first step is to import the necessary libraries. The following libraries are imported:

```
import keras
import numpy as np
import matplotlib.pyplot as plt
```

## Step 2: Load the data

The next step is to load the data. The data is loaded from a CSV file. The CSV file contains the following columns:

* `image_id`: The ID of the image.
* `image`: The image data.
* `label`: The label of the image.

The following code is used to load the data:

```
data = pd.read_csv('data.csv')
```

## Step 3: Preprocess the data

The next step is to preprocess the data. The data is preprocessed by normalizing the pixel values. The pixel values are normalized by dividing them by 255. The following code is used to normalize the pixel values:

```
data['image'] = data['image'] / 255
```

## Step 4: Split the data into training and testing sets

The next step is to split the data into training and testing sets. The data is split into training and testing sets using the `train_test_split()` function from the `sklearn` library. The following code is used to split the data into training and testing sets:

```
X_train, X_test, y_train, y_test = train_test_split(data['image'], data['label'], test_size=0.2)
```

## Step 5: Create the model

The next step is to create the model. The model is created using the `Sequential` class from the `keras` library. The model consists of the following layers:

* `Conv2D`: A convolutional layer with 32 filters and a kernel size of 3x3.
* `MaxPooling2D`: A max pooling layer with a pool size of 2x2.
* `Conv2D`: A convolutional layer with 64 filters and a kernel size of 3x3.
