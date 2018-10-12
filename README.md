# Frost-Research-2018
1 and 2 (overlapping) handwritten digit recognition using capsule networks

This work was completed under Professor Mark Stankus\*, with Alexa White\*, Grant Bernosky\*, and Tim Royston\*

> We chose to use Geoffrey Hinton's Capsule Network 
> Architecture to tackle the issue of overlapping 2 digit images.
> However the readily available MNIST dataset only contains 
> single digit numbers.

 Image Stitcher 
---
This file takes image from the MNIST dataset, and puts it on top of another image from MNIST, overlapping the 2nd image by a randomly sampled value from a pixel range i.e. 0-25. The new image is resized to a 28x56 numpy array containing both digits.

> Now that we can created multi-digit images, we now want to
> create sets of these images, for training, validation, and 
> testing.

Generate Data
---
Train, validation, and test data are returned from this function,
with the appropriate parameters: training length, validation split, etc.

> The 3 datasets are returned as a tuple containing 3 Data\_np
> objects, which take the stitched images, the labels 
> corresponding to each image, and a batch\_size as arguments. 

Data Iterator
---
 The file data\_iterator.py contains the Data\_np class used to pass the image data to the models we train in batches of batch\_size. The reset method is used to start the next batch at the beginning of the image data. This method should be called after an epoch has ended and before the next epoch begins.

> With our stitched image data placed into Data\_np objects we 
> can now begin training. There is a single digit model and a
> double digit model, both of which were based on the code
> written by Aurélien Geron found here: [Capsule Network Code](https://github.com/ageron/handson-ml/blob/master/extra_capsnets.ipynb)

- Thank you Aurélien! :D

Double Caps and Single Caps
---
Both models are capable of running on multiple GPUs thanks to Vahid Kazemi and his 
[Data Parallelism Code](https://github.com/vahidk/EffectiveTensorflow#multi_gpu).
The main difference between these files is the double_caps.py 
saves the output of the decoder network to find the order of the digits in the 2 digit images.

For the final models we received the following results 

1. single_caps test accuracy: 99.1987%
2. double_caps test accuracy: 98.8381%

> If an image containing an '4' and a '2' is given to double\_caps,
> the model will recognize the '4' and '2' in the image but it will
> not tell us if the image contains a '42' or a '24'. For that we 
> need find\_better\_center.py and order\_finder.py.

Find Better Center
---
Using the concept of "center of mass", we use the center of mass equation to find the middle column of a reconstructed double digit image from the double\_caps model. This image can be reconstructed into 2 separate images, each containing 1 of the digits in the original double digit image.

The number of nonzero pixel values in each column is summed up and multiplied by its corresponding column number (starting from left to right, with the first column number as 1). The sum of those products is divided by the total number of nonzero pixels in the image rounded to the nearest integer. This value is the column number corresponding the digit's horizontal center of mass.

Order Finder
---
The code in order_finder.py computes the center of mass for each image containing 1 digit, reconstructed from the 2 digit image, compares them, and outputs a string representing the correct order of the digits.

 i.e. if the '2' center of mass is greater than '4', the string returned would be '42'

Cams Net
---
This is a dense neural network used to determine whether there are 1 or 2 digits in a given image. This small network worked surprisingly well, achieving a test accuracy of above 0.999 and trained in under 30 seconds.

Restore All
---
This is where all the pieces come together. New handwritten data can be converted to numpy arrays, put in a Data\_np object and run through cams\_net to determine which images have 1 digit and which contain 2 digits. The 1 digit data would be passed to the restored single\_caps model and the 2 digit data would be passed to the restored double\_caps model. The predictions from both networks would be printed out and the images/results saved if desired. 

\* Frost Research Fellows, funded by the Bill and Linda Frost Fund, Cal Poly, San Luis Obispo

