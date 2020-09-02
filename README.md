# Preprocessing
in ML, working with images might need pre-processing. In my case this project taking the explicit folder path and find all of the images then doing the below operations.

This project file consist of 3 part.

1- Image Conversion
Converting image into grey scale digital image which is the form of 128x128. For my project color information is not needed.

2- Data Normalization
I rearrange the color values with linear normalization formula, I mean, I projected the grayscale digital image in to a predefined range
which is common values 0 to 1.

3- Dataset Splitting
In machine learning, we try to test our learning capability. In this case we have to split the dataset into two parts which are test and train. I took 80% of the dataset as train set, and rest test set.
