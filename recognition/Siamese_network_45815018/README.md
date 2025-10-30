# Siamese Network Classification on ISIC 2020 Kaggle Dataset using Contrastive Loss

## Problem to solve

Using the ISIC 2020 kaggle dataset and being able to classify images as benign or malignant skin images. Where benign is labeled as 0 and malignant is labeled as 1. The main difficulty with this problem is that around 98% were benign images and only 2% were malignant images, showing a severe imbalance in the classes.

## Model Description and Loss Function 

The Siamese architecture backbone is based off the following Figure 1 seen below where the input into the model is a pair of images and a pair label with 0 for different classes and 1 for similar classes.:

![Figure 1](./README_images/siamese_network_image.webp)

Figure 1: Siamese Network Layout [1]

Minor modifications have been done to this layout such as different kernel sizes and fully connected layer amounts, as well as more batch normalization and max pooling layers being used. The main significance is that this model has a head layer of fully connected layers that converts the outputted embeddings into a probability between the two skin classes.

Because the model outputs probabilities the Binary Cross Entropy loss function has been utilized to optimize the model in predicting the correct classes. This loss function can be seen below:

L<sub>BCE</sub> = $-\frac{1}{N}\sum_{i=1}^{N}\left[y_i\log(p_i) + (1 - y_i)\log(1 - p_i)\right]$

[2]

## Data Preprocessing

The train data was split into the following datasets train/validation/test with the following ratio 0.75/0.15/0.10. This allowed the creation of three datasets to train, validate and then test the model on. The original train dataset has around 33,000 images this split allows for each dataset to have sufficient data to train properly, receive valid validation metrics with losses and have a sufficiently large test set.

Further preprocessing was done where the images were resized to 224x224. Three data augmentations were done doing training which were random vertical flip, random horizontal flip, and finally random rotation at 20 degrees. These augmentations were done to provide the model more training variety, with the goal of increasing the models generalization capabilities. The RGB pixel intensity values were then scaled to a range of 0 to 1 and then normalized with mean = 0.5 and standard deviation = 0.5 for each pixel intensity.

A random weighted sampler was created for the train dataloader which allowed the loader to over sample the marginalized malignant class to provide more sample for the model to train on.

## References

[1] Sean Benhur, “A Friendly Introduction to Siamese Networks,” Built In, April 3 2025. [Online]. Available: https://builtin.com/machine-learning/siamese-network. [Accessed: Oct. 30 2025].

[2] Sanchhaya Education Private Ltd., “Binary Cross Entropy/Log Loss for Binary Classification,” GeeksforGeeks, July 23 2025. [Online]. Available: https://www.geeksforgeeks.org/deep-learning/binary-cross-entropy-log-loss-for-binary-classification/. [Accessed: Oct. 30 2025].