# Face-Mask-Detection
COVID-19: Face Mask Detector, developed  a detection Model with 99% accuracy in training & testing. Automatically detect whether a person is wearing a mask or not in real-time video streams.

Table of Content
----------------
* Demo
* Overview
* Motivation
* Installation
* Features
* Steps/ Process req..
* Project structure
* Result/Summary
* Future scope of project

DEMO
----------
https://www.linkedin.com/posts/activity-6684432737618677760-a7Pz

https://www.youtube.com/watch?v=qI8rEYc5Dgo

![Covid-19 fask mask  (1)](https://user-images.githubusercontent.com/41515202/94375638-b44efc80-0132-11eb-9e43-da29de98f76b.png)

![Covid-19 fask mask  (8)](https://user-images.githubusercontent.com/41515202/94375641-b87b1a00-0132-11eb-9ee2-d3062c4b5f88.png)

![Covid-19 fask mask  (2)](https://user-images.githubusercontent.com/41515202/94375644-bca73780-0132-11eb-89d6-95bba169edc7.png)

![Covid-19 fask mask  (3)](https://user-images.githubusercontent.com/41515202/94375647-c0d35500-0132-11eb-909c-bd80ddc09d97.png)

![Covid-19 fask mask  (4)](https://user-images.githubusercontent.com/41515202/94375649-c335af00-0132-11eb-9e66-8d89dfb66029.png)

![Covid-19 fask mask  (5)](https://user-images.githubusercontent.com/41515202/94375652-c5980900-0132-11eb-8ca4-19a99785af92.png)

![Covid-19 fask mask  (6)](https://user-images.githubusercontent.com/41515202/94375653-c7fa6300-0132-11eb-99d6-11ee4fc19249.png)

![Covid-19 fask mask  (7)](https://user-images.githubusercontent.com/41515202/94375655-ca5cbd00-0132-11eb-9911-8db1eac31762.png)


Overview / What is it ??
------------------------
* COVID-19: Face Mask Detector with OpenCV, Keras/TensorFlow, and Deep Learning
* Automatically detect whether a person is wearing a mask or not in real-time video streams
* Our goal is to train a custom deep learning model to detect whether a person is or is not wearing a mask.

Motivation / Why/Reason ??
--------------------------------
* Our goal is to train a custom deep learning model to detect whether a person is or is not wearing a mask.
* Continuous rapidly Increase of covid virus importance of wearing a mask during the pandemics at these times also         has been increased.
* Universal mask use can significantly reduce virus transmission in communities
* Masks and face coverings can prevent the wearer from transmitting the COVID-19 virus to others and may provide some protection to the wearer. 

Installation / Tech Used
------------------------------
Dataset : Real World Masked Face Dataset (RMFD)
AI/DL Techniques/Libaries : OpenCV, Keras/TensorFlow, MobileNetV2

Features
---------
![image](https://user-images.githubusercontent.com/41515202/94375410-098a0e80-0131-11eb-8a9f-2a2df72359e7.png)

1-dataset consists of 1,376 images belonging to two classes:
    with_mask : 690 images
    without_mask : 686 images

2-Facial landmarks allow us to automatically infer the location of facial structures, including - Eyes, Eyebrows, Nose, Mouth, Jawline

3-Once we know where in the image the face is, we can extract the face Region of Interest (ROI).

![image](https://user-images.githubusercontent.com/41515202/94375362-bdd76500-0130-11eb-9fc3-73b67e2ad162.png)

4-And from there, we apply facial landmarks, allowing us to localize the eyes, nose, mouth, etc

![image](https://user-images.githubusercontent.com/41515202/94375376-d8a9d980-0130-11eb-8a62-8341b74fe2a3.png)

5- Next, we need an image of a mask (with a transparent background) such as the one below:

![image](https://user-images.githubusercontent.com/41515202/94375387-ee1f0380-0130-11eb-8a2c-0417b5de4288.png)

6- This mask will be automatically applied to the face by using the facial landmarks (namely the points along the chin and nose) to compute where the mask will be placed.
The mask is then resized and rotated, placing it on the face:

![image](https://user-images.githubusercontent.com/41515202/94375396-f9722f00-0130-11eb-82c4-e5ef61d86e2f.png)

7- Final Dataset with & without mask will be :::::

![image](https://user-images.githubusercontent.com/41515202/94375422-16a6fd80-0131-11eb-992c-07f7caa4862e.png)

8-Two-phase COVID-19 face mask detector::::

![image](https://user-images.githubusercontent.com/41515202/94375426-27f00a00-0131-11eb-82ac-11e28d0b0d95.png)

9- In order to train a custom face mask detector, we need to break our project into two distinct phases, each with its own respective sub-steps (as shown by Figure 1 above):::::

•	Training: Here we’ll focus on loading our face mask detection dataset from disk, training a model (using Keras/TensorFlow) on this dataset, and then serializing the face mask detector to disk
•	Deployment: Once the face mask detector is trained, we can then move on to loading the mask detector, performing face detection, and then classifying each face as with_mask or without_mask


STEPS/PROCESS REQ...
-----------------------
2.1. Data extraction
2.2. Building the Dataset class
2.3. Building our face mask detector model
2.4. Training our model
2.5. Testing our model on real data -> IMAGE/VIDEO
2.6. Results

Project structure::::
-------------------------
![image](https://user-images.githubusercontent.com/41515202/94375435-35a58f80-0131-11eb-99a2-e9e74ccb911f.png)

Require 3 Python scripts:::
-----------------------------
* train_mask_detector.py :  Accepts our input dataset and fine-tunes MobileNetV2 upon it to create our mask_detector.model. A training history plot.png. Containing accuracy/loss curves is also produced
* Detect_mask_video.py : Using your webcam, this script applies face mask detection to every frame in the stream
* Detect_mask_image.py : Performs face mask detection in static images

Next two sections, we will train our face mask detector.
-----------------------------------------------------------
1 - Implementing our COVID-19 face mask detector training script with Keras and TensorFlow::::::::

* we’ll be fine-tuning the MobileNet V2 architecture, a highly efficient architecture that can be applied to embedded devices with limited computational capacity (ex., Raspberry Pi, Google Coral, NVIDIA Jetson Nano, etc.)

* Reason : Deploying our face mask detector to embedded devices could reduce the cost of manufacturing such face mask detection systems, hence why we choose to use this architecture.

2- Training the COVID-19 face mask detector with Keras/TensorFlow

3 - Implementing our COVID-19 face mask detector for images with OpenCV

4 - COVID-19 face mask detection in images with OpenCV

5 - Implementing our COVID-19 face mask detector in real-time video streams with OpenCV

6 - Detecting COVID-19 face masks with OpenCV in real-time Video Streams 

STEPS FOR TRAINING MODEL
------------------------------
1 - From there, open up a terminal, and execute the following command:
$ python train_mask_detector.py --dataset dataset
[INFO] loading images...
[INFO] compiling model...
[INFO] training head...

Train for 34 steps, validate on 276 samples

Epoch 1/20
34/34 [==============================] - 30s 885ms/step - loss: 0.6431 - accuracy: 0.6676 -
val_loss: 0.3696 - val_accuracy: 0.8242

Epoch 2/20
34/34 [==============================] - 29s 853ms/step - loss: 0.3507 - accuracy: 0.8567 -
val_loss: 0.1964 - val_accuracy: 0.9375

Epoch 3/20
34/34 [==============================] - 27s 800ms/step - loss: 0.2792 - accuracy: 0.8820 -
val_loss: 0.1383 - val_accuracy: 0.9531

Epoch 4/20
34/34 [==============================] - 28s 814ms/step - loss: 0.2196 - accuracy: 0.9148 -
val_loss: 0.1306 - val_accuracy: 0.9492

Epoch 5/20
34/34 [==============================] - 27s 792ms/step - loss: 0.2006 - accuracy: 0.9213 -
val_loss: 0.0863 - val_accuracy: 0.9688

Epoch 16/20
34/34 [==============================] - 27s 801ms/step - loss: 0.0767 - accuracy: 0.9766 -
val_loss: 0.0291 - val_accuracy: 0.9922

Epoch 17/20
34/34 [==============================] - 27s 795ms/step - loss: 0.1042 - accuracy: 0.9616 -
val_loss: 0.0243 - val_accuracy: 1.0000

Epoch 18/20
34/34 [==============================] - 27s 796ms/step - loss: 0.0804 - accuracy: 0.9672 -
val_loss: 0.0244 - val_accuracy: 0.9961

Epoch 19/20
34/34 [==============================] - 27s 793ms/step - loss: 0.0836 - accuracy: 0.9710 -
val_loss: 0.0440 - val_accuracy: 0.9883

Epoch 20/20
34/34 [==============================] - 28s 838ms/step - loss: 0.0717 - accuracy: 0.9710 -
val_loss: 0.0270 - val_accuracy: 0.9922

[INFO] evaluating network...
 precision recall f1-score support
 with_mask 0.99 1.00 0.99 138
 without_mask 1.00 0.99 0.99 138
 accuracy 0.99 276
 macro avg 0.99 0.99 0.99 276
 weighted avg 0.99 0.99 0.99 276


![image](https://user-images.githubusercontent.com/41515202/97676743-16df4380-1ab7-11eb-9463-cfd5ad4e4b3a.png)


2 - training accuracy/loss curves demonstrate → high accuracy and little signs of
overfitting on the data.

3 - obtained ~99% accuracy on our test set.

4 - Looking at Figure , we can see there are little signs of overfitting, with the
validation loss lower than the training loss

5 - Given these results, we are hopeful that our model will generalize well to
images outside our training and testing set.

SUMMARY/RESULT
---------------
* Developed detection Model with 97% accuracy, automatically detect person is wearing a mask
or not in real-time video streams

* Extracted the face ROI & facial landmarks. And applied to the face by using the facial to compute.

* Used the highly efficient MobileNetV2 architecture & fine-tuned MobileNetV2 on our mask/no
mask dataset and obtained a classifier that is 97% accurate.

* Determined the class label encoding based on probabilities associated with color annotation

Future Scope
------------
Can be used in CCTV cameras for capturing wide peoples or group of peoples

Can be improved for further as per requiremnts.
