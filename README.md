# bollywood-actors-recommendation-AIML-Project

@author: Nikhil Jagnade
@date: 05 March 2023

Bollywood actors recommendation project suggest images of actors on the basis of images provided by the user. In this project, I have used image embedding techniques to compare similiar images and recommend the best possible matched images as per dot product score. 

Image embedding refers to the process of representing an image as a vector of numbers or features, which can be used as input to machine learning algorithms. These features capture the important characteristics of the image, such as color, texture, and shape, and can be used for tasks such as image classification, object detection, and image retrieval.

Dataset link: https://www.kaggle.com/datasets/iamsouravbanerjee/indian-actor-images-dataset

In this Dataset, we have 6750 Indian Actor (Male and Female) Images in 135 different categories or classes.

Model used - CLIP model for image embedding in batch

score matrix - dot product

User Interface - Streamlit App

Image retrieval: Given an input image, retrieve similar images from a database based on their image features.

<img width="1714" alt="image" src="https://user-images.githubusercontent.com/64134540/222963482-2ef24cb8-0b62-4da5-98d1-5ce407596b1e.png">

<img width="1695" alt="image" src="https://user-images.githubusercontent.com/64134540/222963641-88bdb31e-f528-44ce-91c2-30c2465b88d3.png">


### Stpes to follow

1. download project folder from github repo to local machine

2. Load project in integrated development environment (IDE) you are using such as Pycharm, vscode etc

3. Select the virtual environment where latest version of python is installed

4. Install libraries listed in requirements.txt file

5. run command on the terminal to run streamlit app > streamlit run face_match_streamlitapp.py                              

6. project will run on your default Web browser with localhost and port number - Safari, googlechrome etc
