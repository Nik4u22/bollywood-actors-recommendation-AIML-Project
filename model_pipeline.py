from PIL import Image
from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel
import torch
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import cv2
from tqdm.auto import tqdm
import os
import data_pipeline as dp
import joblib

# Load dataset        
dataset = dp.load_actors_image_dataset()

def CLIP_Model():
    # if you have CUDA or MPS, set it to the active device like this
    #device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
    model_id = "openai/clip-vit-base-patch32"

    # we initialize a tokenizer, image processor, and the model itself
    tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id, from_tf=True).to(device)
    '''
    image_arr = []
    
    for index, row in dataset.iterrows():
        image_path = row['Image_Path']
        if image_path != "":
            ## Create Image Embedding
            image = processor(
                text=None,
                images=Image.open(image_path),
                return_tensors='pt'
            )['pixel_values'].to(device)
         
            image_emb = model.get_image_features(image)
            # convert to numpy array
            image_emb = image_emb.squeeze(0)
            image_emb_arr = image_emb.cpu().detach().numpy()
            #print("image_emb_arr:",image_emb_arr)
                
            # Normalization
            image_emb_arr = image_emb_arr / np.linalg.norm(image_emb_arr, axis=0)
            #print("user_image_arr after norm:",user_image_arr)
            image_emb_arr.min(), image_emb_arr.max()
            image_arr = np.concatenate((image_arr, image_emb_arr), axis=0)

    '''
    images = [Image.open(dataset['Image_Path'][i]) for i in dataset.index]
    #print(images)
    
    batch_size = 16
    image_arr = None

    for i in tqdm(range(0, len(images), batch_size)):
        # select batch of images
        batch = images[i:i+batch_size]
        # process and resize
        batch = processor(
            text=None,
            images=batch,
            return_tensors='pt',
            padding=True
        )['pixel_values'].to(device)
        
        # get image embeddings
        batch_emb = model.get_image_features(pixel_values=batch)
        # convert to numpy array
        batch_emb = batch_emb.squeeze(0)
        
        batch_emb = batch_emb.cpu().detach().numpy()
    
        # add to larger array of all image embeddings
        if image_arr is None:
            image_arr = batch_emb
        else:
            image_arr = np.concatenate((image_arr, batch_emb), axis=0)
    #print("image_arr shape:", image_arr.shape, len(image_arr))
    
    # Normalization
    image_arr = image_arr / np.linalg.norm(image_arr, axis=0)
    image_arr.min(), image_arr.max()
    
    #print("After Normalization image_arr shape:", image_arr.shape, len(image_arr))
    
    # Saving the model - joblib
    #joblib.dump(image_arr, 'CLIP_model.pkl')

def RESNET50_Model():
    
    # if you have CUDA or MPS, set it to the active device like this
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = ResNet50(include_top=False, weights='imagenet', pooling='avg')
    
    '''
    image_arr = []
    
    for index, row in dataset.iterrows():
        image_path = row['Image_Path']
        if image_path != "":
            img = image.load_img(image_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            image_emb = model.predict(x)
            # convert to numpy array
            image_emb_arr = image_emb.squeeze(0)
                
            # Normalization
            image_emb_arr = image_emb_arr / np.linalg.norm(image_emb_arr, axis=0)
        
            image_emb_arr.min(), image_emb_arr.max()
            image_arr = np.concatenate((image_arr, image_emb_arr), axis=0)
    
    
    '''
    
    batch_size = 16
    image_arr = None

    #images = [Image.open(dataset['Image_Path'][i]) for i in dataset.index]
    images = []
    
    for index, row in dataset.iterrows():
        image_path = row['Image_Path']
        if image_path != "":
            img = image.load_img(image_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            image_emb = model.predict(x)
            images.append(image_emb)
    
    #print("images length:", len(images))
    for i in tqdm(range(0, len(images), batch_size)):
        # select batch of images
        batch = images[i:i+batch_size]
        #print("batch length:", len(batch))
        # get image embeddings
        # convert to numpy array
        batch_emb = np.asarray(batch)
        #print("batch_emb  shape before=", batch_emb.shape)
        batch_emb = np.reshape(batch_emb, (-1, 512))
        #print("batch_emb  shape after=", batch_emb.shape)
        # add to larger array of all image embeddings
        #print("batch_emb  shape 3=", batch_emb.shape, type(batch_emb))
        if image_arr is None:
            image_arr = batch_emb
        else:
            image_arr = np.concatenate((image_arr, batch_emb), axis=0)
    
    #print("image_arr shape:", image_arr.shape, type(image_arr), len(image_arr))
    
    # Normalization
    image_arr = image_arr / np.linalg.norm(image_arr, axis=0)
    image_arr.min(), image_arr.max()
    
    #print("After Normalization image_arr shape:", image_arr.shape, len(image_arr))
    
    # Saving the model - joblib
    joblib.dump(image_arr, 'RESNET50_model.pkl')   

def user_Image_Embedding(user_image_path):
    # if you have CUDA or MPS, set it to the active device like this
    #device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
    model_id = "openai/clip-vit-base-patch32"

    # we initialize a tokenizer, image processor, and the model itself
    tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id, from_tf=True).to(device)
    
    if user_image_path != "":
    ## Create Image Embedding
        image = processor(
            text=None,
            #images=Image.open(dataset['product_image'][0]),
            images=Image.open(user_image_path),
            return_tensors='pt'
        )['pixel_values'].to(device)
        #print("user_image shape:",image.shape)

        user_image_emb = model.get_image_features(image)
        #print("user_image_emb shape:",user_image_emb.shape)
        # convert to numpy array
        user_image_emb = user_image_emb.squeeze(0)
        user_image_emb_arr = user_image_emb.cpu().detach().numpy()
        #print("user_image_emb arr:",user_image_emb_arr)
            
        # Normalization
        user_image_arr = user_image_emb_arr / np.linalg.norm(user_image_emb_arr, axis=0)
        #print("user_image_arr after norm:",user_image_arr)
        user_image_arr.min(), user_image_arr.max()
        
        return user_image_arr

#RESNET50_Model()
#CLIP_Model()