# Import libraries
import streamlit as st
from PIL import Image
import numpy as np 
import streamlit as st 
import pandas as pd
import model_pipeline as md
import data_pipeline as dp
import os 
import glob
import joblib

load_CLIP_model = joblib.load('CLIP_model.pkl')
#load_RESNET50_model = joblib.load('RESNET50_model.pkl')
user_image_path = ""

index_lst1 = []
score_lst1 = []
similar_actors_df1 = pd.DataFrame()

st.set_page_config(
    layout="wide"
)

st.markdown("<h1 style='text-align: center;'>Bollywood Actors Recommendation</h1>", unsafe_allow_html=True)

# Converting links to html tags
def path_to_image_html(path):
    return '<img src="' + path + '" width="200" >'

col1, col2, col3, col4 = st.columns((1.8, 2, 2, 2))

# Function to Read and Manupilate Images
def load_image(img):
    im = Image.open(img)
    image = np.array(im)
    return image

def save_uploadedfile(uploadedfile):
    cwd = os.getcwd()
    #print("CWD:", cwd)
    downloads_path = str(cwd+"/user_image")
    
     # create new directory if not exists
    if not os.path.exists(downloads_path):
        os.makedirs(downloads_path)   
        
    # Check is directory is empty or not
    if len(downloads_path) != 0:
        files = glob.glob(downloads_path+'/*')
        #print("files: ", file_details)
        for f in files:
            os.remove(f)
            
    picture_path  = os.path.join(downloads_path, uploaded_file.name)
    with open(picture_path, "wb") as f:
        f.write(uploadedfile.getbuffer())
    
    #print('picture_path =', picture_path)
    return  picture_path 

# Function for image to image similarity
def image_image_similarity(user_Photo_Path):    
    
    #print('user_Photo_Path =', user_Photo_Path)
    clip_image_arr = load_CLIP_model
    #resnet50_image_arr = load_RESNET50_model
    user_image_arr = md.user_Image_Embedding(user_Photo_Path)
    #print("user_image_arr.shape =", user_image_arr.shape)
    #print("user_image_arr[0] =", user_image_arr[0])
    '''
    if model == 'CLIP':
        image_arr = clip_image_arr
    elif model == 'RESNET50':
        image_arr = resnet50_image_arr
    else:
        print("No Model selected")
    '''
    image_arr = clip_image_arr
    #print("image_arr.shape =", image_arr.shape)
    #print("image_arr[0] =", image_arr[0])

    # Calculate dot product 
    scores = np.dot(user_image_arr.T, image_arr.T)/(np.linalg.norm(user_image_arr.T)*np.linalg.norm(image_arr.T))
    scores = np.interp(scores, (scores.min(), scores.max()), (0, 1))

    top_k = 5
    # get the top k indices for most similar vecs
    idx = np.argsort(-scores)[:top_k]
    print("top 5:",idx)
    index_lst1.clear()
    score_lst1.clear()
    
    # display the results
    for i in idx:
        index_lst1.append(i)
        score_lst1.append(scores[i])
    
    similar_actors_df1.iloc[0:0]
    similar_actors_df1['index'] = index_lst1
    similar_actors_df1['score'] = score_lst1
    print("similar_actors_df1=", similar_actors_df1)


with st.form("my_form"):
    st.markdown("### Upload your Photo")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        uploaded_file = st.file_uploader(label=" Upload image", type=['jpg', 'png', 'jpeg'])
        if uploaded_file is not None:
            file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type}
            user_image = load_image(uploaded_file)
            st.image(user_image, width=150)
            #st.write(file_details)
            user_image_path = save_uploadedfile(uploaded_file)

    submitted = st.form_submit_button("Show Matching Actors Faces")

    if submitted:
        image_image_similarity(user_image_path)
        df = dp.load_actors_image_dataset()
        print("similar_actors_df1=", similar_actors_df1)
                 
        with col3:
            for i in similar_actors_df1.index:
                for j in df.index:
                    if j == similar_actors_df1['index'][i]:   
                        name = df['Name'][j]
                        score = similar_actors_df1['score'][i]
                        st.image(df['Image_Path'][j], width=150)
                        st.write(f"Name: {name}  score: {score}")
                     