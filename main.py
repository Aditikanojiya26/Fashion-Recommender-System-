import streamlit as st
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pickle


feature_list = np.array(pickle.load(open('feature_list.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False,input_shape=(224, 224, 3))
model.trainable = False

model = tf.keras.Sequential([
    model,  
    GlobalAveragePooling2D()
])

st.title("Fashion Recommendation System")

def save_uploaded_file(uploaded_file):
    try: 
        with open(os.path.join("uploads", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img = tf.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distance, indices = neighbors.kneighbors([features])
    return indices
    

uploaded_file = st.file_uploader("Upload an image of clothing item", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        st.image(Image.open(uploaded_file))
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
        indices = recommend(features, feature_list)
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
          st.image(filenames[indices[0][0]], width=150)

        with col2:
            st.image(filenames[indices[0][1]], width=150)

        with col3:
            st.image(filenames[indices[0][2]], width=150)

        with col4:
            st.image(filenames[indices[0][3]], width=150)

        with col5:
            st.image(filenames[indices[0][4]], width=150)


    else:
        st.error("Error uploading file")
    