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


feature_list = np.array(pickle.load(open('feature_list_big.pkl', 'rb')))
filenames = pickle.load(open('filenames_big.pkl', 'rb'))

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

def feature_extraction(img, model):
    # Resize image to 224x224
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    expanded_img = np.expand_dims(img_array, axis=0)
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
   
    display_image = Image.open(uploaded_file).convert('RGB')
    st.image(display_image, caption="Uploaded Image", use_container_width=True)
    
    
    with st.spinner('Extracting features...'):
        features = feature_extraction(display_image, model)
        indices = recommend(features, feature_list)

    
    st.subheader("Recommended for you:")
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            # Ensure filenames[indices[0][i]] points to a valid path in your GitHub repo
            st.image(filenames[indices[0][i]], use_container_width=True)