import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from numpy.linalg import norm
import os 

model = ResNet50(weights='imagenet', include_top=False,input_shape=(224, 224, 3))
model.trainable = False

model = tf.keras.Sequential([
    model,  
    GlobalAveragePooling2D()
])

def extract_features(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img = tf.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

filenames = []
for file in os.listdir('small_images'):
    filenames.append(os.path.join('small_images', file))


feature_list = []
for i, file in enumerate(filenames):
    print(f"Processing {i+1}/{len(filenames)}")
    features = extract_features(file, model)
    feature_list.append(features)


import pickle
pickle.dump(feature_list, open('feature_list_big.pkl', 'wb'))
pickle.dump(filenames, open('filenames_big.pkl', 'wb'))

