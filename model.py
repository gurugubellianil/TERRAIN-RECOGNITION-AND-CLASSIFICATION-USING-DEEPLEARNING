import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model # type: ignore
import tensorflow_hub as hub
import kagglehub

# Function to load and preprocess the image
def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    data=[]
    resized_img = cv2.resize(img, (224, 224))
    data.append(resized_img)
    # resized_img = resized_img.astype('float32') / 255.0
    # return np.expand_dims(resized_img, axis=0)
    return np.array(data)

# Function to load the Vision Transformer model
def load_vision_transformer_model():
    # Download latest version
    path = kagglehub.model_download("spsayakpaul/vision-transformer/tensorFlow2/vit-b16-classification")
    vit_model = hub.KerasLayer(path, trainable=True)
    vit_lambda = tf.keras.layers.Lambda(lambda x: vit_model(x))
    def se_block(input_tensor, ratio=16):
        channels = input_tensor.shape[-1]
        x = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
        x = tf.keras.layers.Dense(channels // ratio, activation='relu')(x)
        x = tf.keras.layers.Dense(channels, activation='sigmoid')(x)
        x = tf.keras.layers.Reshape((1, channels))(x)
        return tf.keras.layers.Multiply()([input_tensor, x])
 
    se_block_lambda = tf.keras.layers.Lambda(lambda x: se_block(x))
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = se_block_lambda(inputs)
    x = vit_lambda(x)
    vit_se_model=Model(inputs,x)
    return vit_se_model

# Function to perform prediction using the loaded model
def predict_image(image_path, model):
    # Load and preprocess the image
    data = load_and_preprocess_image(image_path)

    # Get prediction
    dt = model(data)
    dt_m=np.reshape(dt,(dt.shape[0],1,dt.shape[1]))
    model = tf.keras.models.load_model('ViT_model@1.keras')
    fr = model(dt_m).numpy()
    if (np.max(fr) <= 0.85):
        predicted_class = 4
    else :
        predicted_class = np.argmax(fr)
    return predicted_class

if __name__ == "__main__":
    # Load the Vision Transformer model
    vit_model = load_vision_transformer_model()

    # Load your saved model

    # Path to the image you want to predict
    image_path = "C:/Users/gurug/OneDrive/Documents/Datasets/archive (1)/Data Main/test/Rocky/Rocky (1)_2_27.png"

    # Perform prediction
    predicted_class_index = predict_image(image_path, vit_model)
    
    # Dictionary to map class index to class label
    class_labels = {0: 'Grassy', 1: 'Marssy', 2: 'Rocky', 3: 'Sandy', 4: 'Invalid Image'}

    # Print the predicted class label
    print("Predicted Class:", class_labels[predicted_class_index])
