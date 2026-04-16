model.py:

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

IMG_SIZE = 224

# Load trained model once
model = tf.keras.models.load_model("pneumonia_mobilenetv2_binary.h5")

def predict_pneumonia(img_path):

    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    pred = model.predict(img_array)[0][0]

    if pred > 0.5:
        return "PNEUMONIA", float(pred)
    else:
        return "NORMAL", float(1 - pred)

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def generate_gradcam(img_path):
    last_conv_layer_name = "Conv_1"  # MobileNetV2 last conv layer

    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Create model that outputs conv layer + prediction
    grad_model = tf.keras.models.Model(
    inputs=model.input,
    outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    # Global average pooling
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)

    # Convert to image
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Load original image
    original_img = cv2.imread(img_path)
    original_img = cv2.resize(original_img, (224, 224))

    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

    output_path = f"uploads/heatmap_{os.path.basename(img_path)}"
    cv2.imwrite(output_path, superimposed_img)

    return output_path
