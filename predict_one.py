import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "sem_defect_cnn.keras"
IMG_PATH = r"C:\Users\abish\OneDrive\Documents\Desktop\IESA\test.jpg"

IMG_SIZE = 128

model = tf.keras.models.load_model(MODEL_PATH)

img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img / 255.0
img = np.expand_dims(img, axis=-1)
img = np.expand_dims(img, axis=0)

pred = model.predict(img)
class_id = np.argmax(pred)

classes = list(model.class_names) if hasattr(model, 'class_names') else None
print("Predicted Class Index:", class_id)
print("Raw Probabilities:", pred)
