import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASET_PATH = r"C:\Users\abish\OneDrive\Documents\Desktop\IESA\sem_dataset"
IMG_SIZE = (128, 128)
BATCH_SIZE = 16

# Load model
model = tf.keras.models.load_model("sem_defect_cnn_final.keras")

# Load validation data
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    color_mode="grayscale",
    subset="validation",
    shuffle=False
)

# Evaluate
loss, acc = model.evaluate(val_gen)
print(f"✅ Validation Accuracy: {acc*100:.2f}%")
print(f"❌ Validation Loss: {loss:.4f}")
