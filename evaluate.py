import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# -------------------------
# Paths & Params
# -------------------------
DATASET_PATH = r"C:\Users\abish\OneDrive\Documents\Desktop\IESA\sem_dataset"
MODEL_PATH = "sem_defect_cnn_final.keras"
IMG_SIZE = (128, 128)
BATCH_SIZE = 16

# -------------------------
# Load Model
# -------------------------
model = tf.keras.models.load_model(MODEL_PATH)

# -------------------------
# Load Validation/Test Data
# -------------------------
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

class_names = list(val_gen.class_indices.keys())
print("Classes:", class_names)

# -------------------------
# Predictions
# -------------------------
y_true = val_gen.classes
y_pred_probs = model.predict(val_gen)
y_pred = np.argmax(y_pred_probs, axis=1)

# -------------------------
# Classification Report
# -------------------------
print("\n📊 Classification Report:\n")
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print(report)

with open("classification_report.txt", "w") as f:
    f.write(report)

# -------------------------
# Confusion Matrix
# -------------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - SEM Defect Classifier")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

print("✅ Confusion matrix saved as confusion_matrix.png")

# -------------------------
# Model Size
# -------------------------
model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
print(f"\n📦 Model Size: {model_size_mb:.2f} MB")

with open("model_size.txt", "w") as f:
    f.write(f"Model size: {model_size_mb:.2f} MB\n")

print("✅ Model size saved in model_size.txt")
