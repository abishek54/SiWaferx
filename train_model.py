import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# -------------------------
# 1. Reproducibility
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -------------------------
# 2. Paths & Params
# -------------------------
DATASET_PATH = r"C:\Users\abish\OneDrive\Documents\Desktop\IESA\sem_dataset"
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 30

# -------------------------
# 3. Data Load (GRAYSCALE SEM)
# -------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    color_mode="grayscale",
    subset="training",
    seed=SEED
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    color_mode="grayscale",
    subset="validation",
    shuffle=False,
    seed=SEED
)

NUM_CLASSES = train_gen.num_classes
print("Classes:", train_gen.class_indices)

# -------------------------
# 4. Optimized CNN (Edge-friendly + stable)
# -------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,1)),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),   # light regularization (not too much)
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.0005),  # slightly lower LR for stability
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------
# 5. Callbacks (Stability)
# -------------------------
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
    ModelCheckpoint("sem_defect_cnn_final.keras", monitor="val_accuracy", save_best_only=True)
]

# -------------------------
# 6. Train
# -------------------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)

# -------------------------
# 7. Evaluate
# -------------------------
val_loss, val_acc = model.evaluate(val_gen)
print(f"\n✅ Final Validation Accuracy: {val_acc*100:.2f}%")
print(f"❌ Final Validation Loss: {val_loss:.4f}")

# -------------------------
# 8. Save Model
# -------------------------
model.save("sem_defect_cnn_final.keras")
print("💾 Model saved as sem_defect_cnn_final.keras")
