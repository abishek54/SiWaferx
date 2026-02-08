import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

img_size = 128
batch_size = 16

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    "data/train",
    target_size=(img_size, img_size),
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=batch_size
)

val_data = val_gen.flow_from_directory(
    "data/val",
    target_size=(img_size, img_size),
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=batch_size
)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 1)),
    MaxPooling2D(),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

es = EarlyStopping(patience=5, restore_best_weights=True)

model.fit(train_data, validation_data=val_data, epochs=25, callbacks=[es])

model.save("sem_defect_cnn.keras")
print("✅ Model saved as sem_defect_cnn.keras")
