from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASET_PATH = r"C:\Users\abish\OneDrive\Documents\Desktop\IESA\sem_dataset"

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(128,128),
    batch_size=16,
    class_mode="categorical",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(128,128),
    batch_size=16,
    class_mode="categorical",
    subset="validation"
)

print("Train samples:", train_gen.samples)
print("Val samples:", val_gen.samples)
print("Classes:", train_gen.class_indices)
