import os
import shutil
import random

SOURCE_DIR = r"C:\Users\abish\OneDrive\Documents\Desktop\IESA\sem_dataset"
DEST_DIR = r"C:\Users\abish\OneDrive\Documents\Desktop\IESA\Dataset"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

classes = os.listdir(SOURCE_DIR)

for cls in classes:
    src_class_path = os.path.join(SOURCE_DIR, cls)
    images = os.listdir(src_class_path)
    random.shuffle(images)

    total = len(images)
    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    train_imgs = images[:train_count]
    val_imgs = images[train_count:train_count + val_count]
    test_imgs = images[train_count + val_count:]

    for img in train_imgs:
        src = os.path.join(src_class_path, img)
        dst = os.path.join(DEST_DIR, "Train", cls, img)
        shutil.copy(src, dst)

    for img in val_imgs:
        src = os.path.join(src_class_path, img)
        dst = os.path.join(DEST_DIR, "Validation", cls, img)
        shutil.copy(src, dst)

    for img in test_imgs:
        src = os.path.join(src_class_path, img)
        dst = os.path.join(DEST_DIR, "Test", cls, img)
        shutil.copy(src, dst)

    print(f"✅ {cls}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")

print("\n🎉 Dataset split completed successfully!")
