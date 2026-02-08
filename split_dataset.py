import os, shutil, random

base = "sem_dataset"
train_dir = "data/train"
val_dir = "data/val"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

for cls in os.listdir(base):
    src = os.path.join(base, cls)
    images = os.listdir(src)
    random.shuffle(images)

    split = int(0.8 * len(images))
    train_imgs = images[:split]
    val_imgs = images[split:]

    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

    for img in train_imgs:
        shutil.copy(os.path.join(src, img), os.path.join(train_dir, cls, img))
    for img in val_imgs:
        shutil.copy(os.path.join(src, img), os.path.join(val_dir, cls, img))

print("✅ Train / Validation split completed")
