import os

base = "sem_dataset"

for c in os.listdir(base):
    print(c, len(os.listdir(os.path.join(base, c))))
