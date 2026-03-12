import os
import shutil
from pathlib import Path

# === CONFIGURATION ===
BASE = Path.home() / "Desktop" / "fcml-face"
incoming = BASE / "photos"                 # folder where you copied all 45 images
dataset = BASE / "dataset" / "Spain"

# personalities (must match your folder names)
people = [
    "Pedro_Sanchez",
    "Penelope_Cruz",
    "Rafael_Nadal",
    "Rosalia",
    "Fernando_Alonso",
]

# make sure all destination folders exist
for p in people:
    (dataset / p / "ref").mkdir(parents=True, exist_ok=True)
    (dataset / p / "test").mkdir(parents=True, exist_ok=True)

# iterate through all images in 'photos'
for img_path in incoming.iterdir():
    if not img_path.is_file():
        continue

    name = img_path.name  # e.g. Pedro_Sanchez_ref_01.jpg
    lower = name.lower()

    # detect which person it belongs to
    target_person = None
    for p in people:
        if name.startswith(p):
            target_person = p
            break

    if target_person is None:
        print(f"[SKIP] Can't match person for {name}")
        continue

    # detect if it's reference or test
    if "ref" in lower:
        subfolder = "ref"
    elif "test" in lower:
        subfolder = "test"
    else:
        print(f"[SKIP] Can't tell if ref/test for {name}")
        continue

    dest = dataset / target_person / subfolder / name
    print(f"[MOVE] {name}  →  {dest}")
    shutil.move(str(img_path), str(dest))

print("✅ Done! All images have been sorted into dataset/Spain/...")
