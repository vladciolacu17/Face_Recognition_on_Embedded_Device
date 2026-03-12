
Face Recognition on Embedded Device
===================================

This project implements a face recognition pipeline designed to run on an embedded system such as a Raspberry Pi 4.

The system detects faces, extracts embeddings using the FaceNet model, and identifies the most similar identity using cosine similarity.

The goal of the project is to evaluate whether modern face recognition algorithms can operate efficiently on resource‑constrained hardware.

------------------------------------------------------------
Project Overview
------------------------------------------------------------

The recognition pipeline performs the following steps:

1. Detect and align faces in an image
2. Extract facial embeddings using a pretrained FaceNet model
3. Compare embeddings with reference images
4. Rank identities using cosine similarity
5. Output recognition results

Two versions of the pipeline are implemented:

• MTCNN version  
  Uses MTCNN for face detection and alignment before embedding extraction.

• No‑MTCNN version  
  Skips detection and processes the images directly.

This allows comparison of recognition accuracy and performance.

------------------------------------------------------------
Technologies Used
------------------------------------------------------------

Python
PyTorch
facenet‑pytorch
MTCNN
NumPy
Pandas
Pillow
Raspberry Pi 4

------------------------------------------------------------
Repository Structure
------------------------------------------------------------
```
src/
│
├── dataset/
│   ├── Fernando_Alonso/
│   │   ├── ref/
│   │   └── test/
│   │
│   ├── Pedro_Sanchez/
│   ├── Penelope_Cruz/
│   ├── Rafael_Nadal/
│   └── Rosalia/
│
├── face_rec_spain2.py
├── face_rec_spain_no_mtcnn.py
├── sorter.py
│
├── results_spain.csv
└── results_spain_no_mtcnn.csv
README.md
LICENSE
.gitignore
requirements.txt
```
------------------------------------------------------------
Dataset Structure
------------------------------------------------------------

Each identity contains two folders:

ref/
    Reference images used to build the embedding database.

test/
    Images used for evaluation and recognition testing.

Example:
```
dataset/
└── Fernando_Alonso/
    ├── ref/
    └── test/
```
------------------------------------------------------------
Installation
------------------------------------------------------------

Clone the repository:
```bash

git clone https://github.com/vladciolacu17/Face_Recognition_on_Embedded_Device.git
```
Navigate to the project directory:
```bash
cd Face_Recognition_on_Embedded_Device
```
Install required dependencies:
```bash
pip install -r requirements.txt
```
------------------------------------------------------------
Running the Project
------------------------------------------------------------

Run the pipeline with face detection:
```bash
python src/face_rec_spain2.py
```
Run the pipeline without face detection:
```bash
python src/face_rec_spain_no_mtcnn.py
```
------------------------------------------------------------
Results
------------------------------------------------------------

Recognition results are saved as CSV files:

- results_spain.csv
- results_spain_no_mtcnn.csv

These files contain the predicted identity and similarity scores for each test image.

------------------------------------------------------------
Future Improvements
------------------------------------------------------------

Possible improvements include:

- real‑time webcam recognition
- support for larger datasets
- threshold for unknown identities
- model optimization for faster inference
- deployment in access control systems

------------------------------------------------------------
Author
------------------------------------------------------------

Vlad‑Stefan Ciolacu

GitHub:
https://github.com/vladciolacu17

------------------------------------------------------------
License
------------------------------------------------------------

MIT License
