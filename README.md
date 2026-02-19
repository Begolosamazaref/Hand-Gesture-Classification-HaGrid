# Hand Gesture Classification — HaGrid

A machine learning project that classifies hand gestures using hand landmark data extracted from the [HaGrid dataset](https://github.com/hukenovs/hagrid).

---

## Project Structure

```
ML_Project/
├── Notebook/
│   ├── hand_gesture_classification.ipynb   # Main notebook 
│   └── hand_landmarks_data.csv             # Extracted hand landmark features dataset
├── Screenshots/
│   ├── class_distribution.png              # Distribution of gesture classes
│   ├── confusion_matrices.png              # Confusion matrices for all models
│   ├── hand_landmarks_per_class.png        # Landmark visualizations per class
│   └── normalization_comparison.png        # Effect of feature normalization
├── Videos/
│   ├── Input - Made with Clipchamp.mp4     # input video
│   └── output.mp4                          # Model prediction output video
└── pkl_files/
    ├── MLP.pkl                             # Best trained MLP model
    ├── label_encoder.pkl                   # Label encoder for gesture classes
    └── scaler.pkl                          # StandardScaler for feature normalization
```

---

## Workflow Overview

```
MediaPipe Hand Landmark Extraction (21 landmarks × 3 coords = 63 features)
       ↓
Feature Engineering & Normalization (StandardScaler)
       ↓
Train/Test Split
       ↓
Multiple Model Training (MLP, SVM, RF, KNN)
       ↓
Best Model → pkl_files/MLP.pkl
       ↓
Inference on Video
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Begolosamazaref/Hand-Gesture-Classification-HaGrid.git
cd Hand-Gesture-Classification-HaGrid/ML_Project
```

### 2. Create and activate the environment

```bash
conda create -n hand_gesture python=3.10
conda activate hand_gesture
```

### 3. Install dependencies

```bash
pip install scikit-learn mediapipe pandas numpy matplotlib opencv-python jupyter
```

### 4. Run the notebook

```bash
jupyter lab Notebook/hand_gesture_classification.ipynb
```

## Branches

| Branch | Contents |
|--------|----------|
| `main` | Clean project files: notebook, dataset, models, screenshots, videos |
| `research` | Full MLflow experiment data: runs, model artifacts, registry, screenshots |

---

## Tech Stack

- **MediaPipe** — hand landmark extraction
- **scikit-learn** — model training and evaluation
- **OpenCV** — video processing
- **Pandas / NumPy** — data handling
- **Matplotlib** — visualization

---

## Author

**Begolosamazaref**  
ITI Diploma  
[GitHub Profile](https://github.com/Begolosamazaref)
