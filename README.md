# Hand Gesture Classification — HaGRID

This project classifies 18 hand gestures using 3D landmark coordinates (x, y, z) extracted from the [HaGRID dataset](https://github.com/hukenovs/hagrid) via MediaPipe. Multiple classifiers were trained, tracked with MLflow, and evaluated to select the best model for production.

## Project Structure

```
├── hand_gesture_classification.ipynb   # Main notebook: EDA, training, evaluation
├── hand_landmarks_data.csv             # Processed landmark features + labels
├── mlflow_utils.py                     # MLflow logging helpers
├── mlruns/                             # MLflow experiment tracking data
├── models/                             # Saved model artifacts
└── Mlflow_ScreenShots/                 # MLflow UI screenshots
```

## Dataset

- **Source**: HaGRID (Hand Gesture Recognition Image Dataset)
- **Features**: 21 MediaPipe landmarks x 3 coordinates (x, y, z) = 63 features per sample
- **Classes**: 18 gesture types 

![Class Distribution](class_distribution.png)

![Hand Landmarks per Class](hand_landmarks_per_class.png)

## Model Comparison

| Model | Accuracy | F1-Score | Precision | Recall | Train Time (s) |
|-------|----------|----------|-----------|--------|----------------|
| **MLP-256-128-64** | **98.75%** | **0.99** | **0.99** | **0.99** | 9.02 |
| Random-Forest-200trees | 97.88% | 0.98 | 0.98 | 0.98 | 40.3 |
| SVM-RBF-C10 | 96.98% | 0.97 | 0.97 | 0.97 | 31.5 |
| KNN-k7-distance | 93.69% | 0.94 | 0.94 | 0.94 | 22.1 |

![Confusion Matrices](confusion_matrices.png)

## Why MLP?

The MLP-256-128-64 was selected as the production model. It achieved the highest accuracy (98.75%) and showed the cleanest confusion matrix across all 18 classes — including visually similar pairs like `peace`/`peace_inverted` and `two_up`/`two_up_inverted` where other models struggled.

Beyond raw accuracy, an MLP is well-suited to this problem because hand landmarks have complex geometric interdependencies that benefit from non-linear learning. Random Forest was a strong runner-up but slower to train and predict with 200 trees. SVM offered no meaningful accuracy advantage for its training cost. KNN showed the most inter-class confusion and does not scale well to real-time inference.

The best model was registered in the MLflow Model Registry as `Best-Hand-Gesture-Classifier-MLP` with a `production` alias.

## MLflow Tracking

![MLflow Runs](Runs.png)

![Registered Model](RegisteredModel.png)
