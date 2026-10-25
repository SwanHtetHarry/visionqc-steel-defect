# VisionQC — AI Steel Surface Defect Inspector

VisionQC is a deep-learning computer vision project for automatic steel surface defect classification.

The system uses a fine-tuned **ResNet18** model to classify steel surface images into six defect categories and uses **Grad-CAM explainability** to highlight the image regions that most influenced the model's prediction.

The project covers the complete machine-learning workflow from dataset exploration and CNN development to transfer learning, model evaluation, explainability, and Streamlit deployment.

---

## Features

- Steel surface defect classification using deep learning
- Fine-tuned ResNet18 model
- Six supported defect classes
- Transfer learning from ImageNet
- Custom CNN baseline comparison
- Grad-CAM visual explainability
- Prediction confidence
- Streamlit web application
- Held-out test-set evaluation
- Balanced train, validation, and test splits
- Model checkpointing
- Confusion matrix and classification report evaluation

---

## Supported Defect Classes

VisionQC classifies steel surface images into six categories:

1. **Crazing** — fine crack-like patterns across the steel surface
2. **Inclusion** — foreign material or impurities embedded in the surface
3. **Patches** — irregular surface regions with noticeable texture differences
4. **Pitted Surface** — small pits or depressions across the surface
5. **Rolled-in Scale** — oxide scale pressed into the steel during rolling
6. **Scratches** — linear surface damage caused by mechanical contact or abrasion

---

## How It Works

```text
Steel Surface Image
        |
        v
Image Preprocessing
        |
        v
Fine-Tuned ResNet18
        |
        v
Defect Classification
        |
        +----------------------+
        |                      |
        v                      v
Predicted Defect         Prediction Confidence
        |
        v
Grad-CAM Explainability
        |
        v
Highlighted Defect Region
```

---

## Dataset

The project uses the **NEU Surface Defect Database**, containing 1,800 steel surface images across six primary defect categories.

Each primary class contains 300 images.

The project originally contained:

```text
1800 total images
6 defect classes
300 images per primary class
```

During dataset exploration, the XML annotations also revealed that some images contain multiple annotated defect regions.

This project therefore uses the filename-defined primary defect class for the initial six-class image classification task.

The annotation information can later be reused for multi-label classification and object detection.

---

## Dataset Exploration

Before model training, the dataset was inspected for:

- Image count
- Class balance
- Image dimensions
- Image color mode
- Annotation availability
- Image/annotation mismatches
- Multiple defects within individual images
- Duplicate images across dataset splits

All 1,800 images were confirmed to be:

```text
Size: 200 × 200 pixels
Mode: RGB
```

The project also identified and corrected one misplaced annotation file during dataset inspection.

Exact file hashing was later used to check for duplicate images across the original training and validation folders.

```text
Exact duplicate images found: 0
```

---

## Final Dataset Split

For the final experiment, the full dataset was divided using a fixed stratified split:

```text
Training:   70% = 1260 images
Validation: 15% = 270 images
Test:       15% = 270 images
```

Each class contains:

```text
Training:   210 images
Validation:  45 images
Test:        45 images
```

The split is stored in:

```text
data/final_splits.csv
```

The test set was kept separate from model training and checkpoint selection during the final clean experiment.

---

## Image Preprocessing

Images are prepared using PyTorch and Torchvision.

The preprocessing pipeline includes:

```text
Original Image
200 × 200
    |
    v
Resize
224 × 224
    |
    v
Training Augmentation
    |
    +-- Random Horizontal Flip
    +-- Random Rotation
    |
    v
Convert to Tensor
    |
    v
ImageNet Normalization
```

Validation and test images are not randomly augmented.

ImageNet normalization values are used because the final model uses pretrained ResNet18 weights.

---

## Model Development

Three main model stages were explored.

### 1. Custom CNN

A convolutional neural network was built from scratch using:

- Three convolutional layers
- ReLU activation
- Max pooling
- Fully connected layers
- Dropout
- Cross-entropy loss
- Adam optimizer

The custom CNN served as the project baseline.

It achieved approximately:

```text
Validation Accuracy: ~90%
```

The model performed strongly on several classes but had difficulty recognizing the **Inclusion** class.

---

### 2. ResNet18 Transfer Learning

A pretrained ResNet18 model was then used.

The pretrained convolutional layers were frozen, and only the final classification layer was trained for the six steel defect classes.

This improved performance to approximately:

```text
Validation Accuracy: 93.33%
```

Transfer learning also improved Inclusion recall compared with the custom CNN.

---

### 3. Fine-Tuned ResNet18

The final ResNet block (`layer4`) was unfrozen and trained using a smaller learning rate.

This allowed the pretrained visual features to adapt specifically to steel surface textures.

During the clean final experiment, the best fine-tuned checkpoint achieved:

```text
Validation Accuracy: 100.00%
```

The final model was selected using validation performance before evaluation on the held-out test split.

---

## Final Test Results

The final fine-tuned ResNet18 was evaluated on the held-out test split containing 270 images.

### Test Accuracy

```text
Final Test Accuracy: 100.00%
```

The test set contained:

```text
45 Crazing images
45 Inclusion images
45 Patches images
45 Pitted Surface images
45 Rolled-in Scale images
45 Scratches images
```

The model correctly classified all 270 images in this particular held-out split.

### Classification Report

```text
Class              Precision    Recall    F1-Score

Crazing               1.00       1.00       1.00
Inclusion             1.00       1.00       1.00
Patches               1.00       1.00       1.00
Pitted Surface        1.00       1.00       1.00
Rolled-in Scale       1.00       1.00       1.00
Scratches             1.00       1.00       1.00

Overall Accuracy                           1.00
```

These results represent performance on this specific NEU dataset split and should not be interpreted as guaranteed 100% performance on real-world industrial images.

---

## Model Progression

| Model | Evaluation Result |
|---|---:|
| Custom CNN | ~90.00% validation accuracy |
| Frozen ResNet18 | 93.33% validation accuracy |
| Fine-Tuned ResNet18 | 100.00% validation accuracy |
| **Final Fine-Tuned ResNet18** | **100.00% on 270-image held-out test split** |

One particularly important improvement occurred for the Inclusion class.

```text
Custom CNN Inclusion Recall:      58%
Frozen ResNet18 Inclusion Recall: 65%
Fine-Tuned ResNet18:             100% on evaluated split
```

---

## Grad-CAM Explainability

VisionQC uses **Gradient-weighted Class Activation Mapping (Grad-CAM)** to explain model predictions.

Instead of displaying only:

```text
Prediction: Patches
Confidence: 99.95%
```

the application also generates a heatmap showing which areas of the image had the strongest influence on the prediction.

```text
Steel Image
     |
     v
ResNet18 Prediction
     |
     v
Grad-CAM
     |
     v
Highlighted Important Region
```

Warm-colored regions in the visualization represent areas that contributed more strongly to the model's decision.

Grad-CAM does not prove that a model's reasoning is correct, but it provides useful insight into where the network focused when making a prediction.

---

## Streamlit Application

The Streamlit application allows users to:

- Upload a steel surface image
- Run AI defect classification
- View the predicted defect
- View prediction confidence
- Read a short defect description
- View the original image
- View the Grad-CAM explanation
- Compare the original image and AI attention side by side

Example output:

```text
Inspection Result

Detected Defect:
Crazing

Prediction Confidence:
99.62%

Visual Explanation:
Original Image | Grad-CAM Heatmap
```

---

## Project Structure

```text
visionqc-steel-defect/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── final_splits.csv
│
├── models/
│   └── final_resnet18_finetuned_best.pth
│
├── notebooks/
│   └── 01_dataset_exploration.ipynb
│
└── src/
```

The raw training and validation images are excluded from GitHub through `.gitignore`.

---

## Requirements

Main dependencies:

```text
streamlit
torch
torchvision
pillow
numpy
matplotlib
scikit-learn
pandas
```

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## Run the Application

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd visionqc-steel-defect
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

in your browser.

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- ResNet18
- Convolutional Neural Networks
- Transfer Learning
- Fine-Tuning
- Grad-CAM
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Pillow
- Streamlit
- Jupyter Notebook
- Git
- GitHub

---

## Key Concepts Learned

This project covers several important deep-learning concepts:

- Image classification
- Convolutional Neural Networks
- Convolution layers
- ReLU activation
- Max pooling
- Data augmentation
- Tensors
- DataLoaders
- Batch training
- Cross-entropy loss
- Backpropagation
- Adam optimization
- Learning rates
- Overfitting
- Model checkpointing
- Transfer learning
- Fine-tuning
- Confusion matrices
- Precision
- Recall
- F1-score
- Explainable AI
- Grad-CAM
- Held-out test evaluation
- Model deployment

---

## Limitations

Although the model achieved excellent results on the held-out NEU dataset split, several limitations remain:

- The NEU dataset is relatively small.
- Images come from a controlled dataset rather than live industrial production.
- Real manufacturing environments may contain lighting, camera, material, and surface variations not represented in the dataset.
- Some dataset images contain multiple annotated defects while the current classifier predicts only one primary class.
- Visually similar images may exist even when exact duplicate files do not.
- Model confidence is not a guarantee that a prediction is correct.
- Grad-CAM shows influential regions but does not prove causal reasoning.
- The current system performs classification rather than defect localization.
- The model has not yet been validated against an independent external industrial dataset.

Because of these limitations, the reported 100% held-out test accuracy should not be interpreted as guaranteed real-world industrial performance.

---

## Future Improvements

Planned improvements include:

- Multi-label defect classification
- YOLO-based object detection
- Defect bounding boxes
- Defect localization
- External dataset validation
- Additional industrial datasets
- More advanced augmentation
- EfficientNet comparison
- Additional transfer-learning architectures
- Hyperparameter tuning
- Batch image