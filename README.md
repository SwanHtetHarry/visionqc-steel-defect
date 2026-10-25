# VisionQC — AI Steel Surface Defect Inspector

VisionQC is a deep-learning computer vision application for automatic steel surface defect classification.

The system uses a fine-tuned **ResNet18** model to classify steel surface images into six defect categories. It also uses **Grad-CAM explainability** to highlight the image regions that most influenced each prediction.

This project covers the complete machine-learning workflow from dataset exploration and CNN development to transfer learning, fine-tuning, evaluation, explainable AI, and deployment with Streamlit.

---

## Live Demo

The public Streamlit application will be added here after deployment.

---

## GitHub Repository

https://github.com/SwanHtetHarry/visionqc-steel-defect

---

## Features

- Deep-learning steel defect classification
- Fine-tuned ResNet18 architecture
- Six supported steel surface defect classes
- Transfer learning using ImageNet pretrained weights
- Custom CNN baseline
- Model comparison
- Data augmentation
- Model checkpointing
- Grad-CAM visual explainability
- Prediction confidence
- Confusion matrix evaluation
- Precision, recall, and F1-score evaluation
- Held-out test set
- Streamlit web application
- Upload-and-analyze interface

---

## Supported Defect Classes

VisionQC classifies steel surface images into six categories:

1. **Crazing** — fine crack-like patterns across the steel surface
2. **Inclusion** — foreign material or impurities embedded in the steel surface
3. **Patches** — irregular regions with noticeable texture or appearance differences
4. **Pitted Surface** — small pits or depressions distributed across the surface
5. **Rolled-in Scale** — oxide scale pressed into steel during the rolling process
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
Highlighted Important Region
```

The model analyzes the uploaded image and produces a predicted defect class.

Grad-CAM is then used to visualize the regions that contributed most strongly to the model's prediction.

---

## Dataset

The project uses the **NEU Surface Defect Database**.

The dataset contains:

```text
Total Images: 1800
Defect Classes: 6
Images per Primary Class: 300
Image Size: 200 × 200
Image Mode: RGB
```

The original dataset also contains XML annotation files describing defect types and bounding-box locations.

During exploratory analysis, some images were found to contain more than one annotated defect type.

For the initial classification task, the primary class represented in the filename was used as the image label.

The annotation data can later support multi-label classification and object detection.

---

## Dataset Exploration

Before training, the dataset was inspected for:

- Image count
- Class distribution
- Image dimensions
- Image color mode
- Missing annotations
- Annotation/image mismatches
- Multiple defect classes within images
- Exact duplicate files between dataset splits

All 1,800 images were confirmed to have:

```text
Image Size: 200 × 200
Image Mode: RGB
```

One misplaced annotation file was identified during exploration and corrected.

An exact file-hash comparison was also performed.

```text
Exact duplicate images across original train/validation folders: 0
```

---

## Final Dataset Split

For the final experiment, all 1,800 images were divided using a fixed stratified split.

```text
Training:   70% = 1260 images
Validation: 15% = 270 images
Test:       15% = 270 images
```

Each defect class contains:

```text
Training:   210 images
Validation: 45 images
Test:       45 images
```

The exact split is stored in:

```text
data/final_splits.csv
```

The model learns only from the training set.

The validation set is used for model selection and checkpointing.

The test set is reserved for final evaluation.

---

## Image Preprocessing

Images are processed using PyTorch and Torchvision.

### Training Pipeline

```text
Original Image
200 × 200
      |
      v
Resize
224 × 224
      |
      v
Random Horizontal Flip
      |
      v
Random Rotation
      |
      v
Convert to Tensor
      |
      v
ImageNet Normalization
```

### Validation and Test Pipeline

```text
Original Image
      |
      v
Resize to 224 × 224
      |
      v
Convert to Tensor
      |
      v
ImageNet Normalization
```

Random augmentation is applied only to training images.

---

## Model Development

The project was developed through three main modeling stages.

### 1. Custom CNN

A convolutional neural network was built from scratch using:

- Three convolutional layers
- ReLU activation
- Max pooling
- Fully connected layers
- Dropout
- Cross-entropy loss
- Adam optimizer

The custom CNN served as the baseline model.

It achieved approximately:

```text
Validation Accuracy: ~90%
```

The main weakness was recognizing the **Inclusion** class.

---

## 2. ResNet18 Transfer Learning

A pretrained ResNet18 model was then used.

Most pretrained layers were initially frozen, while the final classification layer was replaced with a six-class output layer.

```text
Original ResNet18:
512 features
      |
      v
1000 ImageNet classes
```

was changed to:

```text
VisionQC ResNet18:
512 features
      |
      v
6 steel defect classes
```

The frozen ResNet18 achieved approximately:

```text
Validation Accuracy: 93.33%
```

It also improved Inclusion recall compared with the custom CNN.

---

## 3. Fine-Tuned ResNet18

The final ResNet block (`layer4`) was unfrozen and trained with a smaller learning rate.

This allowed the model's higher-level visual features to adapt specifically to steel surface textures.

The final fine-tuned model achieved:

```text
Best Validation Accuracy: 100.00%
```

during the final clean training experiment.

---

## Model Progression

| Model | Evaluation Result |
|---|---:|
| Custom CNN | ~90.00% validation accuracy |
| Frozen ResNet18 | 93.33% validation accuracy |
| Fine-Tuned ResNet18 | 100.00% validation accuracy |
| **Final Fine-Tuned ResNet18** | **100.00% on 270-image held-out test split** |

One important improvement occurred in the Inclusion class.

```text
Custom CNN Inclusion Recall:      58%
Frozen ResNet18 Inclusion Recall: 65%
Fine-Tuned ResNet18:             100% on evaluated final split
```

---

## Final Test Evaluation

The final fine-tuned ResNet18 checkpoint was evaluated on the held-out test set containing 270 images.

```text
Test Images: 270
Correct Predictions: 270
Final Test Accuracy: 100.00%
```

The test set contained:

```text
Crazing:          45
Inclusion:        45
Patches:          45
Pitted Surface:   45
Rolled-in Scale:  45
Scratches:        45
```

### Final Classification Report

```text
Class               Precision    Recall    F1-Score

Crazing                1.0000     1.0000     1.0000
Inclusion              1.0000     1.0000     1.0000
Patches                1.0000     1.0000     1.0000
Pitted Surface         1.0000     1.0000     1.0000
Rolled-in Scale        1.0000     1.0000     1.0000
Scratches              1.0000     1.0000     1.0000

Overall Accuracy                            1.0000
Macro Average           1.0000     1.0000     1.0000
Weighted Average        1.0000     1.0000     1.0000
```

These results represent performance on this specific held-out NEU dataset split.

They should not be interpreted as guaranteed 100% accuracy on real-world industrial images.

---

## Grad-CAM Explainability

VisionQC uses **Gradient-weighted Class Activation Mapping (Grad-CAM)** to explain model predictions.

A normal classifier might only return:

```text
Prediction: Patches
Confidence: 99.95%
```

VisionQC also creates a heatmap showing which image regions most influenced that prediction.

```text
Steel Image
     |
     v
ResNet18
     |
     v
Prediction
     |
     v
Grad-CAM
     |
     v
Highlighted Important Region
```

Warm colors indicate areas that had a stronger influence on the model's decision.

Grad-CAM improves interpretability but does not guarantee that the model's reasoning is correct.

---

## Streamlit Application

The VisionQC application allows the user to:

- Upload a steel surface image
- Run AI defect classification
- View the detected defect
- View prediction confidence
- Read a defect description
- View the original image
- View a Grad-CAM explanation
- Compare the original image and Grad-CAM side by side

Example:

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

The raw image dataset is excluded from GitHub to keep the repository smaller.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/SwanHtetHarry/visionqc-steel-defect
cd visionqc-steel-defect
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Requirements

The main project dependencies are:

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

---

## Run the Application

Run:

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

This project covers:

- Computer vision
- Image classification
- Deep learning
- Convolutional Neural Networks
- Convolution layers
- ReLU activation
- Max pooling
- Data augmentation
- Tensors
- PyTorch Dataset
- PyTorch DataLoader
- Batch training
- Cross-entropy loss
- Backpropagation
- Adam optimization
- Learning rates
- Overfitting
- Model checkpointing
- Transfer learning
- Fine-tuning
- ResNet architecture
- Confusion matrices
- Precision
- Recall
- F1-score
- Held-out test evaluation
- Explainable AI
- Grad-CAM
- Streamlit deployment

---

## Limitations

Although the model achieved excellent performance on the NEU dataset, several limitations remain:

- The dataset is relatively small.
- The images come from a controlled dataset.
- Real industrial environments may contain different lighting conditions.
- Different cameras may produce different image characteristics.
- Steel types and textures may differ from the training dataset.
- Some images contain multiple annotated defect types.
- The current classifier predicts one primary class per image.
- Visually similar images may exist even when exact duplicate files do not.
- Prediction confidence is not a guarantee of correctness.
- Grad-CAM shows influential regions but does not prove causal reasoning.
- The model has not yet been validated using a completely independent industrial dataset.
- The current model classifies defects but does not precisely localize them with bounding boxes.

Therefore, the reported 100% held-out test accuracy should not be interpreted as guaranteed real-world industrial performance.

---

## Future Improvements

Possible future improvements include:

- Multi-label defect classification
- YOLO object detection
- Defect bounding boxes
- Automatic defect localization
- External dataset validation
- Additional steel defect datasets
- EfficientNet comparison
- Additional pretrained architectures
- Hyperparameter optimization
- Advanced data augmentation
- Defect severity estimation
- Batch inspection
- Inspection history
- Industrial analytics dashboard
- Real-time camera inspection
- Real-time production-line integration
- Model monitoring

A future object-detection version could produce:

```text
Steel Image
        |
        v

+-----------------------------+
|                             |
|   [Crazing 97%]             |
|                             |
|                [Patch 91%]  |
|                             |
+-----------------------------+
```

---

## Disclaimer

VisionQC is intended for **educational and research purposes**.

The application should not be used as the sole method for industrial quality-control decisions.

Deep-learning models can produce incorrect predictions, especially when analyzing images that differ from the training dataset.

Professional inspection procedures and appropriate quality-control systems should be used in real manufacturing environments.

---

## Author

**Swan Htet (Harry Li)**

B.Sc. Information Technology Student  
AI / Machine Learning Project

GitHub: https://github.com/SwanHtetHarry