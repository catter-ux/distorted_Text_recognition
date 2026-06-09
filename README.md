
# Distorted Text Recognition using Deep Learning (CRNN + CTC)

## Project Overview
This project focuses on recognizing distorted text from grayscale images. The notebook was created as part of an AI/ML competition where the goal is to correctly predict text hidden inside noisy and distorted images.

The solution uses a CRNN (Convolutional Recurrent Neural Network) architecture consisting of:

- CNN for feature extraction
- Bidirectional LSTM for sequence learning
- CTC (Connectionist Temporal Classification) Loss for sequence prediction
- Greedy and Beam Search decoding for text generation

The evaluation metric used is Character Error Rate (CER), where lower values indicate better performance.

---

## Notebook Structure

### 1. Imports and Setup
- Imports required Python libraries.
- Sets random seeds for reproducibility.
- Automatically selects GPU (CUDA) if available.

### 2. Configuration
Defines:
- Dataset paths
- Image dimensions (32 × 128)
- Training hyperparameters
- Character vocabulary (A–Z and 0–9)

### 3. Vocabulary and Encoding
Creates helper functions to:
- Convert text labels into numerical indices.
- Convert model predictions back into text.
- Perform CTC decoding.

### 4. Dataset and Data Loading
Implements:
- CaptchaDataset for training and validation.
- TestDataset for inference.
- Data loading utilities for batch processing.

### 5. Image Preprocessing and Augmentation
Applies:
- Resize operations
- Normalization
- Blur augmentation
- Noise augmentation
- Small image distortions

These augmentations improve model robustness.

### 6. DataLoaders
Creates:
- Training loader
- Validation loader
- Test loader

Also includes a custom CTC collate function.

### 7. Exploratory Data Analysis (EDA)
Performs:
- Label length distribution analysis
- Character frequency analysis
- Sample image visualization

This helps understand dataset characteristics before training.

### 8. CRNN Model Architecture
The model contains:

CNN Backbone
→ Extracts visual features from images

Bidirectional LSTM
→ Learns sequential dependencies between characters

Linear Classification Layer
→ Predicts character probabilities at each time step

### 9. CTC Loss
Uses Connectionist Temporal Classification:
- No character-level alignment required.
- Handles varying sequence lengths.
- Suitable for OCR tasks.

### 10. Character Error Rate (CER)
Implements Levenshtein Distance to calculate:
- Insertions
- Deletions
- Substitutions

CER is used as the primary evaluation metric.

### 11. Training Loop
Includes:
- Forward pass
- Backpropagation
- Validation
- Early stopping
- Checkpoint saving
- Gradient clipping

### 12. Training Visualization
Plots:
- Training vs Validation Loss
- Validation CER

These graphs help evaluate model learning behavior.

### 13. Model Evaluation
Loads the best checkpoint and:
- Evaluates on validation data
- Displays predictions
- Calculates final CER

### 14. Test Inference
Runs prediction on unseen test images and generates:
submission.csv

### 15. Submission Export
Renames the submission file using:
- Student name
- Enrollment number

### 16. Beam Search Decoder
Adds an optional Beam Search decoder which can improve prediction quality compared to simple greedy decoding.

### 17. Final Conclusions
Summarizes:
- Model architecture
- Design decisions
- Performance considerations
- Possible future improvements

---

## What I Learned

As a beginner project, this notebook helped me understand:

- OCR (Optical Character Recognition) pipelines
- Deep learning for sequence prediction
- CNN feature extraction
- LSTM sequence modeling
- CTC Loss
- Character Error Rate evaluation
- Data augmentation techniques
- Model training and validation workflows

---

## Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- PIL (Python Imaging Library)

---

## Future Improvements

- Larger and deeper CNN backbone
- Transformer-based sequence modeling
- Better augmentation strategies
- Language model integration
- More advanced beam search decoding

---

## Conclusion

This notebook presents a complete end-to-end OCR pipeline for distorted text recognition. It demonstrates the full workflow from data preparation and exploratory analysis to model training, evaluation, and submission generation. The project is well-structured, beginner-friendly, and provides a strong introduction to deep learning-based OCR systems.
