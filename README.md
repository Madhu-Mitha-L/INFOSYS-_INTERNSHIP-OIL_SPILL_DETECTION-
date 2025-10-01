# 🛢️ Oil Spill Detection Project

## 📌 Overview
Detect and segment oil spill regions in satellite or SAR images using a lightweight deep learning model based on **U-Net**.  
The project includes dataset preprocessing, model training, evaluation, and visualization of results.

---

## 📑 Table of Contents
- [📌 Overview](#-overview)
- [✨ Key Features](#-key-features)
- [✨ Dataset](#-dataset)
- [🏗️ Model Architecture](#-model-architecture)
- [💻 Installation](#-installation)
- [🛠️ Required Libraries](#-required-libraries)
- [🚀 Google Colab Setup](#-google-colab-setup)
- [🎯 Training Configuration](#-training-configuration)
- [🧪 Making Predictions](#-making-predictions)
- [📊 Results & Metrics](#-results--metrics)
- [📈 Visualization](#-visualization)
- [📉 Model Performance Summary](#-model-performance-summary)
- [📊 Output](#-output)
- [⚡ Limitations & Optimization](#-limitations--optimization)
- [🔮 Future Improvements](#-future-improvements)
- [📚 References & Libraries](#-references--libraries)

---

## ✨ Key Features
- ✅ Segmentation of oil spills in satellite images.  
- ✅ Handles single-channel SAR and multi-channel RGB images.  
- ✅ Lightweight **U-Net** architecture for faster inference.  
- ✅ Metrics: Accuracy, IoU, Dice, Precision, Recall.  
- ✅ Visualizations: Heatmaps, overlays, histograms, pie charts.

---

## ✨ Dataset
- 📦 **Train**: Images and corresponding masks for training.  
- 📦 **Validation**: Images and masks for validation.  
- 🌐 Supports single-channel SAR or multi-channel satellite imagery.  

**Directory Structure**
```bash
dataset/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
```
## 📌 Model Architecture
- ⚙️ **Architecture**: Based on **U-Net** for segmentation tasks.  
- 🔄 **Structure**: Encoder-decoder with skip connections for preserving spatial information.  
- 🖼️ **Input**: Supports single-channel SAR or multi-channel RGB satellite images.  

**Example U-Net block in Python (Keras/TensorFlow):**
```python
from tensorflow.keras import layers, models

def unet_block(input_size=(128, 128, 1)):
    inputs = layers.Input(input_size)
    
    # Encoder
    c1 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2,2))(c1)
    
    # Bottleneck
    c2 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(c2)
    
    # Decoder
    u1 = layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c2)
    u1 = layers.concatenate([u1, c1])
    c3 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(u1)
    c3 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(c3)
    
    outputs = layers.Conv2D(1, (1,1), activation='sigmoid')(c3)
    
    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model
```
##  ARCHITECTURE OUTPUT

- 🧪 Output: Single-channel segmentation mask (oil spill regions).
- 🏎️ Lightweight: Designed for faster inference on moderate hardware (GPU/TPU).

 ## 💻 Installation
 ```bash
 # Clone repository
git clone <repo-url>

# Navigate to project
cd Oil_Spill_Detection

# Install dependencies
pip install -r requirements.txt
```
## 🔮 Required Libraries

- Python 3.12
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- OpenCV
- scikit-learn
  
## 🚀 Google Colab Setup
```
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Change working directory
%cd /content/drive/MyDrive/Oil_Spill_Detection
```
## 🎯 Training Configuration

- Epochs: 50
- Batch Size: 16
- Input Size: 128x128 (or 256x256 depending on dataset)
- Optimizer: Adam (learning_rate=0.001)
- Loss: Binary Cross-Entropy + Dice Loss
```
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', dice_coef_improved, iou_improved])
```
## 🧪 Making Predictions
``` img_batch = tf.expand_dims(sample_img, 0)
pred = model.predict(img_batch)[0]
pred_binary = (pred > 0.5).astype(np.float32)
```
## 📊 Results & Metrics

-Validation Metrics Example
```
BCE Loss   : 0.2329
Dice Loss  : 0.0326
Accuracy   : 0.9365
Dice Coef  : 0.9674
IoU        : 0.9385
Precision  : 0.9394
Recall     : 0.9961

```

## 📈 Visualization

### 🔹 Side-by-Side Comparison
- **Original Image**  
- **Ground Truth Mask** (Gray)  
- **Prediction Heatmap** (Hot)  
- **Prediction Binary Mask**  

### 🔹 Overlay Visualization
- 🔴 **Red**: Ground Truth  
- 🟢 **Green**: Prediction  
- 🔵 **Blue**: Correct Overlap (True Positive)  
- 🔴 **Red**: Missed (False Negative)  
- 🟢 **Green**: False Positive  

### 🔹 Metrics Visuals
- **Bar Charts**: IoU, Dice, Precision, Recall  
- **Scatter Plots**: IoU vs Dice per image  
- **Histogram**: Prediction probability distribution  
- **Pie Chart**: Pixel distribution (Oil Spill vs Background)

## 📉 Model Performance Summary
- High **Dice Coefficient** (~0.9674) and **IoU** (~0.9385)  
- **Precision** & **Recall** above 0.93  
- Lightweight model suitable for moderate GPU/TPU inference

## 📊 Output
- visualization of the metrices
  <img width="1756" height="490" alt="image" src="https://github.com/user-attachments/assets/43c430d0-e5db-4884-adbe-0fad91c0c70c" />
  <img width="1748" height="493" alt="image" src="https://github.com/user-attachments/assets/28c0eed6-0baf-4128-9e1f-e4f70fddb587" />
- overlay outputs
  <img width="1778" height="437" alt="image" src="https://github.com/user-attachments/assets/2d3a8dab-6841-4e7e-ad9b-dae504067a18" />
- prediction outputs
  <img width="1482" height="762" alt="image" src="https://github.com/user-attachments/assets/45581c9f-4c40-47ef-8884-d6c67fb50153" />
- plots outputs
  <img width="773" height="562" alt="image" src="https://github.com/user-attachments/assets/b9d2d9b9-8f07-4ff3-a03f-2690c798f7b4" />
  <img width="792" height="581" alt="image" src="https://github.com/user-attachments/assets/60324d88-eba2-4c3f-9cb8-7b3e9d0d7039" />

## ⚡ Limitations & Optimization
- Limited dataset may affect generalization  
- Image resolution affects performance  
- **Optimizations**: Data augmentation, mixed precision training, TPU acceleration  

## 🔮 Future Improvements
- Advanced architectures (Attention U-Net, DeepLabv3)  
- Multi-class oil spill segmentation  
- Integrate temporal satellite data for change detection  
- Deploy as REST API for real-time monitoring  

## 📚 References & Libraries
- [TensorFlow Documentation](https://www.tensorflow.org/)  
- [Keras API Guide](https://keras.io/)  
- [OpenCV Documentation](https://opencv.org/)  
- [scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)  
















