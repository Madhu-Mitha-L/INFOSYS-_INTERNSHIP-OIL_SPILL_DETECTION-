# 🛢️ Oil Spill Detection Project

## 📌 Overview
Detect and segment oil spill regions in satellite or SAR images using a lightweight deep learning model based on **U-Net**.  
The project includes dataset preprocessing, model training, evaluation, and visualization of results.

---

## 📑 Table of Contents
- [📌 Overview](#-overview)
- [✨ Key Features](#-key-features)
- [🗄️ Dataset](#-dataset)
- [🏗️ Model Architecture](#-model-architecture)
- [💻 Installation](#-installation)
- [🛠️ Required Libraries](#-required-libraries)
- [🚀 Google Colab Setup](#-google-colab-setup)
- [🎯 Training Configuration](#-training-configuration)
- [🧪 Making Predictions](#-making-predictions)
- [📊 Results & Metrics](#-results--metrics)
- [📈 Visualization](#-visualization)
- [📉 Model Performance Summary](#-model-performance-summary)
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

## 🗄️ Dataset
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

