# 🧠 Brain Tumor Detection Using Deep Learning (MRI Images)

This project focuses on **Brain Tumor Detection** using **Computer Vision** and **Deep Learning**.  
We have used the **VGG16 architecture** with **Transfer Learning** for classification.

---

## 🚀 Model Details

- **Base Model:** VGG16 (Pre-trained on ImageNet, 1.4M images)  
- **Input Shape:** (128, 128, 3)  
- **Transfer Learning:**  
  - All layers frozen initially (`trainable = False`)  
  - Last 3 layers unfrozen for fine-tuning (`trainable = True`)  

### 🏗️ Architecture
1. Load VGG16 with `include_top=False`, `weights='imagenet'`.  
2. Freeze all convolutional layers.  
3. Unfreeze the last 3 layers for fine-tuning.  
4. Add **Flatten layer** to convert 3D tensor → 1D tensor.  
5. Add **Dropout(0.3)** to reduce overfitting.  
6. Add **Dense(128, activation='relu')** layer.  
7. Add **Dropout(0.2)**.  
8. Add **Output Dense Layer** with `softmax` activation for multi-class classification.  

---

## 📊 Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
|   0   |   0.97    |  0.98  |   0.98   |   300   |
|   1   |   0.93    |  0.90  |   0.91   |   300   |
|   2   |   0.95    |  1.00  |   0.97   |   405   |
|   3   |   0.93    |  0.91  |   0.92   |   306   |

**Overall Performance**  
- Accuracy: **95%**  
- Macro Avg: Precision **0.95**, Recall **0.94**, F1-Score **0.95**  
- Weighted Avg: Precision **0.95**, Recall **0.95**, F1-Score **0.95**  

---

## 🛠️ Tech Stack

- Python  
- TensorFlow / Keras  
- NumPy, Pandas, Matplotlib  
- Computer Vision Techniques  

---

## 📂 Project Structure
