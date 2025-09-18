# 😷 Face-Mask Detector
Real-time **face-mask detection** 
---

## 🖼️ Working Demo
### With Mask
![with mask](samples/with_mask.jpg)

### Without Mask
![without mask](samples/without_mask.jpg)

---

## 📊 Dataset
- Dataset contains **1314 training images** and **194 test images**.  
- Two categories:  
  - **with_mask/**  
  - **without_mask/**  
- You can download dataset from:  
  - Kaggle face-mask dataset, or  
  - This repo’s `data/` folder (`train/` and `test/`).  

---

## ⚙️ Setup Instructions

### 1️⃣ Clone Repository
```bash
git clone https://github.com/ChauhanVi/face-mask-detector.git
cd face-mask-detector
2️⃣ Create Virtual Environment (Recommended)

python -m venv .venv
# Activate venv
.\.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # Linux/Mac
3️⃣ Install Dependencies:
pip install -r requirements.txt

If no requirements.txt, manually install:
pip install opencv-python opencv-contrib-python scikit-learn tensorflow keras joblib imutils numpy

4️⃣ Download Pre-trained Models
Run:
python src/download_models.py

▶️ How to Use
🖼️ Run on an Image:
python src/detect_mask_image.py --image assets/samples.jpg

🎥 Run on Webcam:
python src/detect_mask_video.py --source 0

📹 Run on a Video File:
python src/detect_mask_video.py --source assets/video.mp4#

