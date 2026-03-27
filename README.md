# PneumoAI – Intelligent Pneumonia Diagnostic System

PulmoAI is an **AI-powered medical diagnostic platform** that detects pneumonia from chest X-ray images using deep learning. The system combines a trained CNN model with a Flask-based web application to provide **real-time predictions, patient data integration, and history tracking**, simulating a clinical workflow.

---

## Features

*  **Pneumonia Detection** from chest X-ray images
*  **Transfer Learning (MobileNetV2)** for high-performance classification
*  **Flask Web App** for real-time predictions
*  **Patient Data Input** (age, gender, medical history)
*  **Prediction Dashboard** with confidence score & recommendations
*  **History Tracking** using SQLite database
*  **User Authentication** (login/signup system)

---

##  Model Details

* Architecture: MobileNetV2 (Transfer Learning)
* Framework: TensorFlow / Keras
* Image Size: 224 × 224
* Dataset: Chest X-ray Pneumonia Dataset (Kaggle)
* Classes: Normal vs Pneumonia

---

## Performance

* Training Accuracy: ~96%
* Validation Accuracy: ~95%
* Test Accuracy: ~86%

**Key Insight:**
The model achieves **very low false negatives**, ensuring critical pneumonia cases are rarely missed — an important factor in medical applications.

---

## System Architecture

```text
User → Web Interface → Flask Backend → ML Model → Prediction → Database
```

---

## Tech Stack

* Python
* TensorFlow / Keras
* Flask
* SQLite
* NumPy, Matplotlib, Seaborn
* HTML, CSS

---

## How to Run

```bash
# Clone repository
git clone https://github.com/your-username/pneumoai.git
cd pneumoai

# Install dependencies
pip install flask tensorflow pillow numpy matplotlib seaborn

# Create database
python create_db.py

# Run app
python app.py
```

Open in browser:
`http://127.0.0.1:5000`

---

## Limitations

* Moderate false positives
* Dataset imbalance
* Not a replacement for clinical diagnosis

---

## Future Improvements

* Grad-CAM for explainability
* Multi-disease detection
* Cloud deployment
* Mobile app integration

---

## Tags

`AI` `Deep Learning` `Healthcare` `Medical Imaging` `Flask` `TensorFlow` `Computer Vision`

---

## Final Note

PulmoAI demonstrates how **AI + full-stack development** can be combined to build a practical healthcare solution, bridging the gap between machine learning models and real-world applications.

---

