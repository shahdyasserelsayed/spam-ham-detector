# 📧 Spam-Ham Detector

A simple **Spam Detection Web App** built with **Streamlit** and **Machine Learning**.  
It classifies text messages as **Spam** or **Ham (Not Spam)** using a pre-trained pipeline model.

<img width="917" height="589" alt="image" src="https://github.com/user-attachments/assets/17d15c9a-2d48-4075-8003-9319a4583927" />


---

##  Features
- Clean Streamlit interface for quick text classification  
- Pre-trained ML model using NLP techniques  
- Instant predictions  
- Modular code with training, testing, and utility scripts  

---

##  Model Overview
The model uses text preprocessing (tokenization, lowercasing, punctuation removal) and a **machine learning classifier** trained on labeled spam datasets.  
It’s saved as `spam_pipeline.joblib` for efficient loading and prediction.

---

##  Project Structure
```
Spam_ham_detection/
│
├── app.py                # Streamlit web app
├── train.py              # Model training script
├── test.py               # Model evaluation
├── utils.py              # Helper functions
├── models/               # Trained model(s)
├── data/                 # Dataset folder
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

---

## 🧩 Installation

### 1. Clone the repository
```bash
git clone https://github.com/shahdyasserelsayed/spam-ham-detector.git
cd spam-ham-detector
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Run the Streamlit App
```bash
streamlit run app.py
```

Then open the local URL shown (usually http://localhost:8501) in your browser.

### Predict Spam/Ham
- Enter a text message in the input box  
- Click **“Predict”**  
- View classification result instantly  

---

##  Retrain the Model
To retrain with new data:
```bash
python train.py
```

---

##  Requirements
- Python 3.8+
- Streamlit
- scikit-learn
- pandas
- numpy
- joblib

---

