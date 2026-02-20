# ♻️ Recyclable Waste Classifier

## Aim
Integrate the RWC-Net (MobileNetV2) model into a web app for real-time classification of recyclable waste images, with an analytics dashboard.

## Features
- Classifies images into 6 categories: cardboard, glass, metal, paper, plastic, trash
- Interactive dashboard (category counts, trends, confidence)
- Streamlit web app interface
- Model trained using transfer learning (MobileNetV2)

## Run the Project
1. Activate environment  
   `venv\Scripts\activate`

2. Install dependencies  
   `pip install -r requirements.txt`

3. Train model  
   `python model/train_model.py`

4. Run web app  
   `streamlit run app/app.py`


