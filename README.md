# CrimeClassNLP


🚔 Crime Classification Predictor

This project is a Streamlit web app for predicting the State/UT of a crime pattern using a pre-trained SVM model. It also provides data analysis and visualization tools for uploaded crime datasets.


---

📌 Features

✅ Load and use pre-trained SVM model (no retraining required)

✅ Predict the most likely State/UT based on crime text patterns

✅ Show confidence scores & top-5 predictions

✅ Interactive charts and probability distribution

✅ Upload a dataset (CSV) for data analytics & visualization

✅ User-friendly Streamlit UI



---

🛠️ Tech Stack

Python 3.9+

Streamlit (Web UI)

scikit-learn (Model handling)

Pandas, NumPy (Data handling)

Plotly Express (Visualizations)



---

📂 Project Structure

NLP Project/
│
├── Model 2/
│   ├── app.py                # Streamlit app
│   ├── saved_models/
│   │   ├── svm_linear_xxxxx/ # Pre-trained model folder
│   │   │   ├── model.pkl
│   │   │   ├── vectorizer.pkl
│   │   │   ├── metadata.pkl
│   │   │   ├── model_info.txt
│
├── sample_dataset.csv        # Example CSV for analytics
├── README.md                 # Documentation


---

⚡ Setup Instructions

1. Clone this repo or copy project files.


2. Create and activate a virtual environment:



python -m venv .venv
.venv\Scripts\activate   # On Windows
source .venv/bin/activate # On Linux/Mac

3. Install required dependencies:



pip install -r requirements.txt

If you don’t have a requirements.txt, install manually:

pip install streamlit pandas numpy plotly scikit-learn

4. Run the app:



streamlit run "Model 2/app.py"


---

🎯 Usage

Go to Model Prediction → Enter a crime pattern like:


district_mumbai crime_theft period_recent

App shows predicted State/UT with confidence & probability graph.

Go to Data Analysis → Upload a CSV file with crime data (must include STATE/UT and DISTRICT columns).



---

📊 Example Output

Prediction Page

Predicted State: Maharashtra

Confidence: 92%

Top-5 States probability chart


Analytics Page

Records count, number of States/UTs & Districts

Dataset preview




---

📌 Notes

Pre-trained model is already included under saved_models/.

Works offline, no internet required.

You can replace model.pkl and vectorizer.pkl with your own trained models.



---

👨‍💻 Author

Built by Shubham Kadam 🚀


---



