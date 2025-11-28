# Tweet Virality Prediction (Machine Learning + Streamlit)

This project predicts tweet virality using machine learning.  
After testing multiple vectorization methods, the **CountVectorizer + Naive Bayes** model achieved **88% accuracy** and **98% recall** for viral tweets.

---

##  Features

- Text vectorization using **CountVectorizer** and **TF-IDF**
- Naive Bayes classification model
- Data cleaning & preprocessing
- **Streamlit web application**
- Real-time tweet virality prediction
- Prediction confidence score

---

##  Technologies Used

- Python
- Pandas
- Scikit-Learn
- CountVectorizer
- TF-IDF Vectorizer
- Naive Bayes
- Streamlit
- Joblib

---

##  Model Performance

Two models were tested:

| Vectorizer | Accuracy | Viral Recall | Viral Precision |
|----------|----------|---------------|----------------|
| **CountVectorizer** | **88%** | **98%** | 73% |
| TF-IDF | 83% | 46% | **95%** |

The **CountVectorizer + Naive Bayes** model was chosen for the Streamlit web app due to its high ability to correctly identify viral tweets.

---<img width="1887" height="838" alt="streamlit" src="https://github.com/user-attachments/assets/01384445-46a6-4642-8578-d0d1db956aca" />


##  How to Run the Web App

Make sure Python is installed, then run:

```bash
pip install streamlit joblib scikit-learn pandas
python -m streamlit run app.py
