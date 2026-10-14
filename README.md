# SMS Spam Detection System

## Cloud-Based SMS Spam Detection Using TF-IDF and Linear SVM

This research prototype classifies SMS messages as
Ham (legitimate) or Spam (unwanted).

## Machine Learning Framework

- Feature Extraction: TF-IDF
- Classifier: Linear Support Vector Machine
- Classification: Binary (Ham/Spam)
- Application Framework: Streamlit
- Cloud Platform: Streamlit Community Cloud

## Model Deployment

The application uses a previously trained
TF-IDF + Linear SVM pipeline saved using Joblib.

The SVM decision score represents a classification
margin and is not a calibrated probability.
