
import streamlit as st
import joblib

# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="SMS Spam Detection System",
    page_icon="📩",
    layout="centered"
)

# ------------------------------------------------------------
# Load trained TF-IDF + Linear SVM model
# ------------------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("sms_spam_linear_svm.joblib")

model = load_model()

# ------------------------------------------------------------
# Dashboard
# ------------------------------------------------------------

st.title("SMS Spam Detection System")

st.write(
    "Cloud-Based SMS Classification Using "
    "TF-IDF and Linear Support Vector Machine"
)

st.divider()

st.subheader("Enter SMS Message")

message = st.text_area(
    "SMS Text",
    height=160,
    placeholder="Type or paste your SMS message here..."
)

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

if st.button("Classify SMS", type="primary"):

    if not message.strip():
        st.warning("Please enter an SMS message.")

    else:
        prediction = int(model.predict([message])[0])

        score = float(
            model.decision_function([message])[0]
        )

        if prediction == 1:
            st.error("SPAM - Suspicious Message")
        else:
            st.success("HAM - Legitimate Message")

        st.metric(
            label="SVM Decision Score",
            value=f"{score:.4f}"
        )

        st.caption(
            "The SVM decision score is a classification "
            "margin, not a calibrated probability."
        )

# ------------------------------------------------------------
# Example SMS messages
# ------------------------------------------------------------

st.divider()

st.subheader("Example SMS Messages")

examples = [
    "Hi, are we meeting today?",
    "Congratulations! Claim your free prize now!",
    "Please call me when you reach home.",
    "URGENT! You have won a cash reward. Reply now!"
]

for i, example in enumerate(examples, start=1):
    st.write(f"{i}. {example}")

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------

st.divider()

st.caption(
    "Research Prototype | TF-IDF + Linear SVM"
)
