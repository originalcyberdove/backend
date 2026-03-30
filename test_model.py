import joblib
import scipy.sparse as sp
import numpy as np
 
m = joblib.load('ml_api/ml/sms_phishing_model_rf.pkl')
v = joblib.load('ml_api/ml/tfidf_vectorizer_rf.pkl')

tests = [
    "i am a girl",
    "Congratulations! You won N500,000. Send your BVN now to claim.",
    "Your account has been suspended. Verify your OTP immediately.",
    "Hello, how are you doing today?",
]

for text in tests:
    vec = v.transform([text])
    manual = np.zeros((1, 8))
    full = sp.hstack([vec, sp.csr_matrix(manual)])
    proba = m.predict_proba(full)[0]
    print(f"Text    : {text[:60]}")
    print(f"Legit   : {proba[0]*100:.1f}%  |  Spam: {proba[1]*100:.1f}%")
    print(f"Model   : {type(m).__name__}")
    print()