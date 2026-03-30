python -c "
import joblib
m = joblib.load('ml_api/ml/sms_phishing_model_rf.pkl')
v = joblib.load('ml_api/ml/tfidf_vectorizer_rf.pkl')
print('Vectorizer features:', v.get_feature_names_out().shape)
print('Model input:', m.estimator.coef_.shape if hasattr(m, 'estimator') else 'unknown')
"