"""
Views for SMS Phishing Detection System
"""
import os
import joblib
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils import timezone
import json
import re

# Get the directory where views.py is located
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, 'ml', 'sms_phishing_model_rf.pkl')
VECTORIZER_PATH = os.path.join(CURRENT_DIR, 'ml', 'tfidf_vectorizer_rf.pkl')

# Try to load models
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    MODEL_LOADED = True
    print(" ML models loaded successfully from:", CURRENT_DIR)
except Exception as e:
    print(f"Error loading models: {e}")
    print(f"Looking for models at: {MODEL_PATH}")
    MODEL_LOADED = False
    model = None
    vectorizer = None


def preprocess_text(text):
    """Clean and normalize text"""
    if not text:
        return ""

    text = text.lower()
    text = ' '.join(text.split())
    text = re.sub(r'http\S+|www\.\S+', ' URL ', text)
    text = re.sub(r'\b\d{10,11}\b', ' PHONE ', text)
    text = re.sub(r'\S+@\S+', ' EMAIL ', text)
    text = re.sub(r'[$₦£€]\d+|\d+\s*(naira|dollars|pounds)', ' MONEY ', text, flags=re.IGNORECASE)
    text = re.sub(r'\d+', ' NUMBER ', text)
    
    return text


def analyze_indicators(message):
    """Analyze message for phishing indicators"""
    phishing_keywords = {
        'urgent': 2, 'click here': 3, 'verify': 2, 'confirm': 2,
        'suspended': 3, 'expire': 2, 'winner': 3, 'won': 3,
        'prize': 2, 'claim': 2, 'congratulations': 2, 'pin': 3,
        'password': 3, 'bvn': 3, 'account number': 3, 'atm': 2,
        'bank': 1, 'update': 2, 'immediately': 2, 'act now': 3,
        'limited time': 2, 'free': 1, 'cash': 2, 'reward': 2,
    }
    
    message_lower = message.lower()
    indicators = []
    risk_score = 0
    
    for keyword, weight in phishing_keywords.items():
        if keyword in message_lower:
            indicators.append(keyword)
            risk_score += weight
    
    # Check for URLs
    urls = re.findall(r'http\S+|www\.\S+', message)
    if urls:
        indicators.append('contains URL')
        risk_score += 2
    
    # Check for phone numbers
    phones = re.findall(r'\b\d{10,11}\b', message)
    if phones:
        indicators.append('contains phone number')
        risk_score += 1
    
    return indicators, risk_score


@csrf_exempt
@require_http_methods(["POST"])
def check_message(request):
    """Main endpoint: Check if SMS is phishing or legitimate"""
    try:
        data = json.loads(request.body)
        message = data.get('message', '').strip()
        
        if not message:
            return JsonResponse({
                'error': 'No message provided'
            }, status=400)
        
        # Preprocess
        message_clean = preprocess_text(message)
        
        # Get indicators
        indicators, risk_score = analyze_indicators(message)
        
        # Predict using ML model
        if MODEL_LOADED and model and vectorizer:
            try:
                message_vec = vectorizer.transform([message_clean])
                prediction = model.predict(message_vec)[0]
                
                # Get probabilities
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(message_vec)[0]
                    spam_prob = float(probs[1] * 100)
                    legit_prob = float(probs[0] * 100)
                else:
                    spam_prob = 85.0 if prediction == 1 else 15.0
                    legit_prob = 15.0 if prediction == 1 else 85.0
                
                # Classify
                if spam_prob > 80 or risk_score > 8:
                    classification = 'phishing'
                elif spam_prob > 50 or risk_score > 4:
                    classification = 'suspicious'
                else:
                    classification = 'safe'
                
            except Exception as e:
                print(f"ML prediction error: {e}")
                # Fallback to rule-based
                if risk_score >= 8:
                    classification = 'phishing'
                    spam_prob = 85.0
                    legit_prob = 15.0
                elif risk_score >= 4:
                    classification = 'suspicious'
                    spam_prob = 65.0
                    legit_prob = 35.0
                else:
                    classification = 'safe'
                    spam_prob = 15.0
                    legit_prob = 85.0
        else:
            # Rule-based fallback
            if risk_score >= 8:
                classification = 'phishing'
                spam_prob = 85.0
                legit_prob = 15.0
            elif risk_score >= 4:
                classification = 'suspicious'
                spam_prob = 65.0
                legit_prob = 35.0
            else:
                classification = 'safe'
                spam_prob = 15.0
                legit_prob = 85.0
        
        # Determine result message
        if classification == 'phishing':
            result = "🚨 Phishing / Spam SMS"
            message_text = "Danger! This is a phishing attempt"
            recommendation = "Do not click any links or respond. Delete this message immediately."
        elif classification == 'suspicious':
            result = "⚠️ Suspicious SMS"
            message_text = "Warning! This content is suspicious"
            recommendation = "Be cautious. Verify the sender before taking any action."
        else:
            result = "✅ Legitimate SMS"
            message_text = "This content appears safe"
            recommendation = "This message seems legitimate, but always stay vigilant."
        
        return JsonResponse({
            'prediction': result,
            'classification': classification,
            'spam_probability': round(spam_prob, 2),
            'legit_probability': round(legit_prob, 2),
            'confidence': round(max(spam_prob, legit_prob), 2),
            'message': message_text,
            'recommendation': recommendation,
            'indicators': indicators,
            'risk_score': risk_score,
            'model_loaded': MODEL_LOADED
        })
        
    except json.JSONDecodeError:
        return JsonResponse({
            'error': 'Invalid JSON'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'error': f'Server error: {str(e)}'
        }, status=500)


@require_http_methods(["GET"])
def health_check(request):
    """Health check endpoint"""
    return JsonResponse({
        'status': 'healthy',
        'model_loaded': MODEL_LOADED,
        'model_path': MODEL_PATH,
        'timestamp': str(timezone.now())
    })


@require_http_methods(["GET"])
def api_home(request):
    """API home endpoint - shows available endpoints"""
    return JsonResponse({
        'message': 'SMS Phishing Detection API',
        'version': '1.0',
        'endpoints': {
            'check_message': '/api/check-message/ (POST)',
            'health': '/api/health/ (GET)',
        },
        'example_request': {
            'url': '/api/check-message/',
            'method': 'POST',
            'body': {
                'message': 'Your SMS message here',
                'user_id': 'optional_user_id',
                'language': 'en'
            }
        },
        'model_loaded': MODEL_LOADED
    })


# Legacy endpoint for backward compatibility
@csrf_exempt
@require_http_methods(["POST"])
def predict_sms(request):
    """Legacy prediction endpoint - redirects to check_message"""
    return check_message(request)