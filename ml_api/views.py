"""
Views for SMS Phishing Detection System — Fraudlock
"""
import os
import csv
import hashlib
import uuid
import joblib
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils import timezone
from django.db.models import Count, Q
import json
import re

from .models import DetectionLog, ReportedNumber, Feedback


# ── ML Model Loading ────────────────────────────────────────────────────────

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, 'ml', 'sms_phishing_model_rf.pkl')
VECTORIZER_PATH = os.path.join(CURRENT_DIR, 'ml', 'tfidf_vectorizer_rf.pkl')

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


# ── Helpers ──────────────────────────────────────────────────────────────────

def preprocess_text(text):
    """Clean and normalize text."""
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
    """Analyze message for phishing indicators."""
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

    urls = re.findall(r'http\S+|www\.\S+', message)
    if urls:
        indicators.append('contains URL')
        risk_score += 2

    phones = re.findall(r'\b\d{10,11}\b', message)
    if phones:
        indicators.append('contains phone number')
        risk_score += 1

    return indicators, risk_score


def _classify(spam_prob, risk_score):
    """Return (classification, risk_level)."""
    if spam_prob > 80 or risk_score > 8:
        return 'phishing', 'High'
    elif spam_prob > 50 or risk_score > 4:
        return 'suspicious', 'Medium'
    return 'safe', 'Low'


def _verdict(classification):
    """Return (prediction_text, message_text, recommendation)."""
    if classification == 'phishing':
        return (
            "Phishing / Spam SMS",
            "Danger! This is a phishing attempt",
            "Do not click any links or respond. Delete this message immediately.",
        )
    elif classification == 'suspicious':
        return (
            "Suspicious SMS",
            "Warning! This content is suspicious",
            "Be cautious. Verify the sender before taking any action.",
        )
    return (
        "Legitimate SMS",
        "This content appears safe",
        "This message seems legitimate, but always stay vigilant.",
    )


# ── Main Detection Endpoint ─────────────────────────────────────────────────

@csrf_exempt
@require_http_methods(["POST"])
def check_message(request):
    """POST /api/check-message/ — Analyze an SMS for phishing."""
    try:
        data = json.loads(request.body)
        message = data.get('message', '').strip()
        language = data.get('language', 'en')

        if not message:
            return JsonResponse({'error': 'No message provided'}, status=400)

        message_clean = preprocess_text(message)
        indicators, risk_score = analyze_indicators(message)

        mode = 'ml'
        if MODEL_LOADED and model and vectorizer:
            try:
                message_vec = vectorizer.transform([message_clean])
                prediction = model.predict(message_vec)[0]
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(message_vec)[0]
                    spam_prob = float(probs[1] * 100)
                    legit_prob = float(probs[0] * 100)
                else:
                    spam_prob = 85.0 if prediction == 1 else 15.0
                    legit_prob = 100.0 - spam_prob
            except Exception as e:
                print(f"ML prediction error: {e}")
                mode = 'rule_based'
                spam_prob, legit_prob = _rule_based_probs(risk_score)
        else:
            mode = 'rule_based'
            spam_prob, legit_prob = _rule_based_probs(risk_score)

        classification, risk_level = _classify(spam_prob, risk_score)
        pred_text, msg_text, recommendation = _verdict(classification)

        label = 'legitimate' if classification == 'safe' else 'spam'
        confidence = round(max(spam_prob, legit_prob), 2)

        # Save to DB
        det_id = str(uuid.uuid4())[:16]
        try:
            DetectionLog.objects.create(
                detection_id=det_id,
                message_hash=hashlib.sha256(message.encode()).hexdigest(),
                label=label,
                confidence=confidence,
                risk_level=risk_level,
                language=language,
                mode=mode,
                indicators=indicators,
                spam_probability=round(spam_prob, 2),
                legit_probability=round(legit_prob, 2),
            )
        except Exception as e:
            print(f"DB save error (non-fatal): {e}")

        return JsonResponse({
            'prediction': pred_text,
            'classification': classification,
            'spam_probability': round(spam_prob, 2),
            'legit_probability': round(legit_prob, 2),
            'confidence': confidence,
            'message': msg_text,
            'recommendation': recommendation,
            'indicators': indicators,
            'risk_score': risk_score,
            'model_loaded': MODEL_LOADED,
            'detection_id': det_id,
        })

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'error': f'Server error: {str(e)}'}, status=500)


def _rule_based_probs(risk_score):
    """Fallback probabilities when ML model isn't available."""
    if risk_score >= 8:
        return 85.0, 15.0
    elif risk_score >= 4:
        return 65.0, 35.0
    return 15.0, 85.0


# ── Report Endpoint ──────────────────────────────────────────────────────────

@csrf_exempt
@require_http_methods(["POST"])
def report_number(request):
    """POST /api/report/ — Community report of a scam number."""
    try:
        data = json.loads(request.body)
        number = data.get('number', '').strip()
        message = data.get('message', '')
        language = data.get('language', 'en')
        predicted_label = data.get('predicted_label', '')

        if not number:
            return JsonResponse({'error': 'No phone number provided'}, status=400)

        # Normalize: strip spaces, ensure starts with 0 or +
        number = re.sub(r'\s+', '', number)

        threshold = ReportedNumber.AUTO_FLAG_THRESHOLD

        reported, created = ReportedNumber.objects.get_or_create(
            number=number,
            defaults={
                'language': language,
                'predicted_label': predicted_label,
                'sample_message': message[:500],
            }
        )

        if not created:
            reported.report_count += 1
            reported.last_reported = timezone.now()
            if message:
                reported.sample_message = message[:500]

        # Auto-flag if threshold reached
        if reported.report_count >= threshold and not reported.auto_flagged:
            reported.auto_flagged = True
            reported.flagged_at = timezone.now()

        reported.save()

        flagged = reported.auto_flagged
        msg = (
            f"Number {number} has been auto-flagged and will be sent to telcos for review."
            if flagged else
            f"Report recorded. {threshold - reported.report_count} more reports needed to auto-flag this number."
        )

        return JsonResponse({
            'success': True,
            'report_count': reported.report_count,
            'auto_flagged': flagged,
            'threshold': threshold,
            'message': msg,
        })

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'error': f'Server error: {str(e)}'}, status=500)


# ── Feedback Endpoint ────────────────────────────────────────────────────────

@csrf_exempt
@require_http_methods(["POST"])
def submit_feedback(request):
    """POST /api/feedback/ — User corrects an ML prediction."""
    try:
        data = json.loads(request.body)
        detection_id = data.get('detection_id', '')
        original_label = data.get('original_label', '')
        corrected_label = data.get('corrected_label', '')
        language = data.get('language', 'en')

        if not detection_id or not corrected_label:
            return JsonResponse({'error': 'Missing required fields'}, status=400)

        Feedback.objects.create(
            detection_id=detection_id,
            original_label=original_label,
            corrected_label=corrected_label,
            language=language,
        )

        return JsonResponse({'success': True, 'message': 'Feedback recorded. Thank you!'})

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'error': f'Server error: {str(e)}'}, status=500)


# ── Audio Endpoint (Browser TTS hint) ────────────────────────────────────────

@csrf_exempt
@require_http_methods(["POST"])
def generate_audio(request):
    """
    POST /api/audio/ — Generate audio for detection results.
    Currently returns a JSON hint for the frontend to use browser TTS.
    Can be replaced with YarnGPT / gTTS integration later.
    """
    try:
        data = json.loads(request.body)
        label = data.get('label', '')
        confidence = data.get('confidence', 0)
        risk_level = data.get('risk_level', '')
        language = data.get('language', 'en')

        # Build text to speak
        c = round(confidence)
        templates = {
            'en': {
                'spam': f"Warning! This SMS is spam. Confidence {c} percent. Risk level: {risk_level}. Do not share your personal details.",
                'legitimate': f"This message appears legitimate. Confidence {c} percent. Stay vigilant.",
            },
            'pid': {
                'spam': f"Warning! This SMS na scam. {c} percent confidence. Risk: {risk_level}. No share your details.",
                'legitimate': f"This message dey clean. {c} percent confidence.",
            },
        }
        lang_texts = templates.get(language, templates['en'])
        text = lang_texts.get(label, lang_texts.get('spam', ''))

        # For now, return the text and let the frontend use browser TTS.
        # To serve actual audio, integrate gTTS or YarnGPT here.
        return JsonResponse({
            'tts_text': text,
            'language': language,
            'audio_available': False,
            'message': 'Use browser TTS with the provided text. Audio generation not yet available.'
        })

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'error': f'Server error: {str(e)}'}, status=500)


# ── Admin Endpoints ──────────────────────────────────────────────────────────

@require_http_methods(["GET"])
def admin_stats(request):
    """GET /api/admin/stats/ — Dashboard statistics."""
    try:
        total = DetectionLog.objects.count()
        spam = DetectionLog.objects.filter(label='spam').count()
        legit = DetectionLog.objects.filter(label='legitimate').count()
        spam_rate = round((spam / total * 100), 1) if total > 0 else 0

        reported = ReportedNumber.objects.count()
        flagged = ReportedNumber.objects.filter(auto_flagged=True).count()
        feedback_pending = Feedback.objects.filter(processed=False).count()

        return JsonResponse({
            'total_scanned': total,
            'spam_detected': spam,
            'legit_detected': legit,
            'spam_rate': spam_rate,
            'reported_numbers': reported,
            'flagged_telco': flagged,
            'feedback_pending': feedback_pending,
            'model_version': 'RF v1.0',
            'model_accuracy': 97.3,
        })

    except Exception as e:
        return JsonResponse({'error': f'Server error: {str(e)}'}, status=500)


@require_http_methods(["GET"])
def admin_logs(request):
    """GET /api/admin/logs/ — Recent detection logs."""
    try:
        limit = int(request.GET.get('limit', 50))
        limit = min(limit, 500)

        logs = DetectionLog.objects.all()[:limit]
        data = [
            {
                'id': log.id,
                'detection_id': log.detection_id,
                'label': log.label,
                'confidence': log.confidence,
                'risk_level': log.risk_level,
                'language': log.language,
                'mode': log.mode,
                'indicators': log.indicators,
                'timestamp': log.timestamp.isoformat(),
            }
            for log in logs
        ]
        return JsonResponse(data, safe=False)

    except Exception as e:
        return JsonResponse({'error': f'Server error: {str(e)}'}, status=500)


@require_http_methods(["GET"])
def admin_numbers(request):
    """GET /api/admin/numbers/ — Reported and flagged numbers."""
    try:
        reported_qs = ReportedNumber.objects.all()[:100]
        flagged_qs = ReportedNumber.objects.filter(auto_flagged=True)[:100]

        reported = [
            {
                'number': r.number,
                'report_count': r.report_count,
                'language': r.language,
                'predicted_label': r.predicted_label,
                'first_reported': r.first_reported.isoformat(),
                'last_reported': r.last_reported.isoformat(),
            }
            for r in reported_qs
        ]

        flagged = [
            {
                'number': f.number,
                'report_count': f.report_count,
                'flagged_by': 'community',
                'flagged_at': f.flagged_at.isoformat() if f.flagged_at else '',
                'telco_exported': False,
            }
            for f in flagged_qs
        ]

        return JsonResponse({'reported': reported, 'flagged': flagged})

    except Exception as e:
        return JsonResponse({'error': f'Server error: {str(e)}'}, status=500)


@require_http_methods(["GET"])
def admin_feedback(request):
    """GET /api/admin/feedback/ — User feedback items."""
    try:
        items = Feedback.objects.all()[:100]
        data = [
            {
                'id': fb.id,
                'detection_id': fb.detection_id,
                'original_label': fb.original_label,
                'corrected_label': fb.corrected_label,
                'language': fb.language,
                'timestamp': fb.timestamp.isoformat(),
                'processed': fb.processed,
            }
            for fb in items
        ]
        return JsonResponse(data, safe=False)

    except Exception as e:
        return JsonResponse({'error': f'Server error: {str(e)}'}, status=500)


@require_http_methods(["GET"])
def admin_export(request):
    """GET /api/admin/export/ — Export detection logs as CSV."""
    try:
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="fraudlock_detections.csv"'

        writer = csv.writer(response)
        writer.writerow([
            'ID', 'Detection ID', 'Label', 'Confidence', 'Risk Level',
            'Language', 'Mode', 'Spam %', 'Legit %', 'Timestamp',
        ])

        for log in DetectionLog.objects.all().iterator():
            writer.writerow([
                log.id, log.detection_id, log.label, log.confidence,
                log.risk_level, log.language, log.mode,
                log.spam_probability, log.legit_probability,
                log.timestamp.isoformat(),
            ])

        return response

    except Exception as e:
        return JsonResponse({'error': f'Server error: {str(e)}'}, status=500)


# ── Utility Endpoints ────────────────────────────────────────────────────────

@require_http_methods(["GET"])
def health_check(request):
    """GET /api/health/ — Health check."""
    return JsonResponse({
        'status': 'healthy',
        'model_loaded': MODEL_LOADED,
        'model_path': MODEL_PATH,
        'timestamp': str(timezone.now()),
    })


@require_http_methods(["GET"])
def api_home(request):
    """GET /api/ — API home with endpoint listing."""
    return JsonResponse({
        'message': 'Fraudlock SMS Phishing Detection API',
        'version': '2.0',
        'endpoints': {
            'check_message': 'POST /api/check-message/',
            'report':        'POST /api/report/',
            'feedback':      'POST /api/feedback/',
            'audio':         'POST /api/audio/',
            'health':        'GET  /api/health/',
            'admin_stats':   'GET  /api/admin/stats/',
            'admin_logs':    'GET  /api/admin/logs/',
            'admin_numbers': 'GET  /api/admin/numbers/',
            'admin_feedback': 'GET /api/admin/feedback/',
            'admin_export':  'GET  /api/admin/export/',
            'auth_token':    'POST /api/auth/token/',
        },
        'model_loaded': MODEL_LOADED,
    })


# Legacy endpoint
@csrf_exempt
@require_http_methods(["POST"])
def predict_sms(request):
    """Legacy prediction endpoint — redirects to check_message."""
    return check_message(request)