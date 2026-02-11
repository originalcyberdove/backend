"""
URL configuration for ML API
"""
from django.urls import path
from . import views

urlpatterns = [
    path('', views.api_home, name='api_home'),
    path('health/', views.health_check, name='health_check'),
    path('check-message/', views.check_message, name='check_message'),
    path('predict/', views.predict_sms, name='predict_sms'),  # Legacy endpoint
]