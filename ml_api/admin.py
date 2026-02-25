from django.contrib import admin
from .models import DetectionLog, ReportedNumber, Feedback


@admin.register(DetectionLog)
class DetectionLogAdmin(admin.ModelAdmin):
    list_display = ('detection_id', 'label', 'confidence', 'risk_level', 'language', 'mode', 'timestamp')
    list_filter = ('label', 'risk_level', 'language', 'mode')
    search_fields = ('detection_id',)
    readonly_fields = ('detection_id', 'timestamp')
    ordering = ('-timestamp',)


@admin.register(ReportedNumber)
class ReportedNumberAdmin(admin.ModelAdmin):
    list_display = ('number', 'report_count', 'auto_flagged', 'language', 'first_reported', 'last_reported')
    list_filter = ('auto_flagged', 'language')
    search_fields = ('number',)
    ordering = ('-report_count',)


@admin.register(Feedback)
class FeedbackAdmin(admin.ModelAdmin):
    list_display = ('detection_id', 'original_label', 'corrected_label', 'language', 'processed', 'timestamp')
    list_filter = ('processed', 'language')
    ordering = ('-timestamp',)
