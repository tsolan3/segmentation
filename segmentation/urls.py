from django.conf.urls import url, include

from segmentation.views.threshold_views import ThresholdSegmentationView

urlpatterns = [
    url(r'^segmentation/threshold/$', ThresholdSegmentationView.as_view()),
]
