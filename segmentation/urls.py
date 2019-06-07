from django.conf.urls import url, include

from segmentation.views.kmeans_views import KMeansSegmentationView
from segmentation.views.mrf_views import MRFSegmentationView
from segmentation.views.threshold_views import ThresholdSegmentationView

urlpatterns = [
    url(r'^segmentation/threshold/$', ThresholdSegmentationView.as_view()),
    url(r'^segmentation/markov-random-field/$', MRFSegmentationView.as_view()),
    url(r'^segmentation/kmeans/$', KMeansSegmentationView.as_view()),

]
