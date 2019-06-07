from django.db import models


class Image(models.Model):
    THRESHOLD_TYPE = 'threshold'
    MRF_TYPE = 'mrf'
    KMEANS_TYPE = 'kmeans'

    SEGMENTATION_TYPES = (
        (THRESHOLD_TYPE, 'Threshold Segmentation'),
        (MRF_TYPE, 'Markov Random Field Segmentation'),
        (KMEANS_TYPE, 'K-Means Segmenation')
    )
    segmentation_type = models.CharField('Segmentation type', choices=SEGMENTATION_TYPES, max_length=30)
    image = models.ImageField('Image')
    segmented_image = models.ImageField('Segmented Image')
