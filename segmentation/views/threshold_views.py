from django.core.files import File
from rest_framework import status
from rest_framework.response import Response

from rest_framework.views import APIView

from segmentation.models import Image
from segmentation.serializers import CreateImageSerializer, GetImageSerializer

from PIL import Image as PILImage
from io import BytesIO

import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from math import *

SEGS = 6
STEP = 255/SEGS


def increase_brightness(img, value=65):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value

    v[v > lim] = 255
    v[v <= lim] += value
    v[v <= 75] = 0

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def initialize2(image):
    labels = np.zeros(shape=image.shape, dtype=np.uint8)

    for i in range(len(image)):
        for j in range(len(image[0])):  # для каждого пикселя изображения

            if image[i][j] < 40:
                l = 1
                labels[i][j] = l

            elif image[i][j] < 84:
                l = 2
                labels[i][j] = l

            elif image[i][j] < 120:
                l = 3
                labels[i][j] = l


            elif image[i][j] < 200:
                l = 4
                labels[i][j] = l

            elif image[i][j] < 230:
                l = 5
                labels[i][j] = l

            else:
                l = 6
                labels[i][j] = l

    return (labels)


def reconstruct(labs):
    labels = labs
    for i in range(len(labels)):
        for j in range(len(labels[0])):
            labels[i][j] = (labels[i][j] * 255) / (SEGS - 1)
    return labels


def auto_canny(image, sigma=0.33):
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


class ThresholdSegmentationView(APIView):

    def post(self, request):
        serializer = CreateImageSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        saved_image = Image.objects.create(segmentation_type=Image.THRESHOLD_TYPE)

        original = cv2.imdecode(np.fromstring(request.FILES['image'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)  # переводим его в градации серого
        labels = reconstruct(initialize2(img))  # присваиваем классы изображению и получаем нужные значения
        canny = auto_canny(labels)
        contoured = cv2.add(labels, canny)
        pil_image = PILImage.fromarray(contoured)
        saved_image.image.save(f'original_{saved_image.pk}.jpg', request.FILES['image'])

        blob = BytesIO()
        pil_image.save(blob, 'JPEG')
        saved_image.segmented_image.save(f'segmented_{saved_image.pk}.jpg', File(blob))

        return Response(status=status.HTTP_200_OK, data={'image': GetImageSerializer(saved_image,
                                                                                     context={'request': self.request}).data})