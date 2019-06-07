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


def auto_canny(image, sigma=0.33):
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


class KMeansSegmentationView(APIView):

    def post(self, request):
        serializer = CreateImageSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        saved_image = Image.objects.create(segmentation_type=Image.KMEANS_TYPE)

        original = cv2.imdecode(np.fromstring(request.FILES['image'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)  # переводим его в градации серого
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        Z = img.reshape((-1, 3))
        # convert to np.float32
        Z = np.float32(Z)
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 6
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))
        canny = auto_canny(res2)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contoured = cv2.add(gray, canny)
        pil_image = PILImage.fromarray(contoured)
        saved_image.image.save(f'original_{saved_image.pk}.jpg', request.FILES['image'])

        blob = BytesIO()
        pil_image.save(blob, 'JPEG')
        saved_image.segmented_image.save(f'segmented_{saved_image.pk}.jpg', File(blob))

        return Response(status=status.HTTP_200_OK, data={'image': GetImageSerializer(saved_image,
                                                                                     context={'request': self.request}).data})