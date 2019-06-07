from django.core.files import File
from rest_framework import status
from rest_framework.response import Response
import imutils
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
STEP = 255 / SEGS
NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BETA = 1
TEMPERATURE = 10
ITERATIONS = 1000000
COOLRATE = 0.9


# функция проверки того, что мы не вышли за пределы изображения
def isSafe(a_b, x, y):
    a, b = a_b
    return 0 <= x < a and 0 <= y < b


# функция дельты Кронекера
def delta(i, j):
    if i == j:
        return -BETA
    return BETA


# получение градаций серого на основе принадлежности к сегменту
def reconstruct(labs):
    labels = labs
    for i in range(len(labels)):
        for j in range(len(labels[0])):
            labels[i][j] = (labels[i][j] * 255) / (SEGS - 1)
    return labels


# подсчет дисперсии
def variance(sums1, squares1, nos1):
    return abs(squares1 / (nos1+0.01) - (sums1 / (nos1+0.01))) + 0.1


# подсчет энергии
def calculate_energy(img, variances, labels):
    """
    :param img: изображение
    :param variances: дисперсии отношений к сегментам
    :param labels: классы, присвоенные пикселям
    :return: энергия
    """
    energy = 0.0
    for i in range(len(img)):
        for j in range(len(img[0])): # для каждого пикселя
            l = labels[i][j] # получаем его класс
            energy += log(sqrt(2*np.pi*variances[l]))
            for (p, q) in NEIGHBORS:
                if isSafe(img.shape, i + p, j + q):
                    energy += (delta(l, labels[i + p][j + q]) / 2.0)
    return energy

# задаем начальное состояние системы
def initialize(img):
    labels = np.zeros(shape=img.shape, dtype=np.uint8)
    nos = [0.0] * SEGS
    sums = [0.0] * SEGS
    squares = [0.0] * SEGS
    for i in range(len(img)):
        for j in range(len(img[0])): # для каждого пикселя изображения
            l = randint(0, SEGS - 1) # присваиваем рандомный класс каждому пикселю
            sums[l] += img[i][j] # сумма значений пикселей данного класса
            squares[l] += img[i][j] ** 2 # сумма квадратов значений пикселя данного класса
            nos[l] += 1.0 # количество пикселей данного класса
            labels[i][j] = l
    return sums, squares, nos, labels


# задаем начальное состояние системы
def initialize2(image):
    labels = np.zeros(shape=image.shape, dtype=np.uint8)
    nos = [1.0] * SEGS
    sums = [0.0] * SEGS
    squares = [0.0] * SEGS
    for i in range(len(image)):
        for j in range(len(image[0])): # для каждого пикселя изображения
            for s in range(SEGS):
                if image[i][j] < (s+1)*STEP:
                    l = s
                    sums[l] += image[i][j] # сумма значений пикселей данного класса
                    squares[l] += image[i][j] ** 2 # сумма квадратов значений пикселя данного класса
                    nos[l] += 1.0 # количество пикселей данного класса
                    labels[i][j] = l
                    break

    return sums, squares, nos, labels

def is_contour_bad(c):
    peri = cv2.contourArea(c)
    peri = len(c)
    return peri >= 5

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


def auto_canny(image, sigma=3):
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


class MRFSegmentationView(APIView):

    def post(self, request):
        serializer = CreateImageSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        saved_image = Image.objects.create(segmentation_type=Image.MRF_TYPE)
        original = cv2.imdecode(np.fromstring(request.FILES['image'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
        brighted = increase_brightness(original)
        # cv2.imwrite(f"brighted_{image}", brighted)
        img = cv2.cvtColor(brighted, cv2.COLOR_BGR2GRAY)  # переводим его в г
        sums, squares, nos, labels = initialize2(img)  # присваиваем классы изображению и получаем нужные значения
        variances = [variance(sums[i], squares[i], nos[i]) for i in range(SEGS)]  # считаем дисперсии
        energy = calculate_energy(img, variances, labels)  # считаем энергию

        # алгоритм имитации отжига
        temp = TEMPERATURE
        it = ITERATIONS
        new_energies = []
        while it > 0:
            (a, b) = img.shape  # размер изображения (axb пискелей)
            change = False  # если True, то класс меняется
            x = randint(0, a - 1)
            y = randint(0, b - 1)  # берем произвольный пиксель
            val = float(img[x][y])  # значение градации серого
            l = labels[x][y]  # получаем его класс
            newl = l
            while newl == l:
                newl = randint(0, SEGS - 1)  # берем любой другой класс

            # значения, которые поменяются при изменении класса
            val = float(val)
            remsums = sums[l] - val
            addsums = sums[newl] + val

            remsquares = squares[l] - val * val
            addsquares = squares[newl] + val * val

            remvar = variance(remsums, remsquares, nos[l] - 1)
            addvar = variance(addsums, addsquares, nos[newl] + 1)

            # получаем новое значение энергии
            newenergy = energy
            newenergy -= log(sqrt(2 * np.pi * variance(sums[l], squares[l], nos[l]))) * (nos[l])
            newenergy += log(sqrt(2 * np.pi * remvar)) * (nos[l] - 1)
            newenergy -= log(sqrt(2 * np.pi * variance(sums[newl], squares[newl], nos[newl]))) * (nos[newl])
            newenergy += log(sqrt(2 * np.pi * addvar)) * (nos[newl] + 1)
            for (p, q) in NEIGHBORS:
                if isSafe((a, b), x + p, y + q):
                    newenergy -= delta(l, labels[x + p][y + q])
                    newenergy += delta(newl, labels[x + p][y + q])
            new_energies.append(newenergy)
            # если новая энергия меньше, однозначно меняем значение
            if newenergy < energy:
                change = True
            # если больше, то меняем с вероятностью
            else:
                prob = 1.1
                if temp != 0:
                    prob = np.exp((energy - newenergy) / temp)
                if prob >= (randint(0, 1000) + 0.0) / 1000:
                    change = True

            # соответственно, меняем все параметры, если энергия меняется
            if change:
                labels[x][y] = newl
                energy = newenergy

                nos[l] -= 1
                sums[l] = remsums
                squares[l] = remsquares

                nos[newl] += 1
                sums[newl] = addsums
                squares[newl] = addsquares

            # понижаем температуру и идем дальше
            temp *= COOLRATE
            it -= 1

        canny = auto_canny(labels)

        cnts = cv2.findContours(canny.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        mask = np.ones(labels.shape[:2], dtype="uint8")

        # loop over the contours
        for c in cnts:
            # if the contour is bad, draw it on the mask
            if is_contour_bad(c):
                cv2.drawContours(mask, [c], -1, 255, 1)
                # remove the contours from the image and show the resulting images
        # labels = cv2.bitwise_and(labels, labels, mask=mask)
        # cv2.imshow("Mask", mask)
        # cv2.imshow("After", image)
        # cv2.waitKey(0)
        contoured = cv2.add(labels, mask)

        pil_image = PILImage.fromarray(contoured)
        saved_image.image.save(f'original_{saved_image.pk}.jpg', request.FILES['image'])

        blob = BytesIO()
        pil_image.save(blob, 'JPEG')
        saved_image.segmented_image.save(f'segmented_{saved_image.pk}.jpg', File(blob))

        return Response(status=status.HTTP_200_OK, data={'image': GetImageSerializer(saved_image,
                                                                                     context={
                                                                                         'request': self.request}).data})

