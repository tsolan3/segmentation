from rest_framework import serializers

from segmentation.models import Image


class CreateImageSerializer(serializers.ModelSerializer):

    class Meta:
        model = Image
        fields = (
            'image',
        )


class GetImageSerializer(serializers.ModelSerializer):

    class Meta:
        model = Image
        fields = (
            'image',
            'segmentation_type',
            'segmented_image',
        )
