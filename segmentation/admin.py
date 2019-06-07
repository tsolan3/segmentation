from django.contrib import admin

# Register your models here.
from segmentation.models import Image


@admin.register(Image)
class ImageAdmin(admin.ModelAdmin):
    pass