from django.conf.urls import url, include

app_name = 'api'
urlpatterns = [
    url(r'', include('segmentation.urls')),
]
