from django.urls import path
from . import views
from .views import *
from django.views.generic.base import TemplateView
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('img_upload/', PhotoUploadView.as_view(), name='img_upload'),
    path('img_upload_success/',
         TemplateView.as_view(template_name='image_upload/img_upload_success.html'), name='img_upload_success'),
    path('open_cvtr/',views.open_converter,name='open_converter'),
    path('get_data/', views.get_data, name='get_data'),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
