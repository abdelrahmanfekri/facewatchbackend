
from django.urls import path

from . import views

urlpatterns = [
    path("receive_images/", views.receive_images, name="receive_images"),
]


