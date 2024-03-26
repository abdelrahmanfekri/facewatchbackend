from django.contrib import admin

# Register your models here.
from .models import LabeledImage

admin.site.register(LabeledImage)
