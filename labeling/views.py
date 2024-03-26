from django.shortcuts import render

# Create your views here.

from django.http import JsonResponse
from .models import LabeledImage

from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def receive_images(request):
    if request.method == "POST":
        images = request.FILES.getlist("images")
        labels = request.POST.getlist("labels")
        for image, label in zip(images, labels):
            labeled_image = LabeledImage(image=image, label=label)
            labeled_image.save()

        return JsonResponse({"status": "success"})
    else:
        return JsonResponse(
            {"status": "error", "message": "Method not allowed"}, status=405
        )
