from django.db import models

# Create your models here.


class LabeledImagePrediction(models.Model):
    image = models.ImageField(upload_to="images_prediction/")
    prediction = models.CharField(max_length=100, null=True, blank=True)
    true_label = models.CharField(max_length=100)
    prediction_json = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.prediction

