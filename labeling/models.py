from django.db import models

# Create your models here.


class LabeledImage(models.Model):
    image = models.ImageField(upload_to="images/")
    label = models.CharField(max_length=100)
    trained = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.label
    
