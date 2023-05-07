from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import LabeledImage
from mlops.tasks import handle_new_data

@receiver(post_save, sender=LabeledImage)
def handle_new_data_signal(sender, instance, created, **kwargs):
    if created:
        handle_new_data.delay(instance.id)



