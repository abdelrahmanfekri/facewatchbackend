from .utils import *
from celery import shared_task
from labeling.models import LabeledImage
import subprocess


@shared_task()
def handle_new_data(label_obj_id):
    label_obj = LabeledImage.objects.get(id=label_obj_id)
    label = label_obj.label
    if label in diseases_values:
        handle_new_data_utils(
            label_obj, prediction=predict_diseases, training=train_diseases
        )
    elif label in fatigue_classes:
        handle_new_data_utils(
            label_obj, prediction=predict_fatigue, training=train_fatigue
        )
    elif label in emotion_classes:
        handle_new_data_utils(
            label_obj, prediction=predict_emotion, training=train_emotion
        )
    else:
        print("label not found")


version = 1


@shared_task()
def my_periodic_task():
    print("new version for models")
    subprocess.run(["git", "-C", "mlops/models", "add", "."])
    subprocess.run(
        ["git", "-C", "mlops/models", "commit", "-m", f"new version {version}"]
    )
    subprocess.run(["git", "-C", "mlops/models", "push", "origin", "main"])
