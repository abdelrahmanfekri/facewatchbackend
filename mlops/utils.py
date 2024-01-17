import tensorflow as tf
import numpy as np
import cv2
import joblib
from deepface import DeepFace
from .models import LabeledImagePrediction
from keras_facenet import FaceNet

facenet = FaceNet()

emotion_model = tf.keras.models.load_model(
    "C:/Users/UG/Desktop/research backend/mlops/models/Emotion_vgg.h5"
)
disease_model = tf.keras.models.load_model(
    "C:/Users/UG/Desktop/research backend/mlops/models/diseaseDetectionModel.h5"
)
fatigue_model = joblib.load(
    "C:/Users/UG/Desktop/research backend/mlops/models/knn_model.joblib"
)

diseases_values = [
    "Alopecia Hair Loss",
    "Butterfly Rash Face",
    "Dehydration Cracked Libs",
    "Drooping Eyelid",
    "Jaundice Yellowish Skin and Eyes",
    "Melasma Face",
    "Moles Face",
    "Normal",
    "Puffy Eyes Face",
    "Sores in face",
    "Stroke Face",
    "Xanthelasma Yellow Spots on Your Eyelids",
]

fatigue_classes = ["tired", "non_vigilant", "alert"]

emotion_classes = [
    "happy",
    "neutral",
    "sad",
    "angry",
    "fear",
    "disgust",
    "surprise",
]


def detect_face(image):
    try:
        analysis = DeepFace.extract_faces(image)
        return image[
            analysis[0]["facial_area"]["y"] : analysis[0]["facial_area"]["y"]
            + analysis[0]["facial_area"]["h"],
            analysis[0]["facial_area"]["x"] : analysis[0]["facial_area"]["x"]
            + analysis[0]["facial_area"]["w"],
        ]
    except Exception as e:
        print("detect face image", e)


def normalize_image(img):
    img = img.astype(np.float32)
    img /= 255.0
    img -= np.mean(img)
    img /= np.std(img)
    return img


def predict_diseases(image):
    face_image = detect_face(image)
    img = cv2.resize(face_image, (160, 160))
    img = normalize_image(img)
    img = np.expand_dims(img, axis=0)
    pred = disease_model.predict(img)
    diseases_prediction = {
        diseases_values[i]: pred[0][i] for i in range(len(diseases_values))
    }
    pred = np.argmax(pred)
    pred = diseases_values[pred]
    return pred, diseases_prediction


def train_diseases(image, label):
    face_image = detect_face(image)
    img = cv2.resize(face_image, (160, 160))
    img = normalize_image(img)
    img = np.expand_dims(img, axis=0)
    binary_label = np.zeros(len(diseases_values))
    binary_label[diseases_values.index(label)] = 1
    label = np.expand_dims(binary_label, axis=0)
    # update the model with the new image
    disease_model.fit(img, label, epochs=1)
    # save the updated model
    disease_model.save(
        "C:/Users/UG/Desktop/research backend/mlops/models/diseaseDetectionModel.h5"
    )


def predict_fatigue(image):
    face_image = detect_face(image)
    img = cv2.resize(face_image, (160, 160))
    img = np.expand_dims(img, axis=0)
    embedding = facenet.embeddings(img)
    pred_json = fatigue_model.predict(embedding)
    pred = pred_json[0]
    return pred, pred_json


def train_fatigue(image, label):
    face_image = detect_face(image)
    img = cv2.resize(face_image, (160, 160))
    img = np.expand_dims(img, axis=0)
    embedding = facenet.embeddings(img)
    fatigue_model.fit(embedding, [label])
    joblib.dump(
        fatigue_model,
        "C:/Users/UG/Desktop/research backend/mlops/models/knn_model.joblib",
    )


def predict_emotion(image):
    face_image = detect_face(image)
    gray_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray_image, (48, 48))
    img = np.reshape(img, (48, 48, 1))
    img = np.expand_dims(img, axis=0)
    print(img.shape)
    pred_json = emotion_model.predict(img)
    pred = np.argmax(pred_json)
    pred = emotion_classes[pred]
    print(pred)
    print(pred_json)
    return pred, pred_json


def train_emotion(image, label):
    face_image = detect_face(image)
    gray_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray_image, (48, 48))
    img = np.reshape(img, (48, 48, 1))
    img = np.expand_dims(img, axis=0)
    binary_label = np.zeros(len(emotion_classes))
    binary_label[emotion_classes.index(label)] = 1
    label = np.expand_dims(binary_label, axis=0)
    # update the model with the new image
    emotion_model.fit(img, label, epochs=1)
    # save the updated model
    emotion_model.save(
        "C:/Users/UG/Desktop/research backend/mlops/models/Emotion_vgg.h5"
    )


def handle_new_data_utils(label_obj, prediction, training):
    image = label_obj.image
    label = label_obj.label
    image_file = label_obj.image.open("rb")
    image = image_file.read()
    image = np.frombuffer(image, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    pred, diseases_prediction = prediction(image)
    training(image, label)
    diseases_prediction = str(diseases_prediction)
    LabeledImagePrediction.objects.create(
        image=label_obj.image,
        prediction=pred,
        true_label=label,
        prediction_json=diseases_prediction,
    )
    label_obj.trained = True
    label_obj.save()
