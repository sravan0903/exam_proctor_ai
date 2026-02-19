import torch
import numpy as np
import cv2
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

model = None


def get_model():
    global model
    if model is None:
        print("Loading FaceNet model...")
        model = InceptionResnetV1(pretrained='vggface2').eval()
    return model


def extract_face_embedding(face_img):
    """
    Takes a cropped face image
    Returns a 512-d embedding
    """

    model = get_model()

    face = cv2.resize(face_img, (160, 160))
    face = face.astype(np.float32) / 255.0
    face = np.transpose(face, (2, 0, 1))
    face = np.expand_dims(face, axis=0)

    with torch.no_grad():
        embedding = model(torch.tensor(face)).numpy()

    return embedding


def is_same_person(live_embedding, stored_embedding, threshold=0.7):
    similarity = cosine_similarity(live_embedding, stored_embedding)[0][0]
    return similarity >= threshold, similarity
