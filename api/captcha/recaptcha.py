import numpy as np
import cv2
import nltk
import spacy
import string
from itertools import chain
from nltk.corpus import wordnet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from typing import Tuple, List

nlp = spacy.load("en_core_web_lg")

nltk.download("wordnet")
nltk.download("omw-1.4")

model = VGG16(weights="imagenet")

def parse_image_grid(img_grid: np.ndarray, row, col) -> List[np.ndarray]:
    ww = [[i.min(), i.max()] for i in np.array_split(range(img_grid.shape[0]),row)]
    hh = [[i.min(), i.max()] for i in np.array_split(range(img_grid.shape[1]),col)]
    grid = [img_grid[j:jj,i:ii,:] for j,jj in ww for i,ii in hh]
    return grid

def make_predictions(imgs: List) -> List:
    # Resize from (x, x, 3) -> (224, 224, 3) to fit model input shape
    imgs = [cv2.resize(img, (224, 224)) for img in imgs]
    # Resize to (image_count, 224, 224, 3) to fit model input shape
    imgs = np.asarray(imgs)
    processed_imgs = preprocess_input(imgs)
    features = model.predict(processed_imgs)
    return decode_predictions(features)

def text_norm(text: str) -> str:
    doc = nlp(text)
    lemma = [token.lemma_ for token in doc]
    rejoined = " ".join([word for word in lemma if nlp.vocab[word].is_stop == False and not word in string.punctuation])
    return rejoined.strip()

i = 1
def is_match(text: str, predictions: List) -> List[bool]:
    doc = nlp(text)
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    phrases_to_match = set([text_norm(np) for np in noun_phrases])
    prediction_syn_lemmas = []
    for prediction in predictions:
        (_, label, score) = prediction
        prediction_syn_lemmas += chain.from_iterable([
            [n.replace("_", " ") for n in ss.lemma_names()]
            for ss in wordnet.synsets(label)
        ])
    prediction_syn_lemmas = set(prediction_syn_lemmas)

    global i
    print(i, phrases_to_match, prediction_syn_lemmas)
    i += 1
    return len(phrases_to_match.intersection(prediction_syn_lemmas)) > 0
    
def solve(text: str, img_bytes: bytes):
    img_grid = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), -1)
    imgs = parse_image_grid(img_grid, 3, 3)
    all_predictions = make_predictions(imgs)
    return [is_match(text, predictions) for predictions in all_predictions]