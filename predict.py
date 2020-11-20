from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import pandas as pd
import os
import cv2
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.layers import GlobalMaxPooling2D

from fastai.vision import *
from fastai.callbacks import *

class Predict():

    print("Load Models  Start")
    model = load_model("./model/recommender")
    DATASET_PATH = "./static/myntradataset/"
    df = pd.read_csv(DATASET_PATH + "styles.csv", nrows=5000, error_bad_lines=False)
    print("Load Models completed")

    def img_path(img):
        return DATASET_PATH+"/images/"+img

    def load_image(img):
        return cv2.imread(img_path(img))

    def get_embedding(self, model, img_name):
        img = image.load_img(img_name, target_size=(80, 60)) ## size는 유지하시고 업로드 받은 파일로 받아주세요.
        x   = image.img_to_array(img)
        x   = np.expand_dims(x, axis=0)
        x   = preprocess_input(x)
        return model.predict(x).reshape(-1).tolist()

    def get_rec(self, get_img, top_n):
        df2 = pd.read_csv("./model/embeddings.csv", error_bad_lines=False).reset_index(drop=True)
        df2 = df2.drop(df2.columns[[0]], axis='columns')
        df3 = df2.append(df2.iloc[-1], ignore_index=True)
        df3.iloc[-1] = get_img
        cosine_sim = 1-pairwise_distances(df3, metric='cosine')
        indices = pd.Series(range(len(df)), index=df.index)
        sim_idx    = 5000
        sim_scores = list(enumerate(cosine_sim[sim_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]
        idx_rec    = [i[0] for i in sim_scores]
        idx_sim    = [i[1] for i in sim_scores]
        img_pth    = [str(df['id'].iloc[j]) for j in idx_rec]
        img_pt     = [DATASET_PATH+"images/"+im+'.jpg' for im in img_pth]

        return indices.iloc[idx_rec].index, idx_sim, img_pt

    # Get and display recs
    def get_recs(self, img_path, n=5):
        print("image_path : "  + img_path)
        imgg = self.get_embedding(model, img_path)
        idx_rec, idx_sim, img_pt = self.get_rec(imgg, n)

        return idx_rec, idx_sim, img_pt
