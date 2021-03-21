import base64

import numpy as np
import pandas as pd
import os
import tensorflow as tf
import tensorflow.keras as keras
from keras import Model
from keras.applications import DenseNet121
from keras.preprocessing import image
from keras.applications.densenet import preprocess_input, decode_predictions
from keras.utils import plot_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import pathlib
from keras.layers import *
from keras.models import *
from sklearn.metrics.pairwise import linear_kernel
from PIL import Image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

img_width = 200
img_height = 200


lista = []

# upload csv
df_embedding = pd.read_csv('csv/df_embedding_6k.csv')
df_embedding.head(5)
df = pd.read_csv('csv/df.csv')
df.head(5)
# calcolate cosine similarity
cosine_sim = linear_kernel(df_embedding, df_embedding)

#load model
model = load_model('model/baseModel.h5')
#model._make_predict_function()
print('Model loaded. Check http://127.0.0.1:5000/')

def img_path(img):
    return img

def model_predict(model, img_name):
    # Reshape
    img = image.load_img(img_path(img_name), target_size=(img_width, img_height))
    # img to Array
    x = image.img_to_array(img)
    # Expand Dim (1, w, h)
    x = np.expand_dims(x, axis=0)
    # Pre process Input
    x = preprocess_input(x)
    return model.predict(x).reshape(-1)



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        print(f)
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join("images", secure_filename(f.filename))
        #f.save(file_path)
        #print("FILE PATH:" + file_path)
        image_name = secure_filename(f.filename)
        #print("Image name:" + image_name)

        # Make prediction
        y = model_predict(model, file_path)
        y = pd.Series(y)

        # Embedding
        list_temp = df_embedding.values.tolist()
        list_temp.append(y)
        df_temp = pd.DataFrame(list_temp)

        # Acquisisco chosen_img_indx andando a vedere quanti sono gli elementi all'interno di df_temp (dopo aver aggiunto y)
        # Assegno chosen_img_indx = ind - 1, poiché l'indice parte da 0.
        ind = len(df_temp)
        chosen_img_indx = ind - 1

        # Per salvare il nuovo dataframe
        # df_temp.to_csv(r'drive/My Drive/Dataset/fashion-dataset/embedding7.csv')

        cosine_sim = linear_kernel(df_temp, df_temp)
        indices = pd.Series(range(len(df_temp)), index=df_temp.index)
        recom = get_recom(chosen_img_indx, df, indices, cosine_sim)
        recom_list = recom.to_list()

        result = str(recom_list) # Convert to string
        return result
    return None

def get_recom(index, df, indices, cosine_sim=cosine_sim):
    idx = indices[index]
    print(idx)
    # Get the pairwsie similarity scores of all clothes with that one
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the clothes based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 5 most similar clothes
    sim_scores = sim_scores[1:6]

    # Get the clothes indices
    cloth_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar clothes
    return df['image'].iloc[cloth_indices]


if __name__ == '__main__':
    app.run(debug=True)

###chosen image
#img_name = 'images/jeans.jpg'
#chosen_img =  mpimg.imread(img_name)

# Per selezionare un'immagine preesistente nella cartella images
#chosen_img =  mpimg.imread(path + 'images/' + img_name)

# Per selezionare un'immagine preesistente nel df
#chosen_img =  mpimg.imread(path + 'images/' + df.iloc[chosen_img_indx].image)


'''y = model_predict(model, img_name)
y = pd.Series(y)


#Embedding
list_temp = df_embedding.values.tolist()
list_temp.append(y)
df_temp = pd.DataFrame(list_temp)

# Acquisisco chosen_img_indx andando a vedere quanti sono gli elementi all'interno di df_temp (dopo aver aggiunto y)
# Assegno chosen_img_indx = ind - 1, poiché l'indice parte da 0.
ind = len(df_temp)
chosen_img_indx = ind - 1

#Per salvare il nuovo dataframe
#df_temp.to_csv(r'drive/My Drive/Dataset/fashion-dataset/embedding7.csv')

cosine_sim = linear_kernel(df_temp, df_temp)
indices = pd.Series(range(len(df_temp)), index=df_temp.index)
recom = get_recom(chosen_img_indx, df, cosine_sim)
recom_list = recom.to_list()'''


# Plot images

# Chosed image
#plt.title("Chosen image")
#plt.imshow(chosen_img)


#recommended images
'''plt.figure(figsize=(20,20))
j=0
for i in recom_list:
    plt.subplot(6, 10, j+1)
    print(i)
    #cloth_img =  mpimg.imread(path + 'images/'+ i)
    #plt.imshow(cloth_img)
    plt.axis("off")
    j+=1

plt.title("recommended images")
plt.subplots_adjust(wspace=-0.5, hspace=1)
plt.show()'''