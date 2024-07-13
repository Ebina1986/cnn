from fastapi import FastAPI , File , UploadFile
from fastapi.responses import HTMLResponse
import keras as ks
import numpy as np
import io
from PIL import Image
import random
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')




(train_images, train_labels), (test_images, test_labels) =ks.datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
model = ks.models.load_model('my_model.keras')


def create_model():
    model = ks.models.Sequential()
    model.add(ks.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(ks.layers.MaxPooling2D((2, 2)))
    model.add(ks.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(ks.layers.MaxPooling2D((2, 2)))
    model.add(ks.layers.Conv2D(64, (3, 3), activation='relu'))
    model.summary()

    model.add(ks.layers.Flatten())
    model.add(ks.layers.Dense(64, activation='relu'))
    model.add(ks.layers.Dense(10))

    model.summary()

    model.compile(optimizer='adam',loss=ks.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=10)
    model.save('my_model.keras')

def predict_image(img):
    img = Image.open(io.BytesIO(img))
    img = img.resize((32, 32))
    img_array = ks.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    return img_array



app = FastAPI()


@app.get("/" , response_class= HTMLResponse)
async def root():

    return "Hello! Go to /predicting/ for predict an image"


@app.post("/")
async def predicting(file: UploadFile = File(...)):
    # خواندن فایل تصویری بارگذاری‌شده
    contents = await file.read()
    # پردازش تصویر
    img_array = predict_image(contents)
    # انجام پیش‌بینی
    predictions = model.predict(img_array)
    score = np.argmax(predictions[0])
    # دریافت برچسب
    predicted_label = class_names[score]
    return {"label": predicted_label}


@app.post("/predicting/")
async def predicting(file: UploadFile = File(...)):
    # خواندن فایل تصویری بارگذاری‌شده
    contents = await file.read()
    # پردازش تصویر
    img_array = predict_image(contents)
    # انجام پیش‌بینی
    predictions = model.predict(img_array)
    score = np.argmax(predictions[0])
    # دریافت برچسب
    predicted_label = class_names[score]
    return {"label": predicted_label}
