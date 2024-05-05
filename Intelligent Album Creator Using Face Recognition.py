pip install keras
pip install tensorflow

import os
os.getcwd()

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

#initialize the model
model.Sequential()

#Add convolution layer
model.add(Convolution2D(32,(3,3),input_shape=(64,64,3),activation="relu"))

#Add pooling layer
model.add(Maxpooling2D(pool_size = (2,2)))

#Add Flattening Layer
model.add(Flatten())

#Add Hidden Layer
model.add(dense(init="uniform",activation="relu",output_dim=50))

#Add output Layer
model.add(Dense(init="uniform",activation="sigmoid",output_dim=1))

#Compile the model
model.compile(loss="binary_crossentropy",optimizer="sgd",metrics=["accuracy"])

from keras.preprocessing.image ImagedataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'C:\Users\SURYA\Desktop\dataset1\trainset',
    target_size=(64,64),
    batch_size=32,
    class_mode='binary')

print(train_generator.class_indices)

model.fit_generator(
    train_generator,
    steps_per_epoch=250,
    epochs=15,
    validation_data=validation_generator,
    validating_steps=60)

model.save("mymodel.h5")

model.compile(optimizer='sgd',loss="binary_crossentropy")

from skimage.transform import resize
def detect(frame):
  try:
      img = resize(frame,(64,64))
      img = np.expand_dims(img,axis=0)
      if(np.max(img)>1):
        img = img/255.0
      prediction = model.predict(img)
      print(prediction)
      prediction = model.predict_classes(img)
      print(prediction)
  except AttributeErrrr:
      print("Shape not found")

frame = cv2.imread("C:/Users/SURYA/Downloads/Photo.jpeg")
data = dataset(frame)

from = cv2.imread("C:\Users\SURYA\Downloads\Photo.jpeg")
data = detect(frame)

