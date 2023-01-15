from keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
import cv2

# NumPy for numerical computing
import numpy as np
import scipy
# Pandas for DataFrames
import pandas as pd
from keras.saving.legacy.model_config import model_from_json

# Matplotlib for visualization
from matplotlib import pyplot as plt

# import color maps
from matplotlib.colors import ListedColormap

# Seaborn for easier visualization
import seaborn as sns

df = pd.read_csv(
    "C:/Users/Filip Iasinovschi/.kaggle/challenges-in-representation-learning-facial-expression-recognition-challenge"
    "/fer2013/fer2013.csv")
# print(df)

image_size = (48, 48)
pixels = df['pixels'].tolist()  # Converting the relevant column element into a list for each row
width, height = 48, 48
faces = []

for pixel_sequence in pixels:
    face = [int(pixel) for pixel in pixel_sequence.split(' ')]  # Splitting the string by space character as a list
    face = np.asarray(face).reshape(width, height)  # converting the list to numpy array in size of 48*48
    face = cv2.resize(face.astype('uint8'), image_size)  # resize the image to have 48 cols (width) and 48 rows (height)
    faces.append(face.astype('float32'))  # makes the list of each images of 48*48 and their pixels in numpyarray form

faces = np.asarray(faces)  # converting the list into numpy array
faces = np.expand_dims(faces, -1)  # Expand the shape of an array -1=last dimension => means color space
emotions = pd.get_dummies(df['emotion']).to_numpy()  # doing the one hot encoding type on emotions

x = faces.astype('float32')
x = x / 255.0  # Dividing the pixels by 255 for normalization  => range(0,1)

# Scaling the pixels value in range(-1,1)
x = x - 0.5
x = x * 2.0

num_samples, num_classes = emotions.shape

num_samples = len(x)
num_train_samples = int((1 - 0.2) * num_samples)

# Traning data
train_x = x[:num_train_samples]
train_y = emotions[:num_train_samples]

# Validation data
val_x = x[num_train_samples:]
val_y = emotions[num_train_samples:]

train_data = (train_x, train_y)
val_data = (val_x, val_y)

""" Building up Model Architecture """

input_shape = (48, 48, 1)
num_classes = 7

model = Sequential()
model.add(Convolution2D(filters=16, kernel_size=(7, 7), padding='same',
                        name='image_array', input_shape=input_shape))
model.add(BatchNormalization())
model.add(Convolution2D(filters=16, kernel_size=(7, 7), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(.5))

model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
model.add(BatchNormalization())
model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(.5))

model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(.5))

model.add(Convolution2D(filters=128, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Convolution2D(filters=128, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(.5))

model.add(Convolution2D(filters=256, kernel_size=(3, 3), padding='same'))

model.add(BatchNormalization())
model.add(Convolution2D(filters=num_classes, kernel_size=(3, 3), padding='same'))
model.add(GlobalAveragePooling2D())
model.add(Activation('softmax', name='predictions'))

# print(model.summary())

"""Data Augmentation => taking the batch and apply some series of random transformations (random rotation, resizing, 
shearing) ===> to increase generalizability of model """

# data generator Generate batches of tensor image data with real-time data augmentation
data_generator = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=.1,
    horizontal_flip=True)

# model parameters/compilation

""" CONFIGURATION ==>.compile(optimizer, loss , metrics) """

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# print(model.summary())

# parameters
batch_size = 32  # Number of samples per gradient update
num_epochs = 200  # Number of epochs to train the model.
# input_shape = (64, 64, 1)
verbose = 1  # per epohs  progress bar
num_classes = 7
patience = 50
datasets = ['fer2013']
num_epochs = 200
base_path = "/content"
for dataset_name in datasets:
    print('Training dataset:', dataset_name)

    # callbacks
    log_file_path = dataset_name + '_emotion_training.log'

    csv_logger = CSVLogger(log_file_path, append=False)
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience / 4), verbose=1)

    trained_models_path = base_path + dataset_name + 'simple_cnn'
    model_names = trained_models_path + '.{epoch:02d}-{val_loss:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1, save_best_only=True)
    my_callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

    # loading dataset
    train_faces, train_emotions = train_data
    history = model.fit(data_generator.flow(train_faces, train_emotions,
                                            batch_size),
                        epochs=num_epochs, verbose=1
                        , callbacks=my_callbacks,
                        validation_data=val_data)

# evaluate() returns [loss,acc]
score = model.evaluate(val_x, val_y, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1] * 100)

""" metrics collected by history object """
history_dict = history.history
history_dict.keys()

# print(history_dict["accuracy"])

""" Visualising model training history """

train_loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(history_dict['accuracy']) + 1)

plt.plot(epochs, train_loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

""" Visualising how the accuracy increases as the number of epochs increases"""
train_acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
plt.plot(epochs, train_acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# added emojis to the results
emotion_dict = {0: "Neutral", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Sad", 5: "Surprised", 6: "Neutral"}
# emojis unicodes
emojis = {0: "\U0001f620", 1: "\U0001f922", 2: "\U0001f628", 3: "\U0001f60A", 4: "\U0001f625", 5: "\U0001f632",
          6: "\U0001f610"}


def _predict(path):
    facecasc = cv2.CascadeClassifier('/content/haarcascade_frontalface_default.xml')
    imagePath = '/content/' + path
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10)
    print("No of faces : ", len(faces))
    i = 1
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]  # croping
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)

        maxindex = int(np.argmax(prediction))
        print("person ", i, " : ", emotion_dict[maxindex], "-->", emojis[maxindex])
        cv2.putText(image, emotion_dict[maxindex], (x + 10, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # if text is not apeared , change coordinates. it may work

    cv2.imshow(image)


# saving weights
model.save_weights("model.h5")

# saving architecture
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)


# serialize weights to HDF5
model.save_weights("model.h5")

def load_model_():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model.h5")
    return model


model = load_model_()
