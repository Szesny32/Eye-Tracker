
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D 
import numpy as np
from sklearn.model_selection import train_test_split
import os
import cv2 as cv


def load_data(data_directory):
    # Listy przechowujące obrazy i etykiety
    samples = []
    labels = []
    
    # Przechodzenie przez pliki w katalogu
    for filename in os.listdir(data_directory):
        if filename.endswith('.jpg'):
            # Pobranie etykiet z nazwy pliku
            parts = filename[:-4].split('_')
            left_x, left_y, right_x, right_y = map(float, parts[1:])
            
            # Wczytanie obrazu
            image_path = os.path.join(data_directory, filename)
            image = cv.imread(image_path)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  
            # Normalizacja obrazu (opcjonalne)
            image = image / 255.0
            
            # Dodanie obrazu i etykiety do list
            samples.append(image)
            labels.append([left_x, left_y, right_x, right_y])
    
    # Konwersja list na tablice numpy
    samples = np.array(samples)
    labels = np.array(labels)
    print(len(samples), labels[0])
    return samples, labels


input_shape = (256, 256, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape = input_shape, activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(4, activation='linear'))


model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])


samples, labels = load_data('dataset')
(trainSamples, testSamples, trainLabels, testLabels) = train_test_split(samples, labels, random_state=32, train_size = 0.75)

H = model.fit(trainSamples, trainLabels, epochs=10, validation_data=(testSamples, testLabels))

save_model(model, 'model.h5')

