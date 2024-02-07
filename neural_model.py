
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D 
import numpy as np
from sklearn.model_selection import train_test_split
import os
import cv2 as cv
import matplotlib.pyplot as plt

def load_data(data_directory):
    # Listy przechowujące obrazy i etykiety
    samples = []
    labels = []
    names = []
    
    # Przechodzenie przez pliki w katalogu
    for root, dirs, files in os.walk(data_directory):
        for filename in files:
            if filename.endswith('.jpg'):
                # Pobranie etykiet z nazwy pliku
                parts = filename[:-4].split('_')
                left_x, left_y, right_x, right_y = map(float, parts[1:])
                
                # Wczytanie obrazu
                image_path = os.path.join(root, filename)
                image = cv.imread(image_path)
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  
                # Normalizacja obrazu (opcjonalne)
                image = image / 255.0
                
                # Dodanie obrazu i etykiety do list
                samples.append(image)
                labels.append([left_x, left_y, right_x, right_y])
                names.append(image_path)
    
    # Konwersja list na tablice numpy
    samples = np.array(samples)
    labels = np.array(labels)
    print(len(samples), labels[0])
    return samples, labels, names


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


model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])


samples, labels, _ = load_data('dataset')
print(samples.shape)
(trainSamples, testSamples, trainLabels, testLabels) = train_test_split(samples, labels, random_state=32, train_size = 0.75)

H = model.fit(trainSamples, trainLabels, epochs=30, validation_data=(testSamples, testLabels))

# Otrzymaj historię treningu
history = H.history

# Pobierz wartości błędu dla danych treningowych i walidacyjnych
mse = history['mean_squared_error']
val_mse = history['val_mean_squared_error']

# Utwórz wykresy
epochs = range(1, len(mse) + 1)

# Wykres funkcji straty
plt.plot(epochs, mse, 'bo', label='Training MSE')
plt.plot(epochs, val_mse, 'b', label='Validation MSE')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
save_model(model, 'model.h5')


