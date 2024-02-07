import matplotlib.pyplot as plt
import random


from tensorflow.keras.models import Sequential, load_model
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

samples, labels = load_data('dataset')
(trainSamples, testSamples, trainLabels, testLabels) = train_test_split(samples, labels, random_state=32, train_size = 0.75)

random_indices = random.sample(range(len(testSamples)), 16)  # Zmiana na 16, aby wyświetlić 4 kolumny i 4 rzędy

model = load_model('model.h5')
# Przewidywanie punktów na wybranych obrazkach
predicted_points = model.predict(testSamples[random_indices])

fig, axes = plt.subplots(4, 4, figsize=(16, 16))

for i, idx in enumerate(random_indices):
    row = i // 4
    col = i % 4
    
    axes[row, col].imshow(testSamples[idx])
    
    # Zaznaczanie przewidzianych punktów na obrazie
    axes[row, col].scatter(predicted_points[i][0], predicted_points[i][1], c='red', marker='o', label='Predicted Left Point')
    axes[row, col].scatter(predicted_points[i][2], predicted_points[i][3], c='blue', marker='o', label='Predicted Right Point')
    
    # Zaznaczanie rzeczywistych punktów na obrazie
    axes[row, col].scatter(testLabels[idx][0], testLabels[idx][1], c='green', marker='x', label='Actual Left Point')
    axes[row, col].scatter(testLabels[idx][2], testLabels[idx][3], c='orange', marker='x', label='Actual Right Point')
    
    axes[row, col].set_title(f'Image {idx+1}')
    axes[row, col].legend()
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()