import matplotlib.pyplot as plt

import numpy as np
import os
import cv2 as cv


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


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


samples, labels, names = load_data('dataset/s4')


for chunk_samples, chunk_labels, chunk_names in zip(chunks(samples, 25), chunks(labels, 25), chunks(names, 25)):
    fig, axes = plt.subplots(5, 5, figsize=(16, 16))

    for i, (sample, label, name) in enumerate(zip(chunk_samples, chunk_labels, chunk_names)):
        row = i // 5
        col = i % 5

        axes[row, col].imshow(sample)

        # Zaznaczanie rzeczywistych punktów na obrazie
        axes[row, col].scatter(label[0], label[1], c='green', marker='x' )
        axes[row, col].scatter(label[2], label[3], c='orange', marker='x')

        axes[row, col].set_title(f'{name}', fontsize=6)
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()