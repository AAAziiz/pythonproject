from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

# Définir la taille de l'entrée
input_shape = (64, 64, 3)

# Définir le modèle CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compiler le modèle
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# Diviser les données en ensembles d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Entraîner le modèle sur les données d'entraînement
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))


# Charger les données de test
test_dir = './eye.png'  # modifier le chemin d'accès au dossier contenant les images de test
test_labels = 'test_labels.txt'

with open(test_labels, 'r') as f:
    test_data = f.readlines()

X_test = []
y_test = []

for line in test_data:
    filename, label = line.strip().split(',')
    image = cv2.imread(os.path.join(test_dir, filename))
    image = cv2.resize(image, (64, 64))
    X_test.append(image)
    y_test.append(int(label))

X_test = np.array(X_test)
y_test = np.array(y_test)

# Évaluer le modèle sur les données de test
score = model.evaluate(X_test, y_test, batch_size=32)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

