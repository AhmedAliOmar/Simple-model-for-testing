
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# Generate a simple dataset
data = np.random.random((1000, 20))
labels = np.random.randint(2, size=(1000, 1))

# Convert labels to categorical one-hot encoding
one_hot_labels = to_categorical(labels, num_classes=2)

# Define a simple Sequential model
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=20))
model.add(Dense(units=2, activation='softmax'))

# Compile the model
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(data, one_hot_labels, epochs=10, batch_size=32, verbose=1)

# Save the model to HDF5 file
model.save('simple_model.h5')
