# https://medium.com/@oluyaled/audio-classification-using-deep-learning-and-tensorflow-a-scontep-by-step-guide-5327467ee9ab
import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.image import resize
from tensorflow.keras.models import load_model

# get the current working directory
current_working_directory = os.getcwd()

# print output to the console
print(current_working_directory)

# Define your folder structure
data_dir = 'training_data'
classes = ['autopilot-engaged', 'autopilot-disengage','lane-change-confirmation','navigate-on-autopilot-disengage','navigate-on-autopilot-engaged']

# Load and preprocess audio data
def load_and_preprocess_data(data_dir, classes, target_shape=(128, 128)):
  data = []
  labels = []

  for i, class_name in enumerate(classes):
    class_dir = os.path.join(data_dir, class_name)
    for filename in os.listdir(class_dir):
      if filename.endswith('.wav'):
        file_path = os.path.join(class_dir, filename)
        audio_data, sample_rate = librosa.load(file_path, sr=None)
        # Perform preprocessing (e.g., convert to Mel spectrogram and resize)
        mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)
        labels.append(i)

  return np.array(data), np.array(labels)

# Split data into training and testing sets
data, labels = load_and_preprocess_data(data_dir, classes)
labels = to_categorical(labels, num_classes=len(classes))  # Convert labels to one-hot encoding
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Create a neural network model
input_shape = X_train[0].shape
input_layer = Input(shape=input_shape)
x = Conv2D(32, (3, 3), activation='relu')(input_layer)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
output_layer = Dense(len(classes), activation='softmax')(x)
model = Model(input_layer, output_layer)


# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


# Train the model
model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test))


test_accuracy=model.evaluate(X_test,y_test,verbose=0)
print(test_accuracy[1])

# Save the model
model.save('model/audio_classification_model.keras')