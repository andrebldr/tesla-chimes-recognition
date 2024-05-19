import librosa
import numpy as np
import tensorflow as tf
from tensorflow.image import resize
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('model/audio_classification_model.keras')

# Define the target shape for input spectrograms
target_shape = (128, 128)

# Define your class labels
classes = ['autopilot-engaged', 'autopilot-disengage','lane-change-confirmation','navigate-on-autopilot-disengage','navigate-on-autopilot-engaged']

# Function to preprocess and classify an audio file
def test_audio(file_path, model):
  # Load and preprocess the audio file
  audio_data, sample_rate = librosa.load(file_path, sr=None)
  mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
  mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
  mel_spectrogram = tf.reshape(mel_spectrogram, (1,) + target_shape + (1,))

  # Make predictions
  predictions = model.predict(mel_spectrogram)

  # Get the class probabilities
  class_probabilities = predictions[0]

  # Get the predicted class index
  predicted_class_index = np.argmax(class_probabilities)

  return class_probabilities, predicted_class_index

# Test an audio file
test_audio_file = 'training_data/autopilot-engaged/Autopilot Engage.wav'
class_probabilities, predicted_class_index = test_audio(test_audio_file, model)

# Display results for all classes
for i, class_label in enumerate(classes):
  probability = class_probabilities[i]
  print(f'Class: {class_label}, Probability: {probability:.4f}')

# Calculate and display the predicted class and accuracy
predicted_class = classes[predicted_class_index]
accuracy = class_probabilities[predicted_class_index]
print(f'The audio is classified as: {predicted_class}')
print(f'Accuracy: {accuracy:.4f}')