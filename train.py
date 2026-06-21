import os
import soundfile as sf
import tensorflow as tf
import tensorflow_io as tfio
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from itertools import groupby
from colorama import Fore, Style, init

init(autoreset=True)

BREAK_FILE = os.path.join('Dataset', 'Positive', 'stone_break1.wav')
NOT_BREAK_FILE = os.path.join('Dataset', 'Negative', 'crit_ambient1.wav')

print(f"{Fore.CYAN}[Info] {Style.DIM}| {Fore.WHITE}Training Script")

def convert_to_16_bit(filename):
    def _convert(filename):
        data, samplerate = sf.read(filename.numpy())
        sf.write(filename.numpy(), data, samplerate, subtype='PCM_16')
    return tf.py_function(_convert, [filename], [])

def load_wav_16k_mono(filename):
    def _load_wav_16k_mono(filename):
        convert_to_16_bit(filename)
        file_contents = tf.io.read_file(filename.numpy())
        wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
        wav = tf.squeeze(wav, axis=-1)
        sample_rate = tf.cast(sample_rate, dtype=tf.int64)
        wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
        return wav
    return tf.py_function(_load_wav_16k_mono, [filename], tf.float32)

POS = os.path.join('Dataset', 'Positive')
NEG = os.path.join('Dataset', 'Negative')

pos = tf.data.Dataset.list_files(POS + '/*.wav')
neg = tf.data.Dataset.list_files(NEG + '/*.wav')

positives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos)))))
negatives = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg)))))
data = positives.concatenate(negatives)

lengths = []
for file in os.listdir(os.path.join('Dataset', 'Positive')):
    tensor_wave = load_wav_16k_mono(os.path.join('Dataset', 'Positive', file))
    lengths.append(len(tensor_wave))

print(f"{Fore.CYAN}[Info] {Style.DIM}| {Fore.WHITE}Mean: {tf.math.reduce_mean(lengths).numpy()}")
print(f"{Fore.CYAN}[Info] {Style.DIM}| {Fore.WHITE}Max: {tf.math.reduce_max(lengths).numpy()}")
print(f"{Fore.CYAN}[Info] {Style.DIM}| {Fore.WHITE}Min: {tf.math.reduce_min(lengths).numpy()}")

def preprocess(file_path, label):
    wav = load_wav_16k_mono(file_path)
    wav = wav[:8000]
    zero_padding = tf.zeros([8000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], 0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram, label

filepath, label = positives.shuffle(buffer_size=10000).as_numpy_iterator().next()
spectrogram, label = preprocess(filepath, label)

data = data.map(preprocess)
data = data.cache()
data = data.shuffle(buffer_size=1000)
data = data.batch(16)
data = data.prefetch(8)

train = data.take(60)
test = data.skip(60).take(26)

samples, labels = train.as_numpy_iterator().next()
print(f"{Fore.CYAN}[Info] {Style.DIM}| {Fore.WHITE}Training samples shape: {samples.shape}")

model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(241, 257, 1)))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='Adam', loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
print(f"{Fore.CYAN}[Info] {Style.DIM}| {Fore.WHITE}Model Summary:")
print(model.summary())

print(f"{Fore.CYAN}[Info] {Style.DIM}| {Fore.WHITE}Starting model training...")
hist = model.fit(train, epochs=200, validation_data=test)
print(f"{Fore.CYAN}[Info] {Style.DIM}| {Fore.WHITE}Model training completed.")

model.save('model.keras')
print(f"{Fore.YELLOW}[Model] {Style.DIM}| {Fore.WHITE}Model saved as 'model.keras'.")
