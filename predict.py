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

print(f"{Fore.CYAN}[Info] {Style.DIM}| {Fore.WHITE}Test Script")

model = tf.keras.models.load_model('model.keras')
print(f"{Fore.CYAN}[Info] {Style.DIM}| {Fore.WHITE}Model loaded successfully.")

def load_mp3_16k_mono(filename, amplification_factor=2):
    res = tfio.audio.AudioIOTensor(filename)
    tensor = res.to_tensor()
    tensor = tf.math.reduce_sum(tensor, axis=1) / 2
    tensor = tensor * amplification_factor
    sample_rate = res.rate
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(tensor, rate_in=sample_rate, rate_out=16000)
    return wav

mp3 = os.path.join('audio.mp3')
wav = load_mp3_16k_mono(mp3)
print(f"{Fore.CYAN}[Info] {Style.DIM}| {Fore.WHITE}Audio file loaded and resampled.")

audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=8000, sequence_stride=8000, batch_size=1)
print(f"{Fore.CYAN}[Info] {Style.DIM}| {Fore.WHITE}Number of audio slices: {len(audio_slices)}")

def preprocess_mp3(sample, index):
    sample = sample[0]
    zero_padding = tf.zeros([8000] - tf.shape(sample), dtype=tf.float32)
    wav = tf.concat([zero_padding, sample],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram

audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=8000, sequence_stride=8000, batch_size=1)
audio_slices = audio_slices.map(preprocess_mp3)
audio_slices = audio_slices.batch(32)

print(f"{Fore.CYAN}[Info] {Style.DIM}| {Fore.WHITE}Starting prediction on audio slices...")
yhat = model.predict(audio_slices)
print(f"{Fore.CYAN}[Info] {Style.DIM}| {Fore.WHITE}Prediction completed.")
print(f"{Fore.CYAN}[Info] {Style.DIM}| {Fore.WHITE}Predictions: {yhat}")

slice_size = 8000
sample_rate = 16000

for i, prediction in enumerate(yhat):
    if prediction > 0.9:
        time_in_seconds = (i * slice_size) / sample_rate
        print(f"{Fore.GREEN}[Sound Found] {Style.DIM}| {Fore.WHITE}Sound found at {time_in_seconds:.2f} seconds.")
