import os
import sys
import math
import contextlib
import io
import tensorflow as tf
import tensorflow_io as tfio
from pymiere import wrappers
import pymiere
from itertools import groupby
from colorama import Fore, Style, init
import logging

init(autoreset=True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
clip_ids = []

def time_from_seconds(seconds):
    t = pymiere.Time()
    t.seconds = seconds
    return t

def timecode_from_time(time, sequence):
    return time.getFormatted(sequence.getSettings().videoFrameRate, sequence.getSettings().videoDisplayFormat)

def timecode_from_seconds(seconds, sequence):
    return timecode_from_time(time_from_seconds(seconds), sequence)

def selected_clip():
    sequence = pymiere.objects.app.project.activeSequence
    if sequence:
        for track in sequence.videoTracks:
            for clip in track.clips:
                if clip.isSelected():
                    return clip
    return None

def render_audio_of_selected_clip():
    clip = selected_clip()
    if clip:
        sequence = pymiere.objects.app.project.activeSequence
        if sequence:

            sequence.setInPoint(clip.start)
            sequence.setOutPoint(clip.end)

            output_path = r"C:\Users\fastd\Documents\Projekt\audio.mp3"
            preset_path = r"C:\Program Files\Adobe\Adobe Premiere Pro 2024\Settings\EncoderPresets\MP3_mono_96kbps_nometadata.epr"

            result = sequence.exportAsMediaDirect(
                output_path,
                preset_path,
                pymiere.objects.app.encoder.ENCODE_IN_TO_OUT,
            )

            print(f"{Fore.GREEN}[Render] {Style.DIM}| {Fore.WHITE}{result.strip() if result else 'No error!'}")
        else:
            print(f"{Fore.RED}[Error] {Style.DIM}| {Fore.WHITE}No active sequence found.")
    else:
        print(f"{Fore.RED}[Error] {Style.DIM}| {Fore.WHITE}No clip selected.")

def list_video(sequence, start_time, end_time):
    for track in sequence.videoTracks:
        for clip in track.clips:
            if any([
                math.floor(clip.start.seconds) == math.floor(start_time) and math.floor(clip.end.seconds) == math.floor(end_time),
                math.ceil(clip.start.seconds) == math.ceil(start_time) and math.ceil(clip.end.seconds) == math.ceil(end_time),
                math.floor(clip.start.seconds) == math.floor(start_time) and math.ceil(clip.end.seconds) == math.ceil(end_time),
                math.ceil(clip.start.seconds) == math.ceil(start_time) and math.floor(clip.end.seconds) == math.floor(end_time)
            ]):
                clip_ids.append(clip.nodeId)

    for track in sequence.audioTracks:
        for clip in track.clips:
            if any([
                math.floor(clip.start.seconds) == math.floor(start_time) and math.floor(clip.end.seconds) == math.floor(end_time),
                math.ceil(clip.start.seconds) == math.ceil(start_time) and math.ceil(clip.end.seconds) == math.ceil(end_time),
                math.floor(clip.start.seconds) == math.floor(start_time) and math.ceil(clip.end.seconds) == math.ceil(end_time),
                math.ceil(clip.start.seconds) == math.ceil(start_time) and math.floor(clip.end.seconds) == math.floor(end_time)
            ]):
                clip_ids.append(clip.nodeId)

def cut_selected_clip(cut_time):
    clip = selected_clip()
    if not clip:
        print(f"{Fore.RED}[Error] {Style.DIM}| {Fore.WHITE}No clip selected.")
        return

    seq = pymiere.objects.app.project.activeSequence
    clip_start = clip.start.seconds + cut_time
    clip_add = clip.start.seconds + cut_time + 0.5

    clip_start_timecode = timecode_from_seconds(clip_start, seq)
    clip_add_timecode = timecode_from_seconds(clip_add, seq)

    print(f"{Fore.YELLOW}[Cut] {Style.DIM}| {Fore.WHITE}Clip start timecode: {clip_start_timecode}")
    print(f"{Fore.YELLOW}[Cut] {Style.DIM}| {Fore.WHITE}Clip add timecode: {clip_add_timecode}")

    video_track = pymiere.objects.qe.project.getActiveSequence().getVideoTrackAt(0)
    video_track.razor(clip_start_timecode)
    video_track.razor(clip_add_timecode)

    audio_track = pymiere.objects.qe.project.getActiveSequence().getAudioTrackAt(0)
    audio_track.razor(clip_start_timecode)
    audio_track.razor(clip_add_timecode)

    list_video(seq, clip_start, clip_add)

def delete_other_clips(clip_begin, clip_end):
    seq = pymiere.objects.app.project.activeSequence
    for track in seq.videoTracks:
        for clip in track.clips:
            if clip.nodeId not in clip_ids and clip.start.seconds > clip_begin and clip.end.seconds <= clip_end:
                clip.remove(True, True)

    for track in seq.audioTracks:
        for clip in track.clips:
            if clip.nodeId not in clip_ids and clip.start.seconds > clip_begin and clip.end.seconds <= clip_end:
                clip.remove(True, True)

def load_mp3_16k_mono(filename, amplification_factor=2):
    res = tfio.audio.AudioIOTensor(filename)
    tensor = res.to_tensor()
    tensor = tf.math.reduce_sum(tensor, axis=1) / 2
    tensor = tensor * amplification_factor
    sample_rate = tf.cast(res.rate, dtype=tf.int64)
    wav = tfio.audio.resample(tensor, rate_in=sample_rate, rate_out=16000)
    return wav

def preprocess_mp3(sample, index):
    sample = sample[0]
    zero_padding = tf.zeros([8000] - tf.shape(sample), dtype=tf.float32)
    wav = tf.concat([zero_padding, sample], 0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram

def main():
    render_audio_of_selected_clip()
    clip = selected_clip()
    clip_begin = clip.start.seconds if clip else 0
    clip_end = clip.end.seconds if clip else 0

    model = tf.keras.models.load_model('model.keras')

    mp3 = os.path.join('audio.mp3')
    wav = load_mp3_16k_mono(mp3)
    audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=8000, sequence_stride=8000, batch_size=1)

    print(f"{Fore.CYAN}[Info] {Style.DIM}| {Fore.WHITE}Number of audio slices: {len(audio_slices)}")

    audio_slices = audio_slices.map(preprocess_mp3)
    audio_slices = audio_slices.batch(32)

    slice_size = 8000
    sample_rate = 16000

    yhat = model.predict(audio_slices)
    cut_times = [i * slice_size / sample_rate for i, prediction in enumerate(yhat) if prediction > 0.9]

    for time_in_seconds in cut_times:
        cut_selected_clip(time_in_seconds)

    delete_other_clips(clip_begin, clip_end)
    sys.exit(0)

if __name__ == "__main__":
    with contextlib.redirect_stderr(io.StringIO()):
        main()
