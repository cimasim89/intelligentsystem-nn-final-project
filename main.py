import click
import glob
import librosa
import soundfile
import version
import numpy as np
from pipetools import pipe
from functools import partial


# 1 - Extract feature
def load_sound_file(func):
    return func


def extract_features(mfcc_required, chroma_required, mel_required):
    def inner(sound_file):
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        result = np.array([])
        if mfcc_required:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma_required:
            stft = np.abs(librosa.stft(X))
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel_required:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
        return result
    return inner


def extract_feature_from_file(mfcc_required, chroma_required, mel_required):
    return pipe | soundfile.SoundFile | extract_features(mfcc_required, chroma_required, mel_required)

def extract_emotion_from_file(file):
    return file.split("-")[2]


def elaborate_file(mfcc_required, chroma_required, mel_required):
    def inner(file):
        return {
            "features": extract_feature_from_file(mfcc_required, chroma_required, mel_required)(file),
            "emotion": extract_emotion_from_file(file)
        }
    return inner


def get_files_in_path(path_name):
    return glob.glob(path_name)


def get_features(mfcc_required, chroma_required, mel_required):
    return pipe | get_files_in_path | partial(map, elaborate_file(mfcc_required, chroma_required, mel_required))


@click.command()
@click.option('--audio-source-path', '-i', prompt='Give source audio files',
              help='Files to elaborate. Es. "./sample/Actor_01/*.wav"')
def emotion_classifier(audio_source_path):
    # 1 - extract features './sample/Actor_01/*.wav'
    print(list(get_features(True, True, True)(audio_source_path)))
    # 2 - TODO divide dataset
    # 3 - TODO define NN
    # 4 - TODO train
    # 5 - TODO test


if __name__ == '__main__':
    print('<----------- {} ------------->'.format(version.__name__))
    print('<----------- Authors: {} ------------->'.format(version.__author__))
    print('<----------- Code Version {} ------------->'.format(version.__version__))
    emotion_classifier()
