import glob
import pickle
from functools import partial

import click
import librosa
import numpy as np
import soundfile
from pipetools import pipe

import version


def none_func(value):
    return None


def print_wrapper(fn):
    def inner(value):
        fn(value)
        return value

    return inner


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
    return pipe | print_wrapper(lambda x: print("Parsing {0}".format(x))) | soundfile.SoundFile | extract_features(
        mfcc_required, chroma_required, mel_required)


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


def store_features(storage_name):
    def inner(data):
        with open(storage_name + '.pkl', 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            return data

    return inner


def load_features(storage_name):
    def inner(data):
        if data is None:
            with open(storage_name + '.pkl', 'rb') as f:
                return pickle.load(f)
        else:
            return data

    return inner


def elaborate_features(mfcc_required, chroma_required, mel_required, storage_name='./feature_dataset', active=True):
    if not active:
        return pipe | none_func
    return pipe \
           | get_files_in_path \
           | partial(map, elaborate_file(mfcc_required, chroma_required, mel_required)) \
           | list \
           | store_features(storage_name)


def get_features(mfcc_required, chroma_required, mel_required, storage_name='./feature_dataset', active=True):
    return pipe \
           | elaborate_features(mfcc_required=mfcc_required, chroma_required=chroma_required, mel_required=mel_required,
                                storage_name=storage_name, active=active) \
           | load_features(storage_name)


def parse_action(action):
    if action == "extract":
        return [True, False]
    if action == "train":
        return [False, True]
    if action == "all" or action is None:
        return [True, True]


@click.command()
@click.option('--audio-source-path', '-i', prompt='Give source audio files',
              help='Files to elaborate. Es. "./sample/Actor_01/*.wav"')
@click.option('--storage-name', '-s', prompt='Give feature storage filename',
              help='Where extracted features will be saved.')
@click.option('--action',
              type=click.Choice(['extract', 'train', 'all'], case_sensitive=False))
def emotion_classifier(audio_source_path, storage_name, action):
    [extraction_active, train_active] = parse_action(action)
    print("Starting...")
    print("Feature extraction: {0}".format(extraction_active))
    print("Network train: {0}".format(train_active))

    print(list(get_features(
        mfcc_required=True,
        chroma_required=True,
        mel_required=True,
        storage_name=storage_name,
        active=extraction_active)(
        audio_source_path)))
    # 3 - TODO divide feature dataset
    # 4 - TODO define NN
    # 5 - TODO train
    # 6 - TODO test


if __name__ == '__main__':
    print('<----------- {} ------------->'.format(version.__name__))
    print('<----------- Authors: {} ------------->'.format(version.__author__))
    print('<----------- Code Version {} ------------->'.format(version.__version__))
    emotion_classifier()
