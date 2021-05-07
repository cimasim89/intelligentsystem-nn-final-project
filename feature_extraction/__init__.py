import glob
import pickle
from functools import partial

import librosa
import numpy as np
import soundfile
from pipetools import pipe

from utils import none_func, print_wrapper


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
