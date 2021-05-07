import click

import version
from feature_extraction import get_features


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
