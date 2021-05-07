# intelligentsystem-nn-final-project

Sample audio file is downloadable 
from https://drive.google.com/file/d/1wWsrN2Ep7x6lWqOXfr4rpKGYrJhWc8z7/view

Note: feature and emotions extraction from dataset is currently based on RAVDESS dataset specification https://zenodo.org/record/1188976#.YJKPym5fhH4


# Execution
## Options

- `--audio-source-path` use wildcard notation to take audio files es.`--audio-source-path=./sample/Actor_01/*.wav`

- `--action=['extract','train','all'] default='all`
    - `extract` execute only feature extraction and store data into provided store filename `--storage-name`
    - `train` load feature from `--storage-name` and perform Network train and test
    - `all` do both preceding action
    
- `--storage-name` name of the feature storage file