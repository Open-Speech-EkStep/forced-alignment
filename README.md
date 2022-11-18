## Installation Instructions 

```
conda create -n alignment python=3.8
conda activate alignment
pip install -r requirements.txt 
git clone https://github.com/harveenchadha/fairseq
cd fairseq
```
Change line $477$ of file `fairseq/dataclass/util.py` from `def merge_with_parent(dc: FairseqDataclass, cfg: DictConfig, remove_missing=False)` to `def merge_with_parent(dc: FairseqDataclass, cfg: DictConfig, remove_missing=True)`

`pip install -e .`

## Download models
```
mkdir -p models/wav2vec2/
wget -P models/wav2vec2/ https://storage.googleapis.com/test_public_bucket/aligner_models.zip
cd models/wav2vec2 
unzip aligner_models.zip
```

## Usage

Currently the following languages are supported.
```
English - en
Hindi - hi
Bengali - bn
Gujarati - gu
Kannada - kn
Malayalam - ml
Marathi - mr
Oriya - or
Punjabi - pa
Sanskrit - sa
Tamil - ta
Telugu - te
Urdu - ur
```

Specify model path, audio file and text file in `src/configuration.py` 

```{python}
from dataclasses import dataclass


@dataclass(order=True)
class ModelPath:
    wav2vec2_path: str = "../models/wav2vec2/aligner_models"
    language_codes = ['en', 'hi'] # Languages to be loaded. Add the corresponding language code
    confomer_path: str = "../models/nemo/eng"


@dataclass(order=True)
class Data:
    wav_path: str = "audio/english.wav"
    txt_path: str = "../examples/sample.txt"
    srt_path: str = 'srt/English_corrected.srt'
    language: str = 'en' # language of subtitle
```

```
cd src
python align.py
```
