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
mkdir -p models/wav2vec2/hindi
wget -P models/wav2vec2/hindi https://storage.googleapis.com/test_public_bucket/alignment_models/wav2vec2/hindi/hi.pt
wget -P models/wav2vec2/hindi https://storage.googleapis.com/test_public_bucket/alignment_models/wav2vec2/hindi/dict.ltr.txt
```

## Usage
Specify model path, audio file and text file in `src/configuration.py` 
```
cd src
python align.py
```
