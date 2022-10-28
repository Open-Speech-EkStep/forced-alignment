from dataclasses import dataclass


@dataclass(order=True)
class ModelPath:
    wav2vec2_path: str = "../models/wav2vec2/eng"
    confomer_path: str = "../models/nemo/eng"


@dataclass(order=True)
class Data:
    wav_path: str = "/home/anirudh/Downloads/eng.wav"
    txt_path: str = "/home/anirudh/Downloads/eng.txt"
    srt_path: str = '/home/anirudh/Downloads/eng.srt'
