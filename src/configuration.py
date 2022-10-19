from dataclasses import dataclass

@dataclass(order=True)
class ModelPath:
    wav2vec2_path: str = '../models/wav2vec2/hindi'
    
@dataclass(order=True)
class Data:
    wav_path: str = '../examples/sample.wav'
    txt_path: str = '../examples/sample.txt'

