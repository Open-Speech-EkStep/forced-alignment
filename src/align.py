from wav2vec2.utils import Wav2vec2
from configuration import ModelPath, Data

if __name__ == '__main__':
    
    obj = Wav2vec2(ModelPath.wav2vec2_path)
    word_segments = obj.merge_words(Data.wav_path, Data.txt_path)
    for word in word_segments:
        print(word)