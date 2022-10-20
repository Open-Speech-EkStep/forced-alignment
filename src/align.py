from wav2vec2.utils import Wav2vec2
from conformer.utils import Conformer
from configuration import ModelPath, Data

if __name__ == "__main__":
    """
    obj = Wav2vec2(ModelPath.wav2vec2_path)
    word_segments = obj.merge_words(Data.wav_path, Data.txt_path)
    for word in word_segments:
        print(word)
    """

    obj = Conformer(ModelPath.confomer_path)
    print(obj.conformer_path)
    print(obj.get_log_probs(Data.wav_path))
    print(obj.prepare_tokenized_text_for_bpe_model(Data.txt_path))
