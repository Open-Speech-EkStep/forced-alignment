from wav2vec2.utils import Wav2vec2
from conformer.utils import Conformer
from configuration import ModelPath, Data
from rich.console import Console
from rich.traceback import install

install()
console = Console()
console.log(f"Audio path: [green underline]{Data.wav_path}")
console.log(f"Text path: [green underline]{Data.txt_path}")

if __name__ == "__main__":
    obj = Wav2vec2(ModelPath.wav2vec2_path)
    word_segments = obj.merge_words(Data.wav_path, Data.txt_path)
    for word in word_segments:
        print(word)
    obj = Conformer(ModelPath.confomer_path)
    alignments = obj.get_word_time_stamps(Data.txt_path, Data.wav_path)
    print(alignments)
