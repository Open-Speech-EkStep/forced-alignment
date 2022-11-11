from wav2vec2.utils import Wav2vec2
from conformer.utils import Conformer
from configuration import ModelPath, Data
from rich.console import Console
from rich.traceback import install
import srt
from pydub import AudioSegment
import numpy as np
import torch
from tqdm import tqdm
import json
import re
import string
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

install()
console = Console()

console.log(f"Audio path: [green underline]{Data.wav_path}")
console.log(f"Subtitle path: [green underline]{Data.srt_path}")


class SubtitleTimestamps:
    def __init__(self, srt_path, wav_path, language):
        self.srt_path = srt_path
        self.segments = self.read_subtitles()
        self.wav = AudioSegment.from_wav(wav_path)
        self.language = language
        self.factory = IndicNormalizerFactory()
        console.log(f"Subtitle path: {srt_path}")
        console.log(f"Audio path: {wav_path}")
        console.log(f"Language:  {language}")

    def read_subtitles(self):

        with open(self.srt_path, "r", encoding="utf-8") as f:
            subtitles = f.read()

        subs = list(srt.parse(subtitles))
        return subs

    def segment_start_end_times_seconds(self, segment):
        return segment.start.total_seconds(), segment.end.total_seconds()

    def clip_audio(self, start, end):
        return self.wav[start * 1000 : end * 1000]

    def filter_text(self, text):

        cleaned_text = re.sub("[%s]" % re.escape(string.punctuation + "।"), "", text)

        if self.language == "en":
            words = cleaned_text.split()
            new_text = " "
            for word in words:
                new_text += word.lower() + " "
            new_text = new_text.strip()
            return new_text

        else:
            normalizer = self.factory.get_normalizer(self.language, remove_nuktas=False)
            return normalizer.normalize(cleaned_text)

    def adjust_alignment(self, data):

        if self.language == "en":
            for d, k in data.items():
                words = k["text"].split()

                for i in range(len(words)):

                    old_key = list(k["timestamps"][i].keys())[0]

                    if old_key != words[i]:
                        k["timestamps"][i][words[i]] = k["timestamps"][i][old_key]
                        del k["timestamps"][i][old_key]
            return data

        else:
            return data


if __name__ == "__main__":
    language_codes = ModelPath.language_codes
    aligner_models = {}
    for language in language_codes:
        aligner_models[language] = Wav2vec2(
            ModelPath.wav2vec2_path, language_code=language, mode="tensor"
        )

    obj = SubtitleTimestamps(Data.srt_path, Data.wav_path, Data.language)
    subs = obj.read_subtitles()

    if Data.language not in language_codes:
        print("Specify the language code also while loading")
        exit

    d = {}

    for sub in tqdm(subs, leave=False):

        alignment = {}

        if sub.content == "[Music]":
            alignment["text"] = sub.content
            alignment["timestamps"] = None
            d[sub.index] = alignment
            continue

        start, end = obj.segment_start_end_times_seconds(sub)
        chunk = obj.clip_audio(start, end)
        float_wav = np.array(chunk.get_array_of_samples()).astype("float64")
        wav_tensor = torch.from_numpy(float_wav).float().view(1, -1)
        cleaned_text = obj.filter_text(sub.content)

        word_segments = aligner_models[Data.language].merge_words(
            wav_tensor, cleaned_text, begin=sub.start.total_seconds()
        )
        alignment["text"] = sub.content
        alignment["timestamps"] = word_segments

        d[sub.index] = alignment

    alignment = obj.adjust_alignment(d)

    with open("test.json", "w+", encoding="utf-8") as f:
        json.dump(alignment, f, indent=4)
