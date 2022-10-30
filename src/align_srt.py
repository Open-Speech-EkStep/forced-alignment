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

install()
console = Console()

console.log(f"Audio path: [green underline]{Data.wav_path}")
console.log(f"Subtitle path: [green underline]{Data.srt_path}")


class SubtitleTimestamps:
    def __init__(self, srt_path, wav_path):
        self.srt_path = srt_path
        self.segments = self.read_subtitles()
        self.wav = AudioSegment.from_wav(wav_path)

    def read_subtitles(self):

        with open(self.srt_path, "r", encoding="utf-8") as f:
            subtitles = f.read()

        subs = list(srt.parse(subtitles))
        return subs

    def segment_start_end_times_seconds(self, segment):
        return segment.start.total_seconds(), segment.end.total_seconds()

    def clip_audio(self, start, end):
        return self.wav[start * 1000 : end * 1000]


if __name__ == "__main__":

    w2v_aligner = Wav2vec2(ModelPath.wav2vec2_path, mode="tensor")
    obj = SubtitleTimestamps(Data.srt_path, Data.wav_path)
    subs = obj.read_subtitles()

    d = {}

    for sub in tqdm(subs, leave=False):

        alignment = {}

        if sub.content == "[Music]":
            alignment["text"] = sub.content
            alignment["timestamps"] = None
            d[sub.index] = alignment
            continue

        s, e = obj.segment_start_end_times_seconds(sub)
        chunk = obj.clip_audio(s, e)
        float_wav = np.array(chunk.get_array_of_samples()).astype("float64")
        t = torch.from_numpy(float_wav).float().view(1, -1)
        word_segments = w2v_aligner.merge_words(
            t, sub.content, begin=sub.start.total_seconds()
        )
        alignment["text"] = sub.content
        alignment["timestamps"] = word_segments

        d[sub.index] = alignment
    with open("alignment.json", "w+", encoding="utf-8") as f:
        json.dump(d, f, indent=4)
