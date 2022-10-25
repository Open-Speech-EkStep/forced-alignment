from nemo.utils import model_utils
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.models.ctc_models import EncDecCTCModel
from glob import glob
import nemo.collections.asr as nemo_asr
import scipy.io.wavfile as wav
import numpy as np
import ctc_segmentation as cs
from typing import List
from tqdm import tqdm
import math
from rich.console import Console
from rich.traceback import install

install()
console = Console()


class Conformer:
    def __init__(self, model_path):
        self.conformer_path = glob(model_path + "/*.nemo")[0]
        self.asr_model = self.load_model()
        self.tokenizer = self.load_tokenizer()
        self.vocabulary = self.load_vocabulary()
        self.config = cs.CtcSegmentationParameters()

    def load_model(self):
        model_cfg = ASRModel.restore_from(
            restore_path=self.conformer_path, return_config=True
        )
        classpath = model_cfg.target
        imported_class = model_utils.import_class_by_path(classpath)
        asr_model = imported_class.restore_from(restore_path=self.conformer_path)
        console.log(f":thumbs_up: Conformer model loaded successfully from {self.conformer_path}")
        return asr_model

    def load_tokenizer(self):
        # True if model is BPE
        bpe_model = isinstance(self.asr_model, nemo_asr.models.EncDecCTCModelBPE)
        if bpe_model:
            tokenizer = self.asr_model.tokenizer
        else:
            tokenizer = None
        return tokenizer

    def load_vocabulary(self):
        vocabulary = ["ε"] + list(self.asr_model.cfg.decoder.vocabulary)
        return vocabulary

    def get_log_probs(self, wav_path):
        log_probs = self.asr_model.transcribe(
            paths2audio_files=[wav_path], batch_size=1, logprobs=True
        )[0]
        blank_col = log_probs[:, -1].reshape((log_probs.shape[0], 1))
        log_probs = np.concatenate((blank_col, log_probs[:, :-1]), axis=1)
        sample_rate, signal = wav.read(wav_path)
        index_duration = len(signal) / log_probs.shape[0] / sample_rate
        return log_probs, index_duration

    def prepare_tokenized_text_for_bpe_model(self, text, blank_idx=0):
        """ Creates a transition matrix for BPE-based models"""
        space_idx = self.vocabulary.index("▁")
        ground_truth_mat = [[-1, -1]]
        utt_begin_indices = []
        for uttr in text:
            ground_truth_mat += [[blank_idx, space_idx]]
            utt_begin_indices.append(len(ground_truth_mat))
            token_ids = self.tokenizer.text_to_ids(uttr)
            # blank token is moved from the last to the first (0) position in the vocabulary
            token_ids = [idx + 1 for idx in token_ids]
            ground_truth_mat += [[t, -1] for t in token_ids]

        utt_begin_indices.append(len(ground_truth_mat))
        ground_truth_mat += [[blank_idx, space_idx]]
        ground_truth_mat = np.array(ground_truth_mat, np.int64)
        return ground_truth_mat, utt_begin_indices

    def determine_utterance_segments(
        self, utt_begin_indices, char_probs, timings, text, char_list
    ):
        """Utterance-wise alignments from char-wise alignments.
        Adapted from https://github.com/lumaku/ctc-segmentation
        """
        segments = []
        min_prob = np.float64(-10000000000.0)
        for i in tqdm(range(len(text))):
            start = self.compute_time(utt_begin_indices[i], "begin", timings)
            end = self.compute_time(utt_begin_indices[i + 1], "end", timings)

            start_t = start / self.config.index_duration_in_seconds
            start_t_floor = math.floor(start_t)

            # look for the left most blank symbol and split in the middle to fix start utterance segmentation
            if char_list[start_t_floor] == self.config.char_list[self.config.blank]:
                start_blank = None
                j = start_t_floor - 1
                while (
                    char_list[j] == self.config.char_list[self.config.blank]
                    and j > start_t_floor - 20
                ):
                    start_blank = j
                    j -= 1
                if start_blank:
                    start_t = int(
                        round(start_blank + (start_t_floor - start_blank) / 2)
                    )
                else:
                    start_t = start_t_floor
                start = start_t * self.config.index_duration_in_seconds

            else:
                start_t = int(round(start_t))

            end_t = int(round(end / self.config.index_duration_in_seconds))

            # Compute confidence score by using the min mean probability after splitting into segments of L frames
            n = self.config.score_min_mean_over_L
            if end_t <= start_t:
                min_avg = min_prob
            elif end_t - start_t <= n:
                min_avg = char_probs[start_t:end_t].mean()
            else:
                min_avg = np.float64(0.0)
                for t in range(start_t, end_t - n):
                    min_avg = min(min_avg, char_probs[t : t + n].mean())
            segments.append((start + 0.5, end + 0.05, min_avg))
        return segments

    def compute_time(self, index, align_type, timings):
        middle = (timings[index] + timings[index - 1]) / 2
        if align_type == "begin":
            return max(timings[index + 1] - 0.5, middle)
        elif align_type == "end":
            return min(timings[index - 1] + 0.5, middle)

    def get_word_time_stamps(self, txt_path, wav_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().splitlines()[0]

        words = text.split()
        log_probs, index_duration = self.get_log_probs(wav_path=wav_path)
        self.config.char_list = self.vocabulary
        self.config.min_window_size = 4000
        self.config.blank = 0
        self.config.index_duration = index_duration
        bpe_model = isinstance(self.asr_model, nemo_asr.models.EncDecCTCModelBPE)

        word_stamps = []

        for word in tqdm(words, leave=False):
            word_alignment = {}
            if bpe_model:
                (
                    ground_truth_mat,
                    utt_begin_indices,
                ) = self.prepare_tokenized_text_for_bpe_model([word], 0)
            else:
                raise Exception("ASR model is not using BPE tokenization")

            timings, char_probs, char_list = cs.ctc_segmentation(
                self.config, log_probs, ground_truth_mat
            )

            segment = self.determine_utterance_segments(
                utt_begin_indices, char_probs, timings, [word], char_list
            )

            word_alignment[word] = {"start": segment[0][0], "end": segment[0][1]}

            word_stamps.append(word_alignment)

        return word_stamps


if __name__ == "__main__":
    pass
