from nemo.utils import model_utils
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.models.ctc_models import EncDecCTCModel
import os
from glob import glob
import torch
import nemo.collections.asr as nemo_asr
import scipy.io.wavfile as wav
import numpy as np
import ctc_segmentation as cs
from typing import List
from tqdm import tqdm
import math

class Conformer:
    def __init__(self, model_path):
        self.conformer_path = glob(model_path + '/*.nemo')[0]
        self.asr_model = self.load_model()
        self.tokenizer = self.load_tokenizer()
        self.vocabulary = self.load_vocabulary()
        self.config = cs.CtcSegmentationParameters()
        
    def load_model(self):
        model_cfg = ASRModel.restore_from(restore_path=self.conformer_path,
                                  return_config=True)
        classpath = model_cfg.target
        imported_class = model_utils.import_class_by_path(classpath)
        asr_model = imported_class.restore_from(restore_path=self.conformer_path)
        return asr_model
        
    def load_tokenizer(self):
        #True if model is BPE
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
        log_probs = self.asr_model.transcribe(paths2audio_files=[wav_path], batch_size=1, logprobs=True)[0]
        blank_col = log_probs[:, -1].reshape((log_probs.shape[0], 1))
        log_probs = np.concatenate((blank_col, log_probs[:, :-1]), axis=1)
        sample_rate, signal = wav.read(wav_path)
        index_duration = len(signal) / log_probs.shape[0] / sample_rate
        return log_probs, index_duration
    
    def prepare_tokenized_text_for_bpe_model(self, txt_path, blank_idx=0):
        with open(txt_path, "r") as f:
            text = f.read().splitlines()
        
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