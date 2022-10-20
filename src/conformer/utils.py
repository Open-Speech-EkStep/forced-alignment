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
        