"""Run notebook Step 12 outside the Jupyter UI while keeping its CSV checkpoints."""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_DIR)
os.environ['HF_HUB_OFFLINE'] = '1'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FEATURES_CSV = Path('extracted_features.csv')
LABELS = ['real', 'fake']

with Path('notebook.ipynb').open(encoding='utf-8') as handle:
    notebook = json.load(handle)

step_12_source = next(
    ''.join(cell['source'])
    for cell in notebook['cells']
    if cell['cell_type'] == 'code' and 'CLIP_MODEL_ID =' in ''.join(cell['source'])
)

exec(compile(step_12_source, 'notebook_step_12.py', 'exec'))
