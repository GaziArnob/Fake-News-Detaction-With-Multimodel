"""Run notebook Steps 9 and 13 outside Jupyter after CLIP features are complete."""

import json
import os
from pathlib import Path

os.environ['MPLBACKEND'] = 'Agg'
os.environ['HF_HUB_OFFLINE'] = '1'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

PROJECT_DIR = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_DIR)

FEATURES_CSV = Path('extracted_features.csv')
LABELS = ['real', 'fake']
SEED = 42


def calculate_metrics(model_name: str, actual, predicted) -> dict:
    return {
        'model': model_name,
        'accuracy': accuracy_score(actual, predicted),
        'precision_fake': precision_score(actual, predicted, pos_label='fake', zero_division=0),
        'recall_fake': recall_score(actual, predicted, pos_label='fake', zero_division=0),
        'f1_fake': f1_score(actual, predicted, pos_label='fake', zero_division=0),
    }


with Path('notebook.ipynb').open(encoding='utf-8') as handle:
    notebook = json.load(handle)

step_9_source = next(
    ''.join(cell['source'])
    for cell in notebook['cells']
    if cell['cell_type'] == 'code' and 'TEXT_MIN_DOCUMENT_FREQUENCY =' in ''.join(cell['source'])
)
step_13_source = next(
    ''.join(cell['source'])
    for cell in notebook['cells']
    if cell['cell_type'] == 'code' and 'MULTIMODAL_TEST_SIZE =' in ''.join(cell['source'])
)

exec(compile(step_9_source, 'notebook_step_9.py', 'exec'))
exec(compile(step_13_source, 'notebook_step_13.py', 'exec'))
