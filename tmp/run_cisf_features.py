"""Rebuild the notebook's CISF export, including its fused PCA components."""

import json
import os
from pathlib import Path

os.environ['MPLBACKEND'] = 'Agg'
os.environ['HF_HUB_OFFLINE'] = '1'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_DIR)

FEATURES_CSV = Path('extracted_features.csv')
LABELS = ['real', 'fake']
SEED = 42
DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)

with Path('notebook.ipynb').open(encoding='utf-8') as handle:
    notebook = json.load(handle)

step_8_source = next(
    ''.join(cell['source'])
    for cell in notebook['cells']
    if cell['cell_type'] == 'code' and 'CISF_NEIGHBORS =' in ''.join(cell['source'])
)

exec(compile(step_8_source, 'notebook_step_8.py', 'exec'))
