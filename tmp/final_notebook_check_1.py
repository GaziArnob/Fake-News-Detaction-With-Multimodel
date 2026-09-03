from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

MULTIMODAL_TEST_SIZE = 0.20
TUNING_CV_FOLDS = 3
TUNING_ITERATIONS = 8
CLIP_PCA_COMPONENTS = 64
MULTIMODAL_METRICS_CSV = Path('multimodal_xgboost_metrics.csv')
MULTIMODAL_TUNING_CSV = Path('multimodal_xgboost_tuning.csv')
MULTIMODAL_IMPORTANCE_CSV = Path('multimodal_xgboost_feature_importance.csv')
MULTIMODAL_IMPORTANCE_PNG = Path('multimodal_xgboost_feature_importance.png')

CLIP_FEATURES_CSV = Path('clip_visual_features.csv')
clip_features_df = pd.read_csv(CLIP_FEATURES_CSV)
cisf_features_df = pd.read_csv(Path('cisf_features.csv'))
CLIP_COLUMNS = [
    column for column in clip_features_df.columns
    if column.startswith('clip_') and column[5:].isdigit()
]
assert len(CLIP_COLUMNS) == 512, 'Expected 512 CLIP embedding features.'

multimodal_df = (
    text_df
    .merge(
        cisf_features_df[
            ['image_path', 'cisf_neighbour_mean_similarity', 'cisf_neighbour_max_similarity']
        ],
        on='image_path',
        how='inner',
    )
    .merge(
        clip_features_df[['image_path', 'clip_error'] + CLIP_COLUMNS],
        on='image_path',
        how='inner',
    )
)
multimodal_df['has_ocr_text'] = (
    multimodal_df['ocr_text'].fillna('').astype(str).str.strip().ne('').astype(float)
)

multimodal_df = multimodal_df.loc[
    multimodal_df['clip_error'].fillna('').eq('')
].copy().reset_index(drop=True)

assert multimodal_df['label'].nunique() == 2, 'Both classes are required.'
assert len(multimodal_df) == len(text_df), (
    'CLIP rows are missing. Re-run Step 12 until all feature rows are saved.'
)

for column in [
    'similarity_score', 'ocr_length', 'caption_length', 'has_ocr_text',
    'token_count', 'unique_token_ratio',
    'cisf_neighbour_mean_similarity', 'cisf_neighbour_max_similarity',
]:
    multimodal_df[column] = pd.to_numeric(
        multimodal_df[column], errors='coerce'
    ).fillna(0.0)

TABULAR_FEATURES = [
    'similarity_score',
    'ocr_length',
    'caption_length',
    'has_ocr_text',
    'token_count',
    'unique_token_ratio',
    'cisf_neighbour_mean_similarity',
    'cisf_neighbour_max_similarity',
]

train_data, test_data = train_test_split(
    multimodal_df,
    test_size=MULTIMODAL_TEST_SIZE,
    stratify=multimodal_df['label'],
    random_state=SEED,
)
y_train_multimodal = train_data['label'].eq('fake').astype(int)
y_test_multimodal = test_data['label'].eq('fake').astype(int)

multimodal_transformer = ColumnTransformer(
    transformers=[
        (
            'tfidf',
            TfidfVectorizer(
                tokenizer=str.split,
                token_pattern=None,
                lowercase=False,
                ngram_range=(1, 2),
                min_df=TEXT_MIN_DOCUMENT_FREQUENCY,
                max_features=TEXT_MAX_FEATURES,
                sublinear_tf=True,
                dtype=np.float32,
            ),
            'processed_text',
        ),
        ('tabular', 'passthrough', TABULAR_FEATURES),
        (
            'clip_pca',
            PCA(n_components=CLIP_PCA_COMPONENTS, random_state=SEED),
            CLIP_COLUMNS,
        ),
    ],
    sparse_threshold=0.30,
)

multimodal_xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=SEED,
    n_jobs=4,
    tree_method='hist',
    importance_type='gain',
)
multimodal_pipeline = Pipeline([
    ('features', multimodal_transformer),
    ('xgb', multimodal_xgb),
])

parameter_space = {
    'xgb__n_estimators': randint(150, 451),
    'xgb__max_depth': randint(3, 7),
    'xgb__learning_rate': loguniform(0.02, 0.15),
    'xgb__min_child_weight': randint(1, 7),
    'xgb__subsample': uniform(0.65, 0.35),
    'xgb__colsample_bytree': uniform(0.55, 0.40),
    'xgb__gamma': uniform(0.0, 1.0),
    'xgb__reg_alpha': loguniform(1e-5, 1e-1),
    'xgb__reg_lambda': loguniform(0.5, 5.0),
}

tuning_cv = StratifiedKFold(
    n_splits=TUNING_CV_FOLDS,
    shuffle=True,
    random_state=SEED,
)
fake_f1_scorer = make_scorer(f1_score, pos_label=1, zero_division=0)

tuned_multimodal_model = RandomizedSearchCV(
    estimator=multimodal_pipeline,
    param_distributions=parameter_space,
    n_iter=TUNING_ITERATIONS,
    scoring=fake_f1_scorer,
    cv=tuning_cv,
    random_state=SEED,
    n_jobs=1,
    refit=True,
    verbose=1,
)
tuned_multimodal_model.fit(train_data, y_train_multimodal)

tuning_results = pd.DataFrame(tuned_multimodal_model.cv_results_).sort_values(
    'rank_test_score'
)
tuning_results[
    ['rank_test_score', 'mean_test_score', 'std_test_score', 'params']
].to_csv(MULTIMODAL_TUNING_CSV, index=False)

multimodal_predictions_binary = tuned_multimodal_model.predict(test_data)
multimodal_predictions = np.where(
    multimodal_predictions_binary == 1, 'fake', 'real'
)
multimodal_actual = np.where(y_test_multimodal == 1, 'fake', 'real')

multimodal_metrics = pd.DataFrame([
    calculate_metrics(
        'Tuned multimodal XGBoost (TF-IDF + numeric + CISF + CLIP)',
        multimodal_actual,
        multimodal_predictions,
    )
])
multimodal_metrics['best_cv_f1_fake'] = tuned_multimodal_model.best_score_
multimodal_metrics['training_rows'] = len(train_data)
multimodal_metrics['test_rows'] = len(test_data)
multimodal_metrics.to_csv(MULTIMODAL_METRICS_CSV, index=False)

best_pipeline = tuned_multimodal_model.best_estimator_
best_feature_names = best_pipeline.named_steps['features'].get_feature_names_out()
best_importance = best_pipeline.named_steps['xgb'].feature_importances_
multimodal_importance = pd.DataFrame({
    'feature': best_feature_names,
    'gain_importance': best_importance,
}).sort_values('gain_importance', ascending=False).reset_index(drop=True)
multimodal_importance.to_csv(MULTIMODAL_IMPORTANCE_CSV, index=False)

top_multimodal_importance = multimodal_importance.head(25).sort_values('gain_importance')
fig, axis = plt.subplots(figsize=(10, 8))
axis.barh(
    top_multimodal_importance['feature'],
    top_multimodal_importance['gain_importance'],
    color='#0F766E',
)
axis.set_title('Tuned multimodal XGBoost: top 25 feature importances')
axis.set_xlabel('Relative gain importance')
fig.tight_layout()
fig.savefig(MULTIMODAL_IMPORTANCE_PNG, dpi=160, bbox_inches='tight')
plt.show()

print(f'Best 3-fold CV F1 (fake): {tuned_multimodal_model.best_score_:.4f}')
print(f'Best parameters: {tuned_multimodal_model.best_params_}')
print(multimodal_metrics.to_string(index=False, float_format=lambda value: f'{value:.4f}'))
print(f'\nMetrics saved to: {MULTIMODAL_METRICS_CSV.resolve()}')
print(f'Tuning results saved to: {MULTIMODAL_TUNING_CSV.resolve()}')
print(f'Feature importance saved to: {MULTIMODAL_IMPORTANCE_CSV.resolve()}')

