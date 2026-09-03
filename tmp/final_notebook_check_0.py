from transformers import CLIPModel, CLIPProcessor

CLIP_MODEL_ID = 'openai/clip-vit-base-patch32'
CLIP_FEATURES_CSV = Path('clip_visual_features.csv')
CLIP_BATCH_SIZE = 16
CLIP_CHECKPOINT_EVERY = 128

clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(DEVICE)
clip_model.eval()
CLIP_DIMENSION = clip_model.config.projection_dim
CLIP_COLUMNS = [f'clip_{index:03d}' for index in range(CLIP_DIMENSION)]
CLIP_FILE_COLUMNS = ['image_path', 'label', 'clip_error'] + CLIP_COLUMNS

source_features = pd.read_csv(FEATURES_CSV)
clip_source = source_features.loc[
    source_features['error'].fillna('').eq('') & source_features['label'].isin(LABELS),
    ['image_path', 'label'],
].drop_duplicates('image_path').reset_index(drop=True)

if CLIP_FEATURES_CSV.exists():
    clip_features_df = pd.read_csv(CLIP_FEATURES_CSV)
    if set(CLIP_FILE_COLUMNS).issubset(clip_features_df.columns):
        clip_features_df = clip_features_df[CLIP_FILE_COLUMNS].copy()
    else:
        print('Existing CLIP CSV has a different schema; starting again.')
        clip_features_df = pd.DataFrame(columns=CLIP_FILE_COLUMNS)
else:
    clip_features_df = pd.DataFrame(columns=CLIP_FILE_COLUMNS)


def save_clip_checkpoint(dataframe: pd.DataFrame) -> None:
    temporary_path = CLIP_FEATURES_CSV.with_name('clip_visual_features.tmp.csv')
    dataframe.to_csv(temporary_path, index=False)
    while True:
        try:
            os.replace(temporary_path, CLIP_FEATURES_CSV)
            return
        except PermissionError:
            print('Close clip_visual_features.csv in Excel. Retrying in 5 seconds...')
            time.sleep(5)


completed_clip_paths = set(clip_features_df['image_path'].dropna().astype(str))
clip_pending = clip_source.loc[
    ~clip_source['image_path'].isin(completed_clip_paths)
].to_dict('records')

print(f'CLIP embeddings already saved: {len(clip_features_df):,}')
print(f'CLIP embeddings remaining: {len(clip_pending):,}')

new_clip_rows = []
for start_index in range(0, len(clip_pending), CLIP_BATCH_SIZE):
    batch_records = clip_pending[start_index:start_index + CLIP_BATCH_SIZE]
    valid_records, batch_images = [], []

    for record in batch_records:
        try:
            with Image.open(record['image_path']) as opened_image:
                batch_images.append(opened_image.convert('RGB'))
            valid_records.append(record)
        except Exception as exc:
            new_clip_rows.append({
                'image_path': record['image_path'],
                'label': record['label'],
                'clip_error': f'{type(exc).__name__}: {exc}',
                **{column: np.nan for column in CLIP_COLUMNS},
            })

    if valid_records:
        clip_inputs = clip_processor(images=batch_images, return_tensors='pt').to(DEVICE)
        with torch.inference_mode():
            image_output = clip_model.get_image_features(**clip_inputs)
            batch_embeddings = image_output.pooler_output
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, dim=1)
        batch_embeddings = batch_embeddings.cpu().numpy()

        for record, embedding in zip(valid_records, batch_embeddings):
            new_clip_rows.append({
                'image_path': record['image_path'],
                'label': record['label'],
                'clip_error': '',
                **dict(zip(CLIP_COLUMNS, embedding.astype(float))),
            })

    processed_count = min(start_index + CLIP_BATCH_SIZE, len(clip_pending))
    if processed_count % CLIP_CHECKPOINT_EVERY == 0 or processed_count == len(clip_pending):
        clip_features_df = pd.concat(
            [clip_features_df, pd.DataFrame(new_clip_rows)],
            ignore_index=True,
        ).drop_duplicates('image_path', keep='last')
        save_clip_checkpoint(clip_features_df)
        new_clip_rows = []
        print(f'CLIP checkpoint: {processed_count:,}/{len(clip_pending):,}')

print(f'CLIP features saved to: {CLIP_FEATURES_CSV.resolve()}')
print(f'Rows: {len(clip_features_df):,}; failed rows: {(clip_features_df["clip_error"].fillna("") != "").sum():,}')

