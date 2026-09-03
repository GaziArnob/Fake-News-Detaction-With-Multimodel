from pathlib import Path
import csv
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sample-size", type=int, default=8000)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

project_dir = Path.cwd()
image_dir = project_dir / "private_image_set"
output_dir = project_dir / "data"
output_dir.mkdir(exist_ok=True)

valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
images = [
    path for path in image_dir.rglob("*")
    if path.is_file() and path.suffix.lower() in valid_extensions
]

if len(images) < args.sample_size:
    raise ValueError(
        f"Only {len(images)} images found, but you requested {args.sample_size}."
    )

random_generator = random.Random(args.seed)
sampled_images = random_generator.sample(images, args.sample_size)

output_file = output_dir / f"dataset_sample_{args.sample_size}.csv"

columns = [
    "image_id",
    "image_path",
    "label",
    "source_url",
    "claim",
    "language",
    "ocr_text",
    "ocr_confidence",
    "text_present",
    "split"
]

with output_file.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()

    for index, image_path in enumerate(sampled_images, start=1):
        writer.writerow({
            "image_id": f"img_{index:05d}",
            "image_path": image_path.relative_to(project_dir).as_posix(),
            "label": "",
            "source_url": "",
            "claim": "",
            "language": "",
            "ocr_text": "",
            "ocr_confidence": "",
            "text_present": "",
            "split": ""
        })

print(f"Created: {output_file}")
print(f"Images sampled: {len(sampled_images)}")