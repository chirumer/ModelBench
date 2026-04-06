from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset, load_dataset_builder
from PIL import Image

from modelbench.catalog import DATASET_DEFINITIONS, DATASET_STATIC_DIR


THUMBNAIL_SIZE = (96, 96)


def parse_args():
    parser = argparse.ArgumentParser(description="Download and persist local Hugging Face dataset samples.")
    parser.add_argument("--count", type=int, default=100, help="Number of images per dataset.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        choices=sorted(DATASET_DEFINITIONS.keys()),
        default=sorted(DATASET_DEFINITIONS.keys()),
        help="Dataset ids to ingest.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete any existing local dataset images before writing fresh files.",
    )
    return parser.parse_args()


def fairface_ground_truth(example, age_names, gender_names, race_names):
    age_bucket = age_names[int(example["age"])]
    gender = gender_names[int(example["gender"])]
    race = race_names[int(example["race"])]
    return {
        "dataset_type": "fairface",
        "age_kind": "bucket",
        "age_display": age_bucket,
        "age_value": age_bucket,
        "gender_display": gender,
        "gender_value": gender.lower(),
        "demographic_label": race,
        "comparison_age_display": age_bucket,
    }


def utkface_ground_truth(example):
    age = int(example["age"])
    gender = str(example["gender"])
    ethnicity = str(example["ethnicity"])
    return {
        "dataset_type": "utkface",
        "age_kind": "exact",
        "age_display": str(age),
        "age_value": age,
        "gender_display": gender,
        "gender_value": gender.lower(),
        "demographic_label": ethnicity,
        "comparison_age_display": str(age),
    }


def save_dataset(dataset_id: str, count: int, overwrite: bool):
    definition = DATASET_DEFINITIONS[dataset_id]
    builder = load_dataset_builder(definition.hf_dataset, definition.hf_config)
    dataset = load_dataset(
        definition.hf_dataset,
        definition.hf_config,
        split=definition.hf_split,
        streaming=True,
    )

    image_dir = DATASET_STATIC_DIR / dataset_id / "images"
    thumb_dir = DATASET_STATIC_DIR / dataset_id / "thumbs"
    manifest_dir = definition.manifest_path.parent

    if overwrite:
        for directory in [image_dir, thumb_dir]:
            if directory.exists():
                for child in directory.iterdir():
                    child.unlink()

    image_dir.mkdir(parents=True, exist_ok=True)
    thumb_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    images = []
    age_names = None
    gender_names = None
    race_names = None
    if dataset_id == "fairface":
        age_names = builder.info.features["age"].names
        gender_names = builder.info.features["gender"].names
        race_names = builder.info.features["race"].names

    for example in dataset:
        if len(images) >= count:
            break

        image = example.get("image")
        if image is None:
            continue

        image = image.convert("RGB")
        item_index = len(images) + 1
        stem = f"{dataset_id}-{item_index:03d}"
        image_path = image_dir / f"{stem}.jpg"
        thumb_path = thumb_dir / f"{stem}.jpg"

        image.save(image_path, format="JPEG", quality=90)
        thumb = image.copy()
        thumb.thumbnail(THUMBNAIL_SIZE)
        if thumb.size != THUMBNAIL_SIZE:
            padded = Image.new("RGB", THUMBNAIL_SIZE, color=(245, 245, 245))
            offset = ((THUMBNAIL_SIZE[0] - thumb.size[0]) // 2, (THUMBNAIL_SIZE[1] - thumb.size[1]) // 2)
            padded.paste(thumb, offset)
            thumb = padded
        thumb.save(thumb_path, format="JPEG", quality=85)

        if dataset_id == "fairface":
            ground_truth = fairface_ground_truth(example, age_names, gender_names, race_names)
            label_summary = f"{ground_truth['age_display']} • {ground_truth['gender_display']} • {ground_truth['demographic_label']}"
        else:
            ground_truth = utkface_ground_truth(example)
            label_summary = f"{ground_truth['age_display']} • {ground_truth['gender_display']} • {ground_truth['demographic_label']}"

        images.append(
            {
                "id": stem,
                "dataset_id": dataset_id,
                "image_path": str(image_path.resolve()),
                "thumbnail_path": str(thumb_path.resolve()),
                "image_url": f"/static/datasets/{dataset_id}/images/{stem}.jpg",
                "thumbnail_url": f"/static/datasets/{dataset_id}/thumbs/{stem}.jpg",
                "label_summary": label_summary,
                "ground_truth": ground_truth,
            }
        )

    manifest = {
        "dataset": {
            "id": dataset_id,
            "name": definition.name,
            "description": definition.description,
            "hf_dataset": definition.hf_dataset,
            "hf_config": definition.hf_config,
            "hf_split": definition.hf_split,
            "image_count": len(images),
        },
        "images": images,
    }
    definition.manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"{dataset_id}: wrote {len(images)} images to {image_dir}")


def main():
    args = parse_args()
    for dataset_id in args.datasets:
        save_dataset(dataset_id, args.count, args.overwrite)


if __name__ == "__main__":
    main()
