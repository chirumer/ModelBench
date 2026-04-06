from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
SSR_NET_DIR = REPO_ROOT / "SSR-Net"
STATIC_DIR = BASE_DIR / "static"
DATASET_STATIC_DIR = STATIC_DIR / "datasets"
DATASET_MANIFEST_DIR = BASE_DIR / "dataset_manifests"


@dataclass(frozen=True)
class DatasetDefinition:
    id: str
    name: str
    description: str
    hf_dataset: str
    hf_config: str | None
    hf_split: str

    @property
    def manifest_path(self) -> Path:
        return DATASET_MANIFEST_DIR / f"{self.id}.json"


@dataclass(frozen=True)
class ModelPresetDefinition:
    id: str
    label: str
    description: str
    age_weight_path: Path
    gender_weight_path: Path


DATASET_DEFINITIONS = {
    "utkface": DatasetDefinition(
        id="utkface",
        name="UTKFace",
        description="Exact-age face crops with gender and ethnicity labels.",
        hf_dataset="nu-delta/utkface",
        hf_config=None,
        hf_split="train",
    ),
    "fairface": DatasetDefinition(
        id="fairface",
        name="FairFace",
        description="Balanced face crops with age buckets, gender labels, and race labels.",
        hf_dataset="HuggingFaceM4/FairFace",
        hf_config="0.25",
        hf_split="train",
    ),
}


MODEL_PRESETS = {
    "wiki": ModelPresetDefinition(
        id="wiki",
        label="WIKI",
        description="WIKI age weights with WIKI gender weights.",
        age_weight_path=SSR_NET_DIR
        / "pre-trained/wiki/ssrnet_3_3_3_64_1.0_1.0/ssrnet_3_3_3_64_1.0_1.0.h5",
        gender_weight_path=SSR_NET_DIR
        / "pre-trained/wiki_gender_models/ssrnet_3_3_3_64_1.0_1.0/ssrnet_3_3_3_64_1.0_1.0.h5",
    ),
    "imdb": ModelPresetDefinition(
        id="imdb",
        label="IMDB",
        description="IMDB age weights with IMDB gender weights.",
        age_weight_path=SSR_NET_DIR
        / "pre-trained/imdb/ssrnet_3_3_3_64_1.0_1.0/ssrnet_3_3_3_64_1.0_1.0.h5",
        gender_weight_path=SSR_NET_DIR
        / "pre-trained/imdb_gender_models/ssrnet_3_3_3_64_1.0_1.0/ssrnet_3_3_3_64_1.0_1.0.h5",
    ),
    "morph": ModelPresetDefinition(
        id="morph",
        label="MORPH",
        description="MORPH2 age weights with MORPH gender weights.",
        age_weight_path=SSR_NET_DIR
        / "pre-trained/morph2/ssrnet_3_3_3_64_1.0_1.0/ssrnet_3_3_3_64_1.0_1.0.h5",
        gender_weight_path=SSR_NET_DIR
        / "pre-trained/morph_gender_models/ssrnet_3_3_3_64_1.0_1.0/ssrnet_3_3_3_64_1.0_1.0.h5",
    ),
}
