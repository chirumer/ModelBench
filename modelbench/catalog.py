from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
SSR_NET_DIR = REPO_ROOT / "SSR-Net"
DEEPFACE_DIR = REPO_ROOT / "deepface"
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
    provider: str
    family: str
    age_weight_path: Path | None = None
    gender_weight_path: Path | None = None
    detector_backend: str | None = None


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
        label="SSR-Net (WIKI)",
        description="SSR-Net using WIKI age weights with WIKI gender weights.",
        provider="ssrnet",
        family="ssrnet",
        age_weight_path=SSR_NET_DIR
        / "pre-trained/wiki/ssrnet_3_3_3_64_1.0_1.0/ssrnet_3_3_3_64_1.0_1.0.h5",
        gender_weight_path=SSR_NET_DIR
        / "pre-trained/wiki_gender_models/ssrnet_3_3_3_64_1.0_1.0/ssrnet_3_3_3_64_1.0_1.0.h5",
    ),
    "imdb": ModelPresetDefinition(
        id="imdb",
        label="SSR-Net (IMDB)",
        description="SSR-Net using IMDB age weights with IMDB gender weights.",
        provider="ssrnet",
        family="ssrnet",
        age_weight_path=SSR_NET_DIR
        / "pre-trained/imdb/ssrnet_3_3_3_64_1.0_1.0/ssrnet_3_3_3_64_1.0_1.0.h5",
        gender_weight_path=SSR_NET_DIR
        / "pre-trained/imdb_gender_models/ssrnet_3_3_3_64_1.0_1.0/ssrnet_3_3_3_64_1.0_1.0.h5",
    ),
    "morph": ModelPresetDefinition(
        id="morph",
        label="SSR-Net (MORPH)",
        description="SSR-Net using MORPH2 age weights with MORPH gender weights.",
        provider="ssrnet",
        family="ssrnet",
        age_weight_path=SSR_NET_DIR
        / "pre-trained/morph2/ssrnet_3_3_3_64_1.0_1.0/ssrnet_3_3_3_64_1.0_1.0.h5",
        gender_weight_path=SSR_NET_DIR
        / "pre-trained/morph_gender_models/ssrnet_3_3_3_64_1.0_1.0/ssrnet_3_3_3_64_1.0_1.0.h5",
    ),
    "deepface": ModelPresetDefinition(
        id="deepface",
        label="DeepFace",
        description="DeepFace age and gender analysis with MTCNN face detection.",
        provider="deepface",
        family="demography",
        detector_backend="mtcnn",
    ),
}
