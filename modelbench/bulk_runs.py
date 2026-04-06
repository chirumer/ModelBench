from __future__ import annotations

import copy
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import uuid4

from modelbench.catalog import DATASET_DEFINITIONS

if TYPE_CHECKING:
    from modelbench.inference import InferenceService


DEFAULT_BABY_MAX = 12
DEFAULT_ADULT_MAX = 59
MAX_AGE = 116
DEFAULT_PRESET_PAIR = (DEFAULT_BABY_MAX, DEFAULT_ADULT_MAX)
PRESET_THRESHOLD_PAIRS = [(baby_max, adult_max) for baby_max in range(0, MAX_AGE) for adult_max in range(baby_max + 1, MAX_AGE + 1)]


class BulkRunInputError(ValueError):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class AgeClassRange:
    id: str
    label: str
    min_age: int
    max_age: int


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_age_class_ranges(baby_max: int, adult_max: int) -> list[AgeClassRange]:
    return [
        AgeClassRange(id="baby", label="Baby", min_age=0, max_age=baby_max),
        AgeClassRange(id="adult", label="Man/Woman", min_age=baby_max + 1, max_age=adult_max),
        AgeClassRange(id="old", label="Old", min_age=adult_max + 1, max_age=MAX_AGE),
    ]


def class_for_exact_age(age: int, baby_max: int, adult_max: int) -> str:
    for age_class in build_age_class_ranges(baby_max, adult_max):
        if age_class.min_age <= age <= age_class.max_age:
            return age_class.id
    return "old"


def parse_age_bucket(bucket: str) -> tuple[int, int]:
    bucket = bucket.strip()
    if "-" in bucket:
        left, right = bucket.split("-", 1)
        return int(left), int(right)
    if bucket.endswith("+"):
        return int(bucket[:-1]), MAX_AGE
    lowered = bucket.lower()
    if lowered.startswith("more than "):
        return int(lowered.replace("more than ", "", 1)), MAX_AGE
    raise BulkRunInputError(f"Unsupported FairFace age bucket '{bucket}'.", status_code=500)


def classes_for_bucket_overlap(bucket: str, baby_max: int, adult_max: int) -> set[str]:
    bucket_min, bucket_max = parse_age_bucket(bucket)
    overlaps = set()
    for age_class in build_age_class_ranges(baby_max, adult_max):
        if bucket_min <= age_class.max_age and age_class.min_age <= bucket_max:
            overlaps.add(age_class.id)
    return overlaps


def choose_evaluation_detection(detections: list[dict]) -> dict | None:
    if not detections:
        return None
    numeric_detections = [item for item in detections if item.get("face_confidence") is not None]
    if numeric_detections:
        return max(numeric_detections, key=lambda item: float(item["face_confidence"]))
    return detections[0]


def actual_age_classes_for_ground_truth(ground_truth: dict, baby_max: int, adult_max: int) -> set[str]:
    if ground_truth["age_kind"] == "exact":
        return {class_for_exact_age(int(ground_truth["age_value"]), baby_max, adult_max)}
    return classes_for_bucket_overlap(str(ground_truth["age_value"]), baby_max, adult_max)


def empty_age_class_breakdown(baby_max: int, adult_max: int) -> dict[str, dict]:
    return {
        age_class.id: {
            "label": age_class.label,
            "correct_count": 0,
            "total_count": 0,
            "accuracy": 0.0,
        }
        for age_class in build_age_class_ranges(baby_max, adult_max)
    }


def compute_age_metrics_for_evaluations(
    evaluations: list[dict],
    baby_max: int,
    adult_max: int,
) -> tuple[int, float, dict[str, dict]]:
    breakdown = empty_age_class_breakdown(baby_max, adult_max)
    age_correct_count = 0

    for evaluation in evaluations:
        actual_classes = actual_age_classes_for_ground_truth(
            evaluation["ground_truth"],
            baby_max,
            adult_max,
        )
        predicted_class = None
        if evaluation["predicted_age_years"] is not None:
            predicted_class = class_for_exact_age(
                int(evaluation["predicted_age_years"]),
                baby_max,
                adult_max,
            )
            if predicted_class in actual_classes:
                age_correct_count += 1

        for class_id in actual_classes:
            breakdown[class_id]["total_count"] += 1
            if predicted_class == class_id:
                breakdown[class_id]["correct_count"] += 1

    for item in breakdown.values():
        item["accuracy"] = item["correct_count"] / item["total_count"] if item["total_count"] > 0 else 0.0

    tested_count = len(evaluations)
    accuracy = age_correct_count / tested_count if tested_count > 0 else 0.0
    return age_correct_count, accuracy, breakdown


def compute_age_accuracy_for_evaluations(
    evaluations: list[dict],
    baby_max: int,
    adult_max: int,
) -> float:
    tested_count = len(evaluations)
    if tested_count == 0:
        return 0.0

    age_correct_count = 0
    for evaluation in evaluations:
        actual_classes = actual_age_classes_for_ground_truth(
            evaluation["ground_truth"],
            baby_max,
            adult_max,
        )
        if evaluation["predicted_age_years"] is None:
            continue
        predicted_class = class_for_exact_age(
            int(evaluation["predicted_age_years"]),
            baby_max,
            adult_max,
        )
        if predicted_class in actual_classes:
            age_correct_count += 1
    return age_correct_count / tested_count


def preset_distance_from_default(baby_max: int, adult_max: int) -> int:
    return abs(baby_max - DEFAULT_PRESET_PAIR[0]) + abs(adult_max - DEFAULT_PRESET_PAIR[1])


class BulkRunManager:
    def __init__(self, service: InferenceService) -> None:
        self._service = service
        self._lock = threading.Lock()
        self._runs: dict[str, dict] = {}
        self._active_run_id: str | None = None

    def start_run(self, model_ids: list[str], baby_max: int, adult_max: int) -> dict:
        unique_model_ids = list(dict.fromkeys(model_ids))
        self._validate_settings(unique_model_ids, baby_max, adult_max)

        with self._lock:
            if self._active_run_id is not None:
                active = self._runs.get(self._active_run_id)
                if active and active["status"] == "running":
                    raise BulkRunInputError("A bulk inference run is already active.", status_code=409)

            snapshot = self._build_snapshot(unique_model_ids, baby_max, adult_max)
            run_id = snapshot["run_id"]
            self._runs[run_id] = snapshot
            self._active_run_id = run_id

        worker = threading.Thread(target=self._execute_run, args=(run_id,), daemon=True)
        worker.start()
        return self.get_run(run_id)

    def update_settings(self, run_id: str, baby_max: int, adult_max: int) -> dict:
        self._validate_range_settings(baby_max, adult_max)
        with self._lock:
            snapshot = self._runs.get(run_id)
            if snapshot is None:
                raise BulkRunInputError(f"Unknown bulk run '{run_id}'.", status_code=404)
            snapshot["settings"]["baby_max"] = baby_max
            snapshot["settings"]["adult_max"] = adult_max
            self._recompute_all_age_metrics_locked(snapshot)
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> dict:
        with self._lock:
            snapshot = self._runs.get(run_id)
            if snapshot is None:
                raise BulkRunInputError(f"Unknown bulk run '{run_id}'.", status_code=404)
            payload = copy.deepcopy(snapshot)
            payload.pop("_evaluation_records", None)
            payload.pop("_preset_cache", None)
            return payload

    def get_class_preview(self, run_id: str, dataset_id: str, model_id: str, class_id: str) -> dict:
        with self._lock:
            snapshot = self._runs.get(run_id)
            if snapshot is None:
                raise BulkRunInputError(f"Unknown bulk run '{run_id}'.", status_code=404)
            if dataset_id not in snapshot["results"]:
                raise BulkRunInputError(f"Unknown dataset '{dataset_id}'.", status_code=404)
            if model_id not in snapshot["results"][dataset_id]["models"]:
                raise BulkRunInputError(f"Unknown model '{model_id}' for dataset '{dataset_id}'.", status_code=404)
            settings = copy.deepcopy(snapshot["settings"])
            evaluations = copy.deepcopy(snapshot["_evaluation_records"][dataset_id][model_id])
            dataset_meta = copy.deepcopy(snapshot["results"][dataset_id]["dataset"])
            selected_models = copy.deepcopy(snapshot["selected_models"])

        class_labels = {item.id: item.label for item in build_age_class_ranges(settings["baby_max"], settings["adult_max"])}
        if class_id not in class_labels:
            raise BulkRunInputError(f"Unknown class '{class_id}'.", status_code=404)

        records = {entry["id"]: entry for entry in self._service.get_dataset_records(dataset_id)}
        model_entry = next((item for item in selected_models if item["id"] == model_id), None)
        if model_entry is None:
            raise BulkRunInputError(f"Unknown model '{model_id}'.", status_code=404)

        items = []
        for evaluation in evaluations:
            actual_class_ids = sorted(
                actual_age_classes_for_ground_truth(
                    evaluation["ground_truth"],
                    settings["baby_max"],
                    settings["adult_max"],
                )
            )
            if class_id not in actual_class_ids:
                continue

            predicted_class_id = None
            if evaluation["predicted_age_years"] is not None:
                predicted_class_id = class_for_exact_age(
                    int(evaluation["predicted_age_years"]),
                    settings["baby_max"],
                    settings["adult_max"],
                )

            record = records.get(evaluation["dataset_image_id"])
            if record is None:
                continue

            items.append(
                {
                    "dataset_image_id": evaluation["dataset_image_id"],
                    "image_url": record.get("image_url"),
                    "thumbnail_url": record.get("thumbnail_url"),
                    "actual_age_display": evaluation["ground_truth"].get("age_display")
                    or str(evaluation["ground_truth"].get("age_value", "n/a")),
                    "actual_class_ids": actual_class_ids,
                    "actual_class_labels": [class_labels[item] for item in actual_class_ids],
                    "predicted_class_id": predicted_class_id,
                    "predicted_class_label": class_labels.get(predicted_class_id, "n/a"),
                    "predicted_age_years": evaluation["predicted_age_years"],
                    "correct_for_clicked_class": predicted_class_id == class_id,
                    "missed_detection": evaluation["missed_detection"],
                }
            )

        correct_count = sum(1 for item in items if item["correct_for_clicked_class"])
        return {
            "run_id": run_id,
            "dataset": dataset_meta,
            "model": model_entry,
            "class_id": class_id,
            "class_label": class_labels[class_id],
            "settings": settings,
            "summary": {
                "total_count": len(items),
                "correct_count": correct_count,
            },
            "items": items,
        }

    def get_presets(self, run_id: str) -> dict:
        with self._lock:
            snapshot = self._runs.get(run_id)
            if snapshot is None:
                raise BulkRunInputError(f"Unknown bulk run '{run_id}'.", status_code=404)
            settings = copy.deepcopy(snapshot["settings"])
            version = snapshot["progress"]["tested_images"]
            cached = snapshot.get("_preset_cache")
            if cached and cached["version"] == version:
                return self._serialize_presets(copy.deepcopy(cached["groups"]), settings)
            datasets = copy.deepcopy(snapshot["datasets"])
            selected_models = copy.deepcopy(snapshot["selected_models"])
            evaluation_records = copy.deepcopy(snapshot["_evaluation_records"])

        groups = self._compute_preset_groups(datasets, selected_models, evaluation_records)

        with self._lock:
            latest = self._runs.get(run_id)
            if latest is not None and latest["progress"]["tested_images"] == version:
                latest["_preset_cache"] = {
                    "version": version,
                    "groups": copy.deepcopy(groups),
                }
        return self._serialize_presets(groups, settings)

    def _validate_settings(self, model_ids: list[str], baby_max: int, adult_max: int) -> None:
        if not model_ids:
            raise BulkRunInputError("model_ids must include at least one model.")
        self._validate_range_settings(baby_max, adult_max)

        available_models = {entry["id"] for entry in self._service.list_models()}
        unknown = [model_id for model_id in model_ids if model_id not in available_models]
        if unknown:
            raise BulkRunInputError(f"Unknown model ids: {', '.join(unknown)}.")

    def _validate_range_settings(self, baby_max: int, adult_max: int) -> None:
        if baby_max < 0 or adult_max > MAX_AGE or baby_max >= adult_max:
            raise BulkRunInputError("Use slider values where 0 <= baby_max < adult_max <= 116.")

    def _build_snapshot(self, model_ids: list[str], baby_max: int, adult_max: int) -> dict:
        models_by_id = {entry["id"]: entry for entry in self._service.list_models()}
        selected_models = [copy.deepcopy(models_by_id[model_id]) for model_id in model_ids]
        datasets = self._service.list_datasets()
        total_images = sum(dataset["image_count"] for dataset in datasets) * len(selected_models)

        results = {}
        evaluation_records = {}
        for dataset in datasets:
            dataset_models = {}
            dataset_records = {}
            for model in selected_models:
                dataset_models[model["id"]] = {
                    "tested_count": 0,
                    "total_count": dataset["image_count"],
                    "gender_correct_count": 0,
                    "gender_accuracy": 0.0,
                    "age_class_correct_count": 0,
                    "age_class_accuracy": 0.0,
                    "age_class_breakdown": self._empty_age_class_breakdown(),
                    "missed_detection_count": 0,
                    "status": "queued",
                    "last_error": None,
                }
                dataset_records[model["id"]] = []
            results[dataset["id"]] = {"dataset": copy.deepcopy(dataset), "models": dataset_models}
            evaluation_records[dataset["id"]] = dataset_records

        return {
            "run_id": uuid4().hex,
            "status": "queued",
            "created_at": utc_now_iso(),
            "started_at": None,
            "finished_at": None,
            "selected_models": selected_models,
            "datasets": copy.deepcopy(datasets),
            "settings": {"baby_max": baby_max, "adult_max": adult_max},
            "progress": {
                "tested_images": 0,
                "total_images": total_images,
                "current_dataset_id": None,
                "current_model_id": None,
            },
            "results": results,
            "_evaluation_records": evaluation_records,
            "_preset_cache": None,
        }

    def _execute_run(self, run_id: str) -> None:
        with self._lock:
            snapshot = self._runs[run_id]
            snapshot["status"] = "running"
            snapshot["started_at"] = utc_now_iso()

        try:
            run_snapshot = self.get_run(run_id)
            selected_models = run_snapshot["selected_models"]
            datasets = run_snapshot["datasets"]

            for dataset in datasets:
                dataset_id = dataset["id"]
                records = self._service.get_dataset_records(dataset_id)

                for record in records:
                    for model in selected_models:
                        model_id = model["id"]
                        self._mark_row_running(run_id, dataset_id, model_id)
                        self._set_progress(run_id, dataset_id=dataset_id, model_id=model_id)
                        self._evaluate_record(
                            run_id=run_id,
                            dataset_id=dataset_id,
                            record=record,
                            model_id=model_id,
                        )

                self._mark_dataset_done(run_id, dataset_id)

            with self._lock:
                snapshot = self._runs[run_id]
                snapshot["status"] = "done"
                snapshot["finished_at"] = utc_now_iso()
                snapshot["progress"]["current_dataset_id"] = None
                snapshot["progress"]["current_model_id"] = None
                if self._active_run_id == run_id:
                    self._active_run_id = None
        except Exception as exc:  # pragma: no cover - defensive guard
            with self._lock:
                snapshot = self._runs[run_id]
                snapshot["status"] = "error"
                snapshot["finished_at"] = utc_now_iso()
                snapshot["progress"]["current_dataset_id"] = None
                snapshot["progress"]["current_model_id"] = None
                snapshot["last_error"] = str(exc)
                if self._active_run_id == run_id:
                    self._active_run_id = None

    def _mark_row_running(self, run_id: str, dataset_id: str, model_id: str) -> None:
        with self._lock:
            row = self._runs[run_id]["results"][dataset_id]["models"][model_id]
            if row["status"] == "queued":
                row["status"] = "running"

    def _mark_dataset_done(self, run_id: str, dataset_id: str) -> None:
        with self._lock:
            dataset_rows = self._runs[run_id]["results"][dataset_id]["models"].values()
            for row in dataset_rows:
                if row["status"] == "running":
                    row["status"] = "done"

    def _set_progress(self, run_id: str, *, dataset_id: str | None, model_id: str | None) -> None:
        with self._lock:
            progress = self._runs[run_id]["progress"]
            progress["current_dataset_id"] = dataset_id
            progress["current_model_id"] = model_id

    def _evaluate_record(
        self,
        *,
        run_id: str,
        dataset_id: str,
        record: dict,
        model_id: str,
    ) -> None:
        row = None
        try:
            payload = self._service.analyze_dataset_image(record["id"], model_id)
            ground_truth = payload["ground_truth"]
            detection = choose_evaluation_detection(payload["detections"])

            gender_correct = False
            missed_detection = detection is None
            predicted_age_years = None
            predicted_gender_label = None

            if detection is not None:
                predicted_gender_label = detection.get("gender_label")
                gender_correct = predicted_gender_label == ground_truth.get("gender_value")
                if detection.get("age_years") is not None:
                    predicted_age_years = int(detection["age_years"])

            with self._lock:
                row = self._runs[run_id]["results"][dataset_id]["models"][model_id]
                evaluations = self._runs[run_id]["_evaluation_records"][dataset_id][model_id]
                evaluations.append(
                    {
                        "dataset_image_id": record["id"],
                        "ground_truth": copy.deepcopy(ground_truth),
                        "predicted_age_years": predicted_age_years,
                        "predicted_gender_label": predicted_gender_label,
                        "missed_detection": missed_detection,
                    }
                )
                row["tested_count"] += 1
                if gender_correct:
                    row["gender_correct_count"] += 1
                if missed_detection:
                    row["missed_detection_count"] += 1
                row["gender_accuracy"] = self._ratio(row["gender_correct_count"], row["tested_count"])
                settings = self._runs[run_id]["settings"]
                self._recompute_row_age_metrics_locked(
                    self._runs[run_id],
                    dataset_id,
                    model_id,
                    settings["baby_max"],
                    settings["adult_max"],
                )
                self._runs[run_id]["_preset_cache"] = None
                self._runs[run_id]["progress"]["tested_images"] += 1
        except Exception as exc:
            with self._lock:
                row = self._runs[run_id]["results"][dataset_id]["models"][model_id]
                evaluations = self._runs[run_id]["_evaluation_records"][dataset_id][model_id]
                evaluations.append(
                    {
                        "dataset_image_id": record["id"],
                        "ground_truth": copy.deepcopy(record["ground_truth"]),
                        "predicted_age_years": None,
                        "predicted_gender_label": None,
                        "missed_detection": True,
                    }
                )
                row["tested_count"] += 1
                row["missed_detection_count"] += 1
                row["gender_accuracy"] = self._ratio(row["gender_correct_count"], row["tested_count"])
                settings = self._runs[run_id]["settings"]
                self._recompute_row_age_metrics_locked(
                    self._runs[run_id],
                    dataset_id,
                    model_id,
                    settings["baby_max"],
                    settings["adult_max"],
                )
                self._runs[run_id]["_preset_cache"] = None
                row["status"] = "error"
                row["last_error"] = str(exc)
                self._runs[run_id]["progress"]["tested_images"] += 1

    def _empty_age_class_breakdown(self) -> dict[str, dict]:
        return empty_age_class_breakdown(DEFAULT_BABY_MAX, DEFAULT_ADULT_MAX)

    def _recompute_all_age_metrics_locked(self, snapshot: dict) -> None:
        baby_max = snapshot["settings"]["baby_max"]
        adult_max = snapshot["settings"]["adult_max"]
        for dataset_id, dataset_payload in snapshot["results"].items():
            for model_id in dataset_payload["models"]:
                self._recompute_row_age_metrics_locked(
                    snapshot,
                    dataset_id,
                    model_id,
                    baby_max,
                    adult_max,
                )

    def _recompute_row_age_metrics_locked(
        self,
        snapshot: dict,
        dataset_id: str,
        model_id: str,
        baby_max: int,
        adult_max: int,
    ) -> None:
        row = snapshot["results"][dataset_id]["models"][model_id]
        evaluations = snapshot["_evaluation_records"][dataset_id][model_id]
        age_correct_count, _, breakdown = compute_age_metrics_for_evaluations(
            evaluations,
            baby_max,
            adult_max,
        )
        row["age_class_correct_count"] = age_correct_count
        row["age_class_accuracy"] = self._ratio(age_correct_count, row["tested_count"])
        row["age_class_breakdown"] = breakdown

    def _compute_preset_groups(self, datasets: list[dict], selected_models: list[dict], evaluation_records: dict) -> dict:
        rows = []
        for dataset in datasets:
            for model in selected_models:
                evaluations = evaluation_records[dataset["id"]][model["id"]]
                tested_images = len(evaluations)
                if tested_images <= 0:
                    continue
                rows.append(
                    {
                        "dataset_id": dataset["id"],
                        "dataset_name": dataset["name"],
                        "model_id": model["id"],
                        "model_label": model["label"],
                        "evaluations": evaluations,
                        "tested_images": tested_images,
                    }
                )

        if not rows:
            return {
                "dataset_presets": [],
                "model_presets": [],
                "combination_presets": [],
            }

        row_scores = []
        for row in rows:
            row_scores.append(
                [
                    compute_age_accuracy_for_evaluations(row["evaluations"], baby_max, adult_max)
                    for baby_max, adult_max in PRESET_THRESHOLD_PAIRS
                ]
            )

        dataset_groups: dict[str, list[int]] = {}
        model_groups: dict[str, list[int]] = {}
        for index, row in enumerate(rows):
            dataset_groups.setdefault(row["dataset_id"], []).append(index)
            model_groups.setdefault(row["model_id"], []).append(index)

        dataset_presets = [
            self._build_best_preset(
                scope_type="dataset",
                scope_label=row_group[0]["dataset_name"],
                dataset_id=dataset_id,
                model_id=None,
                row_indices=indices,
                rows=rows,
                row_scores=row_scores,
            )
            for dataset_id, indices in dataset_groups.items()
            for row_group in [[rows[index] for index in indices]]
        ]
        model_presets = [
            self._build_best_preset(
                scope_type="model",
                scope_label=row_group[0]["model_label"],
                dataset_id=None,
                model_id=model_id,
                row_indices=indices,
                rows=rows,
                row_scores=row_scores,
            )
            for model_id, indices in model_groups.items()
            for row_group in [[rows[index] for index in indices]]
        ]
        combination_presets = [
            self._build_best_preset(
                scope_type="combination",
                scope_label=f"{row['dataset_name']} · {row['model_label']}",
                dataset_id=row["dataset_id"],
                model_id=row["model_id"],
                row_indices=[index],
                rows=rows,
                row_scores=row_scores,
            )
            for index, row in enumerate(rows)
        ]

        return {
            "dataset_presets": dataset_presets,
            "model_presets": model_presets,
            "combination_presets": combination_presets,
        }

    def _build_best_preset(
        self,
        *,
        scope_type: str,
        scope_label: str,
        dataset_id: str | None,
        model_id: str | None,
        row_indices: list[int],
        rows: list[dict],
        row_scores: list[list[float]],
    ) -> dict:
        best_index = 0
        best_key = None
        for pair_index, (baby_max, adult_max) in enumerate(PRESET_THRESHOLD_PAIRS):
            score = sum(row_scores[row_index][pair_index] for row_index in row_indices) / len(row_indices)
            key = (
                score,
                -preset_distance_from_default(baby_max, adult_max),
                -baby_max,
                -adult_max,
            )
            if best_key is None or key > best_key:
                best_key = key
                best_index = pair_index

        baby_max, adult_max = PRESET_THRESHOLD_PAIRS[best_index]
        return {
            "id": f"{scope_type}:{dataset_id or 'all'}:{model_id or 'all'}",
            "scope_type": scope_type,
            "label": scope_label,
            "dataset_id": dataset_id,
            "model_id": model_id,
            "baby_max": baby_max,
            "adult_max": adult_max,
            "score_accuracy": best_key[0] if best_key is not None else 0.0,
            "tested_images": sum(rows[row_index]["tested_images"] for row_index in row_indices),
        }

    def _serialize_presets(self, groups: dict, settings: dict) -> dict:
        payload = copy.deepcopy(groups)
        for group_name in ("dataset_presets", "model_presets", "combination_presets"):
            for item in payload[group_name]:
                item["is_active"] = (
                    item["baby_max"] == settings["baby_max"]
                    and item["adult_max"] == settings["adult_max"]
                )
        payload["settings"] = settings
        return payload

    @staticmethod
    def _ratio(correct: int, tested: int) -> float:
        if tested <= 0:
            return 0.0
        return correct / tested
