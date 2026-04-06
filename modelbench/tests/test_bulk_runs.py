import time

from modelbench.bulk_runs import (
    BulkRunManager,
    choose_evaluation_detection,
    class_for_exact_age,
    classes_for_bucket_overlap,
)


class FakeBulkService:
    def __init__(self):
        self.analysis_calls = 0

    def list_models(self):
        return [
            {
                "id": "wiki",
                "label": "SSR-Net (WIKI)",
                "description": "WIKI preset.",
                "provider": "ssrnet",
                "family": "ssrnet",
            }
        ]

    def list_datasets(self):
        return [
            {
                "id": "utkface",
                "name": "UTKFace",
                "description": "Exact-age face crops.",
                "image_count": 2,
            }
        ]

    def get_dataset_records(self, dataset_id):
        assert dataset_id == "utkface"
        return [
            {
                "id": "utkface-001",
                "image_url": "/static/datasets/utkface/images/utkface-001.jpg",
                "thumbnail_url": "/static/datasets/utkface/thumbs/utkface-001.jpg",
                "ground_truth": {
                    "dataset_type": "utkface",
                    "age_kind": "exact",
                    "age_display": "10",
                    "age_value": 10,
                    "gender_display": "Male",
                    "gender_value": "male",
                },
            },
            {
                "id": "utkface-002",
                "image_url": "/static/datasets/utkface/images/utkface-002.jpg",
                "thumbnail_url": "/static/datasets/utkface/thumbs/utkface-002.jpg",
                "ground_truth": {
                    "dataset_type": "utkface",
                    "age_kind": "exact",
                    "age_display": "70",
                    "age_value": 70,
                    "gender_display": "Female",
                    "gender_value": "female",
                },
            },
        ]

    def analyze_dataset_image(self, dataset_image_id, model_id):
        self.analysis_calls += 1
        assert model_id == "wiki"
        if dataset_image_id == "utkface-001":
            return {
                "ground_truth": {
                    "dataset_type": "utkface",
                    "age_kind": "exact",
                    "age_display": "10",
                    "age_value": 10,
                    "gender_display": "Male",
                    "gender_value": "male",
                },
                "detections": [
                    {
                        "face_confidence": 0.2,
                        "age_years": 50,
                        "gender_label": "female",
                    },
                    {
                        "face_confidence": 0.95,
                        "age_years": 10,
                        "gender_label": "male",
                    },
                ],
            }
        return {
            "ground_truth": {
                "dataset_type": "utkface",
                "age_kind": "exact",
                "age_display": "70",
                "age_value": 70,
                "gender_display": "Female",
                "gender_value": "female",
            },
            "detections": [],
        }


def wait_for_run(manager, run_id, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        snapshot = manager.get_run(run_id)
        if snapshot["status"] not in {"queued", "running"}:
            return snapshot
        time.sleep(0.02)
    raise AssertionError("Timed out waiting for bulk run to complete.")


def test_class_for_exact_age_uses_slider_ranges():
    assert class_for_exact_age(2, 12, 59) == "baby"
    assert class_for_exact_age(30, 12, 59) == "adult"
    assert class_for_exact_age(90, 12, 59) == "old"


def test_fairface_bucket_overlap_supports_split_buckets():
    assert classes_for_bucket_overlap("10-19", 12, 59) == {"baby", "adult"}
    assert classes_for_bucket_overlap("60-69", 12, 59) == {"old"}


def test_choose_evaluation_detection_prefers_highest_confidence():
    detection = choose_evaluation_detection(
        [
            {"face_confidence": 0.1, "age_years": 50},
            {"face_confidence": 0.9, "age_years": 12},
        ]
    )
    assert detection["age_years"] == 12


def test_bulk_run_tracks_accuracy_and_missed_detections():
    service = FakeBulkService()
    manager = BulkRunManager(service)
    snapshot = manager.start_run(["wiki"], baby_max=12, adult_max=59)
    finished = wait_for_run(manager, snapshot["run_id"])

    row = finished["results"]["utkface"]["models"]["wiki"]
    assert finished["status"] == "done"
    assert finished["progress"]["tested_images"] == 2
    assert row["tested_count"] == 2
    assert row["gender_correct_count"] == 1
    assert row["age_class_correct_count"] == 1
    assert row["missed_detection_count"] == 1
    assert row["gender_accuracy"] == 0.5
    assert row["age_class_accuracy"] == 0.5
    assert row["age_class_breakdown"]["baby"]["correct_count"] == 1
    assert row["age_class_breakdown"]["baby"]["total_count"] == 1
    assert row["age_class_breakdown"]["old"]["correct_count"] == 0
    assert row["age_class_breakdown"]["old"]["total_count"] == 1
    assert row["status"] == "done"
    assert service.analysis_calls == 2


def test_bulk_run_recomputes_age_metrics_without_rerunning_inference():
    service = FakeBulkService()
    manager = BulkRunManager(service)
    snapshot = manager.start_run(["wiki"], baby_max=12, adult_max=59)
    finished = wait_for_run(manager, snapshot["run_id"])

    updated = manager.update_settings(finished["run_id"], baby_max=5, adult_max=50)
    row = updated["results"]["utkface"]["models"]["wiki"]

    assert service.analysis_calls == 2
    assert updated["settings"] == {"baby_max": 5, "adult_max": 50}
    assert row["gender_correct_count"] == 1
    assert row["gender_accuracy"] == 0.5
    assert row["age_class_correct_count"] == 1
    assert row["age_class_accuracy"] == 0.5
    assert row["age_class_breakdown"]["baby"]["total_count"] == 0
    assert row["age_class_breakdown"]["adult"]["correct_count"] == 1
    assert row["age_class_breakdown"]["adult"]["total_count"] == 1
    assert row["age_class_breakdown"]["old"]["total_count"] == 1


def test_class_preview_returns_all_actual_class_images():
    service = FakeBulkService()
    manager = BulkRunManager(service)
    snapshot = manager.start_run(["wiki"], baby_max=12, adult_max=59)
    finished = wait_for_run(manager, snapshot["run_id"])

    preview = manager.get_class_preview(finished["run_id"], "utkface", "wiki", "old")

    assert preview["class_id"] == "old"
    assert preview["summary"]["total_count"] == 1
    assert preview["summary"]["correct_count"] == 0
    assert preview["items"][0]["dataset_image_id"] == "utkface-002"
    assert preview["items"][0]["actual_age_display"] == "70"
    assert preview["items"][0]["predicted_class_label"] == "n/a"
    assert preview["items"][0]["predicted_age_years"] is None
    assert preview["items"][0]["correct_for_clicked_class"] is False
    assert preview["items"][0]["missed_detection"] is True
