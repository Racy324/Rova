from __future__ import annotations


def test_run_manifest_record_is_json_stable_and_contains_profile_fixture_and_request_facts() -> None:
    from evals.runtime_v1.spec import RunManifestRecord

    record = RunManifestRecord(
        experiment="context_ab",
        case_id="CM01_fixture_probe",
        profile="full",
        repeat_index=1,
        fixture_id="CM01_fixture_probe",
        fixture_sha256="a" * 64,
        eval_run_id="eval-1",
        run_id="trace-1",
        provider_requests=2,
    )

    assert record.to_dict() == {
        "schema_version": 1,
        "experiment": "context_ab",
        "case_id": "CM01_fixture_probe",
        "profile": "full",
        "repeat_index": 1,
        "fixture_id": "CM01_fixture_probe",
        "fixture_sha256": "a" * 64,
        "eval_run_id": "eval-1",
        "run_id": "trace-1",
        "provider_requests": 2,
    }
