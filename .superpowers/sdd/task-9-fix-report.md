# Task 9 fix report — import-run review findings

Branch: `worktree-agent-adb78d6819305fe49` (feature base `phase-4e-1`)
Scope: `src/eva_agent/packages/manifest.py`, `src/eva_agent/packages/run_bundle.py`,
`tests/unit/packages/test_manifest.py`, `tests/unit/packages/test_run_bundle.py`.
The export staging code in `build_run_bundle` was not touched; `_run_document` changed only for F3.

All three Important findings (F1, F2, F3) and all five Minor items (M6, M7, M11, M12, M8/M9/M10)
are fixed. TDD was used: the new tests were run first and produced 13 failures, then the
implementation made them green.

## F1 — `PackageError` messages leaked absolute host paths

Changed `manifest.py`:

- Added module-level `_reason(exc)` (`OSError` → `strerror` or class name, otherwise `str(exc)`
  or class name) exactly as specified.
- Used it in `read_manifest` (OSError/ValueError branch), both `verify_tree` stat/hash branches,
  both `write_zip` stat/write branches, `_open_zip`, and the `_extract_zip_into` catch-all.
- Replaced the `ValidationError` branch's `str(exc)` echo (which embeds Pydantic `input_value`,
  e.g. a hostile absolute path) with a `loc: msg` summary.

Tests added in `test_manifest.py`:

- `test_missing_manifest_error_does_not_leak_the_host_path`
- `test_missing_archive_error_does_not_leak_the_host_path`
- `test_invalid_manifest_error_does_not_echo_hostile_values`
- `test_write_zip_error_does_not_leak_the_destination_path` (requested mid-task by the
  coordinator: injects `PermissionError(errno.EACCES, "Permission denied", <abs path>)` from
  `zipfile.ZipFile`; asserts "Permission denied" is present and neither the absolute path nor a
  drive letter appears)

The shared `_assert_path_free` helper rejects a `<letter>:\` drive path, the
`eva-import-`/`eva-bundle-` temp prefixes, and `Traceback`.

## F2 — half-imported run reported `skipped` on retry

Changed `run_bundle.py`:

- `_storage` now yields the session factory first:
  `(factory, runs, reports, artifacts)`.
- Added `_discard_import(factory, config, run_id)`: deletes artifact, report and
  event-counter rows for the run, deletes the run row, then `shutil.rmtree`s
  `run_paths(...).root`. It runs only when this call created the run.
- The write block sets `created = False` before `runs.create`, and wraps the rest
  (`set_plan`, `reports.upsert`, `_write_artifacts`, `set_status`, `set_public`) in
  `except BaseException`; on failure it performs best-effort cleanup (never masking the
  original exception) and re-raises.
- The `IntegrityError` path (M6) is caught around `runs.create` only and returns
  `skipped` without compensation.

Test added: `test_failed_import_rolls_back_and_a_retry_imports` monkeypatches
`RunRepository.set_plan` to fail once; asserts no `RunRow`, no `ReportRow`, no artifact or
event-counter rows and no run directory after the failure, then re-imports and asserts
`status == "imported"`.

## F3 — DATASET bundles did not round-trip

Export (`_run_document`): added `_dataset_basename(payload)` (basename only; rejects empty,
`.`, `..` and any name containing `/` or `\`). After sanitizing the input, when the raw
stored input has a usable dataset basename and the sanitized input still has a dataset
object, the basename is re-attached as `dataset.path`. Added the required comment explaining
that the sanitizer drops every `path` key and that the basename is required by `RunInput`
and is not an absolute path.

Import: added `_validate_dataset_path(run_input)`; it rejects a non-object `dataset` and any
`dataset.path` that is not a plain file name (`C:\secret\x.csv`, `a/b`, `.`, `..`, ...).
It is called on the raw document input *before* `_safe_input`, so the existing basename
reduction cannot launder a hostile path; `RunInput.model_validate` follows it.

Tests added:

- `test_dataset_bundle_round_trips_with_a_basename` (built from a `RunRow` whose
  `input_json` holds an absolute `DatasetRef.path`; asserts the exported `run.json` carries
  `data.csv`, then imports and asserts the stored `input_json["dataset"]["path"]`).
- `test_import_rejects_an_absolute_dataset_path` (rewrites `dataset.path` to
  `C:\secret\x.csv`, asserts `PackageError` and that nothing was written).

## M6 — concurrent import race

`runs.create` is wrapped in `try/except IntegrityError`, returning
`ImportResult(status="skipped")` and deliberately skipping compensation.
Test: `test_import_treats_a_concurrent_create_race_as_skipped` (monkeypatched `create`
raises `IntegrityError`).

## M7 — unbounded `report_version`

`_report_version` now accepts only `0 <= value <= 2**31 - 1`; the message is
"report_version must be a non-negative integer below 2^31".
Test: `test_import_rejects_an_out_of_range_report_version` (rewrites document and manifest
to `2**31`; asserts `PackageError` and no writes).

## M11 — artifacts container must be a list

`_restore_artifacts` now reads `raw = document.get("artifacts")` and requires
`isinstance(raw, list)`, so a missing key, `{}`, or `""` is rejected and `[]` is accepted.
Test: `test_import_rejects_a_non_list_artifacts_container`.

## M12 — artifact names that shadow run files

Added `_RUN_FILE_NAMES` (`report.json`, `report.md`, `plan.draft.yaml`,
`plan.approved.yaml`, `target_probe.jsonl`) and reject those names, case-insensitively,
in `_artifact_name` with `PackageError("artifact name shadows a run file")`.
Tests: parametrized `test_import_rejects_artifact_names_that_shadow_run_files` for
`report.json` and `REPORT.JSON`.

## M8/M9/M10 — test strength

- M8 `test_import_does_not_overwrite_an_existing_run`: after the first import the stored
  report payload is changed and the run status is set to `FAILED`; a second import returns
  `skipped` and both changes survive.
- M9 `test_import_rejects_a_traversal_artifact_name_without_writing` and
  `test_import_rejects_a_missing_artifact_member_without_writing`: assert no run/report/
  artifact/event-counter rows and no `data_dir/runs/<run_id>` directory. (The rejection tests
  for M7/M11/M12/F3 use the same `_assert_nothing_written` helper.)
- M10 `test_import_with_public_false_keeps_the_run_private`: asserts `is_public is False`
  and `published_at is None` after a `--private` import.

Helpers added in `test_run_bundle.py`: `_completed_dataset_run`, `_archive_for_completed_run`,
`_archive_for_dataset_run`, `_rewrite_run_document` (re-extracts a bundle, mutates `run.json`,
rehashes the member, optionally rewrites the manifest `report_version`, repacks), and
`_assert_nothing_written`. `_reset_run` now also removes `run_event_counters` (explicitly) and
the run directory so "nothing written" assertions are meaningful.

## Deviations from the findings file (and why)

1. **F2 cleanup uses `contextlib.suppress(Exception)`** instead of `try/except/pass`.
   Ruff's `SIM105` rejects the latter; behavior is identical and the "never mask the original
   failure" comment is retained.
2. **Existing `test_write_zip_translates_a_write_failure` was adjusted** to raise
   `OSError(errno.ENOSPC, "disk full")` rather than `OSError("disk full")`. A bare
   `OSError("...")` has no `strerror`, so the specified `_reason` reports only "OSError";
   a real OS failure always carries `errno`/`strerror`, so the test now models one and keeps
   matching "disk full". This is the "match regex relied on the old tail" adjustment the
   findings file permits.
3. **`_dataset_basename` reads the raw stored `input_json`**, not the sanitized copy,
   because sanitization has already removed `path`. The basename is re-attached to the
   sanitized copy. This is the only ordering freedom in the F3 snippet.
4. `_validate_dataset_path` is invoked before `_safe_input` (not just before
   `RunInput.model_validate`) so a hostile path is rejected rather than reduced to a
   basename first.

## Commands and results

- `UV_LINK_MODE=copy uv sync` → exit 0.
- RED check: `uv run pytest tests/unit/packages/test_run_bundle.py tests/unit/packages/test_manifest.py -q`
  → 13 failed, 56 passed (the 13 new behaviour tests failed as intended).
- `uv run pytest tests/unit/packages/test_run_bundle.py tests/unit/packages/test_manifest.py -q`
  → 70 passed, 1 warning (a pre-existing Pydantic field-name warning).
- `uv run pytest -q` → **1468 passed, 3 skipped, 1 deselected, 1 warning**.
- `uv run ruff check src tests` → **All checks passed!**
- `uv run pyright` → **0 errors, 0 warnings, 0 informations**.

## Not fixed / remaining concerns

None within the findings. Two notes for the reviewer:

- M12 only guards import; a public artifact literally named e.g. `report.json` can still be
  exported (the bundle layout is unchanged, per the constraints) but such a bundle is now
  rejected on import with a clear message rather than silently overwriting a run file.
- The `suppress(Exception)` compensation is best effort by design; if the database itself is
  unavailable the partial run may survive, and the retry will then report `skipped`. This
  matches the finding's intended best-effort semantics.
