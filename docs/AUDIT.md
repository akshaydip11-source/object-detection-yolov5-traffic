> **Follow-up:** The annotation failure was investigated against the newer GitHub
> commit `8159a35` and its Render deployment. The corrected code now passes **90
> tests, including all 9 real-browser tests**, with no skips. See
> [the current investigation](ANNOTATION_FIX.md) for the latest evidence and changes.
> Browser limitations/counts below describe the earlier audit, not the current run.
> The original custom checkpoint/data were not supplied; the new public-data pilot
> is pending and no deployment has been performed. Draft PR #1 is published and application CI at
> `41c1c9a` passed, including Docker/container/browser checks. Remaining statements
> below about unpublished code and unavailable checks are historical.

# Repository and deployment-preparation audit — 2026-09-24

## Verdict

**Application changes and local checks completed; final deployment sign-off is
still blocked by the missing trained checkpoint and unexecuted host/container checks.**
Nothing has been deployed, committed or pushed in this session. Work remains on
`arena/01a0cfd1-object-detection-yolov5-traffi`.

No ZIP, trusted custom `best.pt`, original Postman-tested backend or training results
were supplied. Workspace, repository history and GitHub releases were checked; no
checkpoint/release artifact was found. Do not claim that the new adapter is the
previously tested backend or that an untrained test fixture is the internship model.

## Work completed

### Model and application correctness

- Removed old YOLO11 ONNX runtime/weights and unsupported COCO violation heuristics.
- Added local YOLOv5 `.pt` loading; verify exactly Helmet, NoHelmet and LicensePlate,
  using checkpoint class order. No model download/fallback or runtime package install.
- Corrected backend package imports and consistent root launch commands. UI paths,
  API documentation and deployment entry point are exercised by tests.
- Only learned NoHelmet predictions create review flags. No invented OCR, seatbelt,
  triple-riding, signal, speed, unique-offender or model-performance claims.
- Image/video decode guards, checked writes and video handle cleanup. Video counts
  are per-frame observations; automatic video ticket generation is disabled.
- Bounded multipart/body copies, pixel/frame limits, validated thresholds, a
  cooperative video processing timeout and one-job inference admission (503 on busy).
- Corrected YOLOv5 dataset path resolution and training helper preconditions.

### Access control and privacy

- All operational inference/data APIs require valid sign-in. Invalid upload credentials
  are rejected before multipart spooling. Public endpoints are landing/static UI,
  API metadata/docs, auth options/login/optional registration and health.
- Analysts only see their own jobs. Officers/admins can review all jobs and records;
  dashboard/tickets/PDF/CSV require officer/admin access; user listing is admin-only.
- Removed public `/media` static mount. Results use expiring, path-scoped signed URLs
  with active-user and ownership checks. Media tokens cannot serve as access tokens.
- Registration disabled by default; when enabled it cannot self-grant privileged roles.
  Demo seeding is opt-in and forbidden in production. Production also rejects a DB
  containing the publicly documented demo identities.
- Environment-driven signing key. Created a private, ignored local `.env` with a
  random secret; never printed/stored it in documentation. Production rejects a
  missing/placeholder/short configured secret and fails startup without valid weights.
- Session-scoped browser token storage, signed-link-aware previews, authenticated
  PDF/CSV downloads, HTML escaping and CSV formula protection. PDFs are explicitly
  illustrative review records and wrap/escape user-entered text.
- Bounded in-process request limiting and response security headers. Added administrative
  account creation/disable commands; password prompts/errors do not log the password.

### Deployment, source hygiene and documentation

- Added non-root CPU Docker configuration, readiness check, read-only-root Compose
  service, resource limits, persistent volumes, Caddy HTTPS configuration and a runbook.
- Added a retention CLI with non-destructive dry-run default and explicit `--apply`.
- Added real-checkpoint verification CLI that checks loading/inference and writes an
  ignored SHA-256/annotation report; it exits nonzero for the missing checkpoint.
- Removed frontend photographs with incomplete provenance. Replaced them with three
  reproducible synthetic illustrations from the project script; confirmed checksums
  match across separate runs. They are not validation/training evidence.
- Extended `.gitignore` and added `.dockerignore` to exclude secrets, checkpoints,
  archives, private dataset contents, DB sidecars and generated media/cache files.
  Directory placeholders and configuration examples remain included.
- Updated README, model instructions, training guide, deployment guide and third-party
  notices. Upstream vendored YOLOv5/license remains unchanged; no blanket relicensing
  rights are invented. Owner must ensure the combined distribution meets its licensing
  obligations. Removed files remain in Git history; no history rewrite was performed.
- Added GitHub CI for application/security tests, Docker build/container smoke and real
  browser tests. **The new workflow has not run on GitHub for these unpublished changes.**

## Checks actually executed

| Check | Result |
| --- | --- |
| `python -m pytest -q` | **46 passed, 2 skipped** |
| Actual local YOLOv5 loader + CPU forward pass | Passed with a temporary, randomly initialized three-class network |
| Authenticated image API + signed annotated-media fetch using that actual network | Passed; still not a trained-model accuracy/compatibility result |
| Stub-prediction image/frame/video contracts, real OpenCV IO/drawing | Passed |
| Missing weights and production startup failure | Passed |
| Private endpoints, cross-user job isolation, officer permissions | Passed |
| Signed media expiry/path scope, access-token separation, disabled accounts | Passed |
| Upload/auth/body/pixel limits, inference overload, role escalation prevention | Passed |
| Account disable and CLI password-error redaction | Passed |
| Retention dry-run/apply, PDF/CSV and formula escaping | Passed |
| Python lint/compile, shell syntax, CLI help | Passed |
| All inline and shared frontend JavaScript syntax | Passed |
| Installed dependency consistency (`pip check`) | Passed |
| Git exclusions and `git diff --check` | Passed |
| YAML syntax, documentation links and frontend sample references | Passed |
| Deterministic synthetic illustration regeneration | Passed |
| Common private-key/token pattern scan of publishable files | No matches; limited scan, not a full secret-history/security audit |
| Real checkpoint verification command | Correctly exits nonzero: **trusted model missing** |

**Two browser tests are skipped**, not passed. Playwright installed, but Chromium
could not be downloaded (`cdn.playwright.dev` connection reset). Docker was absent;
a static Docker download attempt failed (`download.docker.com` TLS connection failure).
Therefore local Docker/Compose/Caddy build/runtime validation and browser rendering/
interaction/webcam validation cannot be claimed. YAML syntax checks do not replace
`docker compose config` or a real container run.

Test environment: Python 3.11.2, PyTorch 2.14.0, torchvision 0.29.0, OpenCV 4.14.0,
Ultralytics 8.4.160. Local sandbox tests use an additional headless OpenCV wheel
because libGL could not be installed through the sandbox's blocked apt downloads.
Docker/normal Linux instructions install libGL for the declared regular OpenCV
runtime; that separate container environment has not been executed here.
One upstream TestClient/httpx deprecation warning remains; tests pass.

## What still requires an external artifact or deployment host

1. **Supply your trusted custom `best.pt` and preferably the known-good backend,
   Postman collection, training classes and exact working dependency versions.**
2. Run real Helmet/NoHelmet/LicensePlate image cases, negative/blank cases and a video;
   compare boxes/classes/confidence with the known-good implementation. Record actual
   held-out metrics and checkpoint SHA-256, not UI mock scores or fixture outputs.
3. On a Docker-capable host/CI, build and run container/browser checks, then repeat
   with the actual checkpoint. Resolve any failures before sign-off.
4. Select the deployment host/domain, configure DNS/TLS/storage and provision private
   accounts. Verify HTTPS, permissions, signed media, PDF/CSV, video preview/webcam,
   backup/restore and capacity on that host. No hosting account/domain was provided.
5. Review final Git diff and licensing/data permissions before committing/publishing.
   Keep `.env`, private footage, DB files and large checkpoints out of Git.

## Deliberate limits

This remains a single-worker internship prototype, not a certified enforcement or
high-load hostile-workload service. The video deadline cannot interrupt a native
codec/model call that hangs. In-process limits are not distributed; the gateway's
peer IP may be shared. There is no durable job queue, automatic retention schedule,
full penetration test, automated migration system or strict nonce-based frontend
CSP. Signed URLs are temporary bearer credentials; logout does not revoke already
issued access tokens. See the deployment runbook for operational boundaries.
