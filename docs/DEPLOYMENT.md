# Deployment runbook

## Release gates

Do not call the application deployment-ready until all applicable checks pass:

- Trusted custom `best.pt`, known training class metadata and checkpoint SHA-256.
- Real image/video inference and visual comparison with the previously tested backend.
- Held-out validation metrics and permission to use/distribute the dataset/weights.
- Docker build, container smoke, browser flow and host-specific smoke tests.
- Strong secret, private accounts, persistent storage, backups, TLS and capacity checks.

The current sandbox cannot supply the missing trained artifact. Docker could not
be installed here; Chromium was obtained via an alternate package source and the
browser suite passed. GitHub application CI at `41c1c9a` subsequently passed the
Docker build, container smoke and browser checks. No real-host deployment or trained-model
accuracy sign-off is claimed; see [the current investigation](ANNOTATION_FIX.md).

## Docker + Compose

Requires Docker Engine and Compose v2 on a Linux host. Recommended starting budget:
2 CPU cores, 3 GB application memory **as a starting limit, not a measured guarantee**;
allow additional disk/RAM for image build and dependencies. Tune after testing the
actual checkpoint. CPU builds install PyTorch from its CPU wheel index first.

```bash
python scripts/setup_env.py      # skip if an existing private .env is configured
# Put your trusted model at models/best.pt.
# In .env, set SITE_ADDRESS=https://traffic.your-domain.example.
# Point DNS at the host and allow incoming ports 80 and 443.
docker compose build
# Create the first account before launching the web service (password prompt):
docker compose run --rm --no-deps app python -m backend.app.manage create-user \
  --email you@example.com --name "Project Admin" --role admin
docker compose up -d
docker compose ps
```

PowerShell: run commands on one line or replace shell `\` continuations with backticks.
`SITE_ADDRESS=http://localhost` is only for local testing, not public use. With a real
HTTPS hostname, Caddy obtains/renews certificates; this needs correct DNS and network
access. Do not expose the application port directly to bypass the gateway.

Compose forces `ENVIRONMENT=production`, disables demo seeding and uses explicit
container paths. The application runs as UID 10001, with a read-only root filesystem,
dropped capabilities, limited processes/resources and writable named data volumes.
The model is mounted read-only. `/tmp` is an ephemeral 256 MB mount; large multipart
spools can fill it under concurrent uploads, so keep proxy limits/rates conservative.
Increase only after capacity testing. The app refuses to start with missing/bad
weights, a missing/placeholder secret or known demo-account identities in its DB.

Named volumes:
- `app-data`: SQLite database
- `app-uploads`: uploaded originals, annotated results and reports
- `caddy-data` / `caddy-config`: TLS state/configuration

**Do not use `docker compose down -v` unless intentionally destroying all data.**
Back up the database and upload volume together while the app is stopped; test a
restore on a separate instance. A Docker image is not a database backup.

Compose uses a single app worker. SQLite and the in-process admission controls are
not a distributed architecture. Rate limits use the ASGI peer IP; behind Caddy with
the default Uvicorn trust settings this is a shared gateway limit. Do not solve this
by blindly trusting arbitrary forwarded headers. Configure a trusted-proxy allowlist
and an edge rate limiter if you need accurate per-client limits at scale.

## Validate on the host

```bash
docker compose exec app python -m scripts.verify_release --help
curl -f https://traffic.your-domain.example/api/health
```

Health must report `model_loaded: true`; check status, not just HTTP 200. Log in with
your provisioned account, upload representative image and video cases, inspect class
names/boxes and signed media, then check an unauthenticated request to `/api/jobs`
returns 401. Analysts must not access other accounts' jobs or officer-only records.
Verify PDF/CSV downloads, session expiry, HTTPS webcam permission and video preview.
Run real-checkpoint verification locally or mount your licensed evaluation samples
read-only into a one-shot container; the default image intentionally contains no
private dataset. Archive the actual environment (`pip freeze`), checkpoint checksum,
commands, test reports and measured metrics with the internship handover.

For infrastructure CI only, after building `safecity-ci`:

```bash
python -m tests.container_smoke
```

This runs a disposable read-only non-root container with a **random temporary
YOLOv5n fixture**, signs in, performs real inference and fetches private media.
It proves plumbing only, not compatibility or accuracy of the user's checkpoint.

## Retention and account administration

```bash
# Additional accounts (password is prompted):
docker compose exec app python -m backend.app.manage create-user \
  --email officer@your-domain.example --name "Officer" --role officer

# Stop processing and back up both volumes before cleanup:
docker compose stop app
docker compose run --rm --no-deps app python -m backend.app.manage cleanup --days 30
# Inspect the dry-run counts first; --apply is destructive:
docker compose run --rm --no-deps app python -m backend.app.manage cleanup --days 30 --apply
docker compose up -d
```

Cleanup removes old completed/failed jobs, their linked review tickets/media, old
audit rows and orphaned uploads. `.gitkeep` and files outside the configured upload
root are protected. In-progress jobs are excluded; investigate stale processing
records after crashes. Choose retention according to project/data obligations;
there is no automatic deletion schedule enabled by default.

Changing `SECRET_KEY` invalidates access tokens and media links. Logout only clears
the browser token. For immediate account revocation, run
`python -m backend.app.manage disable-user --email user@your-domain.example`
inside the app container; a dedicated account-management UI is not included.
Do not reuse the publicly documented demo identities in a production DB.

## Remaining architectural limits

- Inference is synchronous with a one-job admission limit (503 on overload), not a
  persistent background queue. Video timeouts are checked between frames; they cannot
  interrupt a native codec/model call that hangs. Use isolated worker processes and
  a durable queue before accepting hostile public workloads.
- Decode/pixel/body limits reduce resource risk but do not replace sandboxing,
  dependency vulnerability review, rate limiting at the edge or a penetration test.
- Signed media URLs grant access to anyone holding the link until expiry; do not log
  or share them. Application/gateway access logs are off by default for this reason.
- Browser tokens remain readable by same-origin JavaScript in sessionStorage. The
  legacy static UI uses inline scripts, so the gateway CSP allows them; migrating
  to nonce/hash-based scripts or HttpOnly-cookie auth is further hardening work.
- No payment integration, legally enforceable fines, identity tracking, OCR or
  guaranteed model accuracy. Human review and appropriate authorization are required.


## Managed deployment / existing Render service

The GitHub-linked service currently runs the legacy ONNX fallback; changes in this
workspace do not automatically update it. `render.yaml` now describes a separate
reviewed deployment of the working branch with a **paid standard service and persistent
disk**, not the old free/ephemeral setup. No paid resources were created. Review cost,
region and capacity before importing the Blueprint. The branch must be published first.

Configure your own trusted artifact:

- `MODEL_URL`: HTTPS URL accessible by the host (a private signed URL is acceptable).
- `MODEL_SHA256`: 64-character SHA-256 computed from the trusted file locally.
- `MODEL_PATH`: `/app/data/models/best.pt` on the persistent disk.

Never put signed URLs in Git or logs. Existing matching weights skip the download;
new downloads have byte/time limits, require the exact checksum, and replace weights
atomically only after verification. There is no default/pretrained fallback URL.
A checksum confirms integrity, not model safety, provenance, training or accuracy.
Only load trusted PyTorch checkpoints. Expiring URLs must remain valid for the first
download or a future model update. Production startup fails if provisioning/load fails.

The disk must be writable by UID 10001. Database/uploads share `/app/data`; the health
check uses `/api/ready`, which returns 503 unless the custom checkpoint loads. After
startup, use the managed host's private shell to run `backend.app.manage create-user`;
there is no auto-created production admin with public credentials. Use a fresh DB
rather than silently carrying over legacy heuristic records/demo accounts. Render's
platform provides TLS; Caddy is for the standalone Compose alternative.

`PORT` is honored by the Python launcher and Docker health check. Real checkpoint
inference, host permissions, TLS and persistence still need a deployment smoke test.
