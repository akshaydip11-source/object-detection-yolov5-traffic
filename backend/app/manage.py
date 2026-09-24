"""Administrative commands. Run as: python -m backend.app.manage --help."""

import argparse
from datetime import datetime, timedelta
from getpass import getpass
from pathlib import Path

from sqlalchemy import delete
from pydantic import ValidationError

from backend.app.config import settings
from backend.app.db.database import Base, engine, SessionLocal
from backend.app.db.models import AuditLog, DetectionJob, User, Violation
from backend.app.models.schemas import UserCreate
from backend.app.services.auth import hash_password


def cleanup(days: int, apply: bool = False) -> dict:
    """Delete completed/failed old jobs, linked tickets/media and old audit records.

    Explicit --apply required. Run with the app stopped; back up the DB/media first.
    Never follows paths outside UPLOAD_DIR. Includes orphaned uploads by file mtime.
    """
    if days < 1:
        raise ValueError("Retention must be at least one day")
    cutoff = datetime.utcnow() - timedelta(days=days)
    root = settings.upload_dir.resolve()
    with SessionLocal() as db:
        jobs = (
            db.query(DetectionJob)
            .filter(
                DetectionJob.created_at < cutoff, DetectionJob.status != "processing"
            )
            .all()
        )
        files = set()
        ids = [job.id for job in jobs]
        for job in jobs:
            for value in (job.input_path, job.result_path):
                if value:
                    path = Path(value).resolve()
                    if path.is_relative_to(root):
                        files.add(path)
                        if path.suffix == ".mp4":
                            files.add(path.with_suffix(".jpg"))
        protected = set()
        for job in db.query(DetectionJob).filter(~DetectionJob.id.in_(ids)).all():
            for value in (job.input_path, job.result_path):
                if value:
                    path = Path(value).resolve()
                    protected.add(path)
                    protected.add(path.with_suffix(".jpg"))
        for path in root.rglob("*"):
            if (
                path.is_file()
                and path.name != ".gitkeep"
                and not path.is_symlink()
                and path.resolve().is_relative_to(root)
                and path.resolve() not in protected
                and datetime.utcfromtimestamp(path.stat().st_mtime) < cutoff
            ):
                files.add(path.resolve())
        files = {path for path in files if path.is_file() and path not in protected}
        audit_count = db.query(AuditLog).filter(AuditLog.created_at < cutoff).count()
        if apply:
            for path in files:
                path.unlink()
            if ids:
                db.execute(delete(Violation).where(Violation.job_id.in_(ids)))
                db.execute(delete(DetectionJob).where(DetectionJob.id.in_(ids)))
            db.execute(delete(AuditLog).where(AuditLog.created_at < cutoff))
            db.commit()
        return {
            "jobs": len(ids),
            "files": len(files),
            "audit_records": audit_count,
            "applied": apply,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    user = sub.add_parser(
        "create-user",
        help="Create an account; password is prompted, never a command argument",
    )
    user.add_argument("--email", required=True)
    user.add_argument("--name", required=True)
    user.add_argument(
        "--role", choices=["admin", "officer", "analyst"], default="analyst"
    )
    disable = sub.add_parser(
        "disable-user", help="Revoke an account and its signed-media access"
    )
    disable.add_argument("--email", required=True)
    prune = sub.add_parser(
        "cleanup", help="Preview retention cleanup (stop app/back up before --apply)"
    )
    prune.add_argument("--days", type=int, default=30)
    prune.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    Base.metadata.create_all(bind=engine)
    if args.command == "cleanup":
        print(cleanup(args.days, args.apply))
        return
    if args.command == "disable-user":
        with SessionLocal() as db:
            account = (
                db.query(User).filter(User.email == args.email.lower().strip()).first()
            )
            if account is None:
                parser.error("Account not found")
            account.is_active = False
            db.commit()
        print("Account disabled. Existing access tokens and signed links are rejected.")
        return
    if settings.environment == "production" and args.email.lower() in {
        "admin@safecity.ai",
        "officer@safecity.ai",
        "analyst@safecity.ai",
    }:
        parser.error("Public demo identities are forbidden in production")
    password = getpass("Password (12+ characters recommended): ")
    if password != getpass("Confirm password: "):
        parser.error("Passwords do not match")
    try:
        payload = UserCreate(email=args.email, full_name=args.name, password=password)
    except ValidationError:
        # Never print a Pydantic error containing the plaintext password input.
        parser.error(
            "Invalid email/name/password. Use a valid email, nonempty name and 8–72 UTF-8 password bytes."
        )
    with SessionLocal() as db:
        email = str(payload.email).lower()
        if db.query(User).filter(User.email == email).first():
            parser.error("Account already exists; not changing its password or role")
        db.add(
            User(
                email=email,
                full_name=payload.full_name,
                role=args.role,
                organization=payload.organization,
                hashed_password=hash_password(password),
            )
        )
        db.commit()
    print(f"Created {args.role} account {email}. No password was logged.")


if __name__ == "__main__":
    main()
