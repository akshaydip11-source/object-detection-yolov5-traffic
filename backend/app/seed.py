"""Seed default admin, demo cameras, and sample reference data."""
from datetime import datetime, timedelta
import random

from app.config import settings
from app.db.database import SessionLocal, engine, Base
from app.db.models import User, CameraZone, Violation, DetectionJob
from app.services.auth import hash_password, get_user_by_email


CAMERAS = [
    ("CAM-01", "Howrah Bridge North", "Howrah Bridge Approach", "Kolkata", 22.5851, 88.3468),
    ("CAM-02", "Park Street Junction", "Park Street & AJC Bose", "Kolkata", 22.5522, 88.3531),
    ("CAM-03", "Science City Flyover", "EM Bypass Science City", "Kolkata", 22.5390, 88.3958),
    ("CAM-04", "Esplanade Crossing", "Esplanade Dharmatala", "Kolkata", 22.5646, 88.3509),
    ("CAM-05", "Salt Lake Sector V", "Sector V Main Gate", "Kolkata", 22.5760, 88.4330),
    ("CAM-06", "Gariahat Crossing", "Gariahat Market Junction", "Kolkata", 22.5186, 88.3665),
    ("CAM-07", "Dumdum Airport Road", "VIP Road Airport", "Kolkata", 22.6430, 88.4390),
    ("CAM-08", "New Town Circle", "New Town Action Area I", "Kolkata", 22.5790, 88.4790),
]


def seed():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        if not get_user_by_email(db, settings.default_admin_email):
            admin = User(
                email=settings.default_admin_email,
                full_name="SafeCity Admin",
                hashed_password=hash_password(settings.default_admin_password),
                role="admin",
                organization="SafeCityAI HQ",
            )
            db.add(admin)
            if settings.app_env.lower() != "production":
                db.add(User(
                    email="officer@safecity.ai",
                    full_name="Traffic Officer Roy",
                    hashed_password=hash_password("officer123"),
                    role="officer",
                    organization="Kolkata Traffic Police",
                ))
                db.add(User(
                    email="analyst@safecity.ai",
                    full_name="CV Analyst Mehta",
                    hashed_password=hash_password("analyst123"),
                    role="analyst",
                    organization="SafeCity Data Lab",
                ))
            db.commit()
            print("✓ Seeded administrator account")

        if db.query(CameraZone).count() == 0:
            for i, (cid, name, loc, city, lat, lng) in enumerate(CAMERAS):
                db.add(
                    CameraZone(
                        camera_id=cid,
                        name=name,
                        location=loc,
                        city=city,
                        lat=lat,
                        lng=lng,
                        status="online" if i < 7 else "maintenance",
                        violation_count=random.randint(0, 12),
                    )
                )
            db.commit()
            print(f"✓ Seeded {len(CAMERAS)} camera zones")
        else:
            print("• Cameras already present")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
