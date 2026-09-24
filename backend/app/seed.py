"""Seed default admin, demo cameras, and sample reference data."""

from backend.app.config import settings
from backend.app.db.database import SessionLocal, engine, Base
from backend.app.db.models import User, CameraZone
from backend.app.services.auth import hash_password, get_user_by_email


CAMERAS = [
    (
        "CAM-01",
        "Howrah Bridge North",
        "Howrah Bridge Approach",
        "Kolkata",
        22.5851,
        88.3468,
    ),
    (
        "CAM-02",
        "Park Street Junction",
        "Park Street & AJC Bose",
        "Kolkata",
        22.5522,
        88.3531,
    ),
    (
        "CAM-03",
        "Science City Flyover",
        "EM Bypass Science City",
        "Kolkata",
        22.5390,
        88.3958,
    ),
    (
        "CAM-04",
        "Esplanade Crossing",
        "Esplanade Dharmatala",
        "Kolkata",
        22.5646,
        88.3509,
    ),
    ("CAM-05", "Salt Lake Sector V", "Sector V Main Gate", "Kolkata", 22.5760, 88.4330),
    (
        "CAM-06",
        "Gariahat Crossing",
        "Gariahat Market Junction",
        "Kolkata",
        22.5186,
        88.3665,
    ),
    ("CAM-07", "Dumdum Airport Road", "VIP Road Airport", "Kolkata", 22.6430, 88.4390),
    (
        "CAM-08",
        "New Town Circle",
        "New Town Action Area I",
        "Kolkata",
        22.5790,
        88.4790,
    ),
]


def seed():
    if not settings.demo_mode:
        return
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        for email, password, name, role in (
            ("admin@safecity.ai", "admin123", "Demo Admin", "admin"),
            ("officer@safecity.ai", "officer123", "Demo Officer", "officer"),
            ("analyst@safecity.ai", "analyst123", "Demo Analyst", "analyst"),
        ):
            if not get_user_by_email(db, email):
                db.add(
                    User(
                        email=email,
                        full_name=name,
                        hashed_password=hash_password(password),
                        role=role,
                        organization="SafeCityAI Demo",
                    )
                )
        db.commit()

        if db.query(CameraZone).count() == 0:
            for cid, name, loc, city, lat, lng in CAMERAS:
                db.add(
                    CameraZone(
                        camera_id=cid,
                        name=name,
                        location=loc,
                        city=city,
                        lat=lat,
                        lng=lng,
                        status="demo",
                        violation_count=0,
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
