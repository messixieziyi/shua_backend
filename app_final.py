"""
Final Meetup Service Backend (FastAPI, Python) — Working Version
================================================================

This version includes the core meetup service schema with working relationships and sample data.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import uuid
from contextlib import asynccontextmanager
from enum import Enum
from typing import Annotated, Optional

from fastapi import Depends, FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sqlalchemy import (
    BigInteger, Boolean, DateTime, Enum as SQLEnum, ForeignKey, String, Text, 
    UniqueConstraint, func, select, create_engine, Integer, Float, Index, delete
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker
from starlette.concurrency import run_in_threadpool

# ---------------------
# Database
# ---------------------
DATABASE_URL = "sqlite:///./meetup_final.db"
engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)

class Base(DeclarativeBase):
    pass

# ---------------------
# Enums
# ---------------------
class RequestStatus(str, Enum):
    SUBMITTED = "SUBMITTED"
    ACCEPTED = "ACCEPTED"
    DECLINED = "DECLINED"

class RSVPStatus(str, Enum):
    GOING = "going"
    MAYBE = "maybe"
    NOT_GOING = "not_going"

class MessageKind(str, Enum):
    USER = "user"
    SYSTEM = "system"

# ---------------------
# Core Models
# ---------------------
class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True)
    full_name: Mapped[str] = mapped_column(String(255))
    handle: Mapped[str] = mapped_column(String(50), unique=True)
    bio: Mapped[Optional[str]] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=func.now())

    # Relationships
    created_events: Mapped[list["Event"]] = relationship(back_populates="creator")
    rsvps: Mapped[list["RSVP"]] = relationship(back_populates="user")

class Activity(Base):
    __tablename__ = "activities"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)

    events: Mapped[list["Event"]] = relationship(back_populates="activity")

class Event(Base):
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(255))
    activity_type: Mapped[str] = mapped_column(ForeignKey("activities.name"))
    description: Mapped[Optional[str]] = mapped_column(Text)
    starts_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    ends_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    capacity: Mapped[int] = mapped_column(Integer)
    created_by: Mapped[int] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=func.now())
    
    # Location fields
    location_name: Mapped[Optional[str]] = mapped_column(String(255))
    address_line1: Mapped[Optional[str]] = mapped_column(String(255))
    locality: Mapped[Optional[str]] = mapped_column(String(100))
    country_code: Mapped[Optional[str]] = mapped_column(String(2))
    lat: Mapped[Optional[float]] = mapped_column(Float)
    lon: Mapped[Optional[float]] = mapped_column(Float)

    # Relationships
    creator: Mapped[User] = relationship(back_populates="created_events")
    activity: Mapped[Activity] = relationship(back_populates="events")
    rsvps: Mapped[list["RSVP"]] = relationship(back_populates="event")

class RSVP(Base):
    __tablename__ = "rsvps"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("events.id"))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    status: Mapped[RSVPStatus] = mapped_column(SQLEnum(RSVPStatus))
    guests_count: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=func.now())

    event: Mapped[Event] = relationship(back_populates="rsvps")
    user: Mapped[User] = relationship(back_populates="rsvps")

    __table_args__ = (
        UniqueConstraint("event_id", "user_id"),
        Index("idx_rsvp_event_status", "event_id", "status"),
    )

# ---------------------
# Chat/Booking System
# ---------------------
class Request(Base):
    __tablename__ = "requests"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    event_id: Mapped[int] = mapped_column(ForeignKey("events.id"))
    guest_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    host_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    status: Mapped[RequestStatus] = mapped_column(SQLEnum(RequestStatus), default=RequestStatus.SUBMITTED)
    auto_accept: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=func.now())

    event: Mapped[Event] = relationship()
    guest: Mapped[User] = relationship(foreign_keys=[guest_id])
    host: Mapped[User] = relationship(foreign_keys=[host_id])

class Thread(Base):
    __tablename__ = "threads"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    event_id: Mapped[int] = mapped_column(ForeignKey("events.id"))
    is_locked: Mapped[bool] = mapped_column(Boolean, default=False)

class ThreadParticipant(Base):
    __tablename__ = "thread_participants"
    thread_id: Mapped[str] = mapped_column(ForeignKey("threads.id", ondelete="CASCADE"), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    role: Mapped[str] = mapped_column(String(30))

class Message(Base):
    __tablename__ = "messages"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    thread_id: Mapped[str] = mapped_column(ForeignKey("threads.id", ondelete="CASCADE"))
    sender_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"))
    client_msg_id: Mapped[str] = mapped_column(String(64))
    kind: Mapped[MessageKind] = mapped_column(SQLEnum(MessageKind), default=MessageKind.USER)
    body: Mapped[str] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=func.now())
    seq: Mapped[int] = mapped_column(BigInteger)

    __table_args__ = (
        UniqueConstraint("thread_id", "client_msg_id"),
        UniqueConstraint("thread_id", "seq"),
    )

# ---------------------
# Schemas
# ---------------------
class RequestCreate(BaseModel):
    event_id: int
    host_id: int
    auto_accept: bool = False

class RequestAction(BaseModel):
    action: str = Field(pattern="^(accept|decline)$")

class RSVPCreate(BaseModel):
    event_id: int
    status: RSVPStatus
    guests_count: int = 0

class EventCreate(BaseModel):
    title: str
    activity_type: str
    description: Optional[str] = None
    starts_at: dt.datetime
    ends_at: dt.datetime
    capacity: int
    location_name: Optional[str] = None
    address_line1: Optional[str] = None
    locality: Optional[str] = None
    country_code: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None

# ---------------------
# Dependencies
# ---------------------
def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_user_id(x_user_id: int = Header(None)):
    if not x_user_id:
        raise HTTPException(401, "Missing X-User-Id header")
    return x_user_id

# ---------------------
# Helpers
# ---------------------
def _next_seq(session, thread_id: str) -> int:
    return session.execute(select(func.coalesce(func.max(Message.seq), 0) + 1).where(Message.thread_id == thread_id)).scalar_one()

async def system_message(session, thread_id: str, body: str):
    m = Message(thread_id=thread_id, sender_id=None, client_msg_id=str(uuid.uuid4()), kind=MessageKind.SYSTEM, body=body, seq=_next_seq(session, thread_id))
    session.add(m)
    session.flush()

# ---------------------
# Lifespan
# ---------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await run_in_threadpool(Base.metadata.create_all, bind=engine)
    yield
    # Shutdown
    pass

# ---------------------
# FastAPI app
# ---------------------
app = FastAPI(title="Meetup Service API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True)

# ---------------------
# Endpoints
# ---------------------
@app.post("/requests")
async def create_request(payload: RequestCreate, user_id: int = Depends(get_user_id), session=Depends(get_session)):
    # Ensure guest and host are different
    if user_id == payload.host_id:
        raise HTTPException(400, "Guest and host cannot be the same user")
    
    req = Request(event_id=payload.event_id, guest_id=user_id, host_id=payload.host_id, auto_accept=payload.auto_accept)
    session.add(req)
    session.flush()

    thread = Thread(event_id=payload.event_id)
    session.add(thread)
    session.flush()

    session.add_all([
        ThreadParticipant(thread_id=thread.id, user_id=req.guest_id, role="guest"), 
        ThreadParticipant(thread_id=thread.id, user_id=req.host_id, role="host")
    ])
    session.flush()

    if req.auto_accept:
        req.status = RequestStatus.ACCEPTED
        await system_message(session, thread.id, "Request auto-accepted!")

    session.commit()
    return {"request_id": req.id, "thread_id": thread.id, "status": req.status}

@app.post("/requests/{request_id}/act")
async def act_on_request(request_id: str, payload: RequestAction, user_id: int = Depends(get_user_id), session=Depends(get_session)):
    req = session.execute(select(Request).where(Request.id == request_id)).scalar_one_or_none()
    if not req:
        raise HTTPException(404, "Not found")
    if user_id != req.host_id:
        raise HTTPException(403, "Forbidden")

    if payload.action == "accept":
        req.status = RequestStatus.ACCEPTED
        session.commit()
        return {"status": req.status, "message": "Request accepted"}

    if payload.action == "decline":
        req.status = RequestStatus.DECLINED
        session.commit()
        return {"status": req.status, "message": "Request declined"}

    raise HTTPException(400, "Invalid action")

@app.post("/events")
async def create_event(payload: EventCreate, user_id: int = Depends(get_user_id), session=Depends(get_session)):
    event = Event(
        title=payload.title,
        activity_type=payload.activity_type,
        description=payload.description,
        starts_at=payload.starts_at,
        ends_at=payload.ends_at,
        capacity=payload.capacity,
        created_by=user_id,
        location_name=payload.location_name,
        address_line1=payload.address_line1,
        locality=payload.locality,
        country_code=payload.country_code,
        lat=payload.lat,
        lon=payload.lon
    )
    session.add(event)
    session.flush()
    session.commit()
    return {"event_id": event.id, "title": event.title}

@app.post("/rsvps")
async def create_rsvp(payload: RSVPCreate, user_id: int = Depends(get_user_id), session=Depends(get_session)):
    rsvp = RSVP(
        event_id=payload.event_id,
        user_id=user_id,
        status=payload.status,
        guests_count=payload.guests_count
    )
    session.add(rsvp)
    session.commit()
    return {"rsvp_id": rsvp.id, "status": rsvp.status}

@app.get("/events")
async def list_events(session=Depends(get_session)):
    events = session.execute(select(Event).order_by(Event.starts_at)).scalars().all()
    return [{"id": e.id, "title": e.title, "starts_at": e.starts_at, "capacity": e.capacity} for e in events]

@app.get("/users")
async def list_users(session=Depends(get_session)):
    users = session.execute(select(User).where(User.is_active == True)).scalars().all()
    return [{"id": u.id, "full_name": u.full_name, "handle": u.handle} for u in users]

@app.post("/dev/seed")
async def seed_database(session=Depends(get_session)):
    """Seed the database with sample data"""
    
    # Clear existing data
    session.execute(delete(Message))
    session.execute(delete(ThreadParticipant))
    session.execute(delete(Thread))
    session.execute(delete(Request))
    session.execute(delete(RSVP))
    session.execute(delete(Event))
    session.execute(delete(Activity))
    session.execute(delete(User))
    session.commit()

    # Create Activities
    activities = [
        Activity(name="Tennis"),
        Activity(name="Basketball"),
        Activity(name="Soccer"),
        Activity(name="Golf"),
        Activity(name="Yoga"),
        Activity(name="Running"),
        Activity(name="Swimming"),
        Activity(name="Volleyball"),
        Activity(name="Hiking")
    ]
    session.add_all(activities)
    session.flush()

    # Create Users
    users = [
        User(email="sarah@example.com", full_name="Sarah Johnson", handle="sarah_j", bio="Tennis enthusiast and event organizer"),
        User(email="mike@example.com", full_name="Mike Chen", handle="mike_c", bio="Basketball player and coach"),
        User(email="emma@example.com", full_name="Emma Davis", handle="emma_d", bio="Yoga instructor and wellness advocate"),
        User(email="alex@example.com", full_name="Alex Rodriguez", handle="alex_r", bio="Soccer player and fitness enthusiast"),
        User(email="lisa@example.com", full_name="Lisa Wang", handle="lisa_w", bio="Rock climbing instructor"),
        User(email="david@example.com", full_name="David Brown", handle="david_b", bio="Golf pro and course designer"),
        User(email="jessica@example.com", full_name="Jessica Taylor", handle="jessica_t", bio="Running coach and marathoner"),
        User(email="ryan@example.com", full_name="Ryan Kim", handle="ryan_k", bio="Swimming instructor and triathlete"),
        User(email="sophia@example.com", full_name="Sophia Martinez", handle="sophia_m", bio="Volleyball player and team captain"),
        User(email="james@example.com", full_name="James Wilson", handle="james_w", bio="Hiking guide and outdoor enthusiast")
    ]
    session.add_all(users)
    session.flush()

    # Create Events
    now = dt.datetime.now(dt.timezone.utc)
    events = [
        Event(
            title="Morning Tennis Doubles",
            activity_type="Tennis",
            description="Join us for a fun morning of tennis doubles! All skill levels welcome.",
            starts_at=now + dt.timedelta(days=1, hours=9),
            ends_at=now + dt.timedelta(days=1, hours=11),
            capacity=8,
            created_by=users[0].id,
            location_name="Central Park Tennis Courts",
            address_line1="123 Tennis Ave",
            locality="New York",
            country_code="US",
            lat=40.7829,
            lon=-73.9654
        ),
        Event(
            title="Basketball Pickup Game",
            activity_type="Basketball",
            description="Weekly pickup basketball game. Bring your A-game!",
            starts_at=now + dt.timedelta(days=2, hours=18),
            ends_at=now + dt.timedelta(days=2, hours=20),
            capacity=20,
            created_by=users[1].id,
            location_name="Downtown Sports Complex",
            address_line1="456 Sports Blvd",
            locality="Los Angeles",
            country_code="US",
            lat=34.0522,
            lon=-118.2437
        ),
        Event(
            title="Sunset Yoga Session",
            activity_type="Yoga",
            description="Relaxing yoga session as the sun sets. All levels welcome.",
            starts_at=now + dt.timedelta(days=3, hours=18),
            ends_at=now + dt.timedelta(days=3, hours=19),
            capacity=25,
            created_by=users[2].id,
            location_name="Beach Yoga Studio",
            address_line1="789 Ocean Dr",
            locality="Miami",
            country_code="US",
            lat=25.7617,
            lon=-80.1918
        ),
        Event(
            title="Golf Tournament",
            activity_type="Golf",
            description="Annual golf tournament with prizes for top performers.",
            starts_at=now + dt.timedelta(days=5, hours=8),
            ends_at=now + dt.timedelta(days=5, hours=16),
            capacity=32,
            created_by=users[5].id,
            location_name="Championship Golf Course",
            address_line1="555 Fairway Ln",
            locality="Augusta",
            country_code="US",
            lat=33.4735,
            lon=-82.0105
        ),
        Event(
            title="Morning Run Group",
            activity_type="Running",
            description="Join our weekly morning run group. All paces welcome!",
            starts_at=now + dt.timedelta(days=6, hours=7),
            ends_at=now + dt.timedelta(days=6, hours=8),
            capacity=30,
            created_by=users[6].id,
            location_name="Riverside Park",
            address_line1="100 River Rd",
            locality="Portland",
            country_code="US",
            lat=45.5152,
            lon=-122.6784
        )
    ]
    session.add_all(events)
    session.flush()

    # Create RSVPs
    rsvps = [
        RSVP(event_id=events[0].id, user_id=users[1].id, status=RSVPStatus.GOING, guests_count=0),
        RSVP(event_id=events[0].id, user_id=users[2].id, status=RSVPStatus.MAYBE, guests_count=1),
        RSVP(event_id=events[1].id, user_id=users[0].id, status=RSVPStatus.GOING, guests_count=0),
        RSVP(event_id=events[1].id, user_id=users[2].id, status=RSVPStatus.GOING, guests_count=0),
        RSVP(event_id=events[2].id, user_id=users[0].id, status=RSVPStatus.GOING, guests_count=0),
        RSVP(event_id=events[2].id, user_id=users[1].id, status=RSVPStatus.MAYBE, guests_count=0),
    ]
    session.add_all(rsvps)

    session.commit()
    
    return {
        "message": "Database seeded successfully",
        "users_created": len(users),
        "events_created": len(events),
        "activities_created": len(activities),
        "rsvps_created": len(rsvps)
    }

# ---------------------
# Smoke tests
# ---------------------
async def _run_smoke_tests():
    from asgi_lifespan import LifespanManager
    from httpx import AsyncClient

    async with LifespanManager(app):
        async with AsyncClient(app=app, base_url="http://testserver") as client:
            # Seed database
            r = await client.post("/dev/seed")
            r.raise_for_status()
            print("✅ Database seeded")

            # Test creating a request (user 2 requesting to join event 1 hosted by user 1)
            r = await client.post("/requests", headers={"X-User-Id": "2"}, json={"event_id": 1, "host_id": 1, "auto_accept": False})
            r.raise_for_status()
            data = r.json()
            print(f"✅ Request created: {data}")

            # Test listing events
            r = await client.get("/events")
            r.raise_for_status()
            events = r.json()
            print(f"✅ Found {len(events)} events")

            # Test creating RSVP
            r = await client.post("/rsvps", headers={"X-User-Id": "3"}, json={"event_id": 3, "status": "going", "guests_count": 0})
            r.raise_for_status()
            print("✅ RSVP created")

            print("✅ All smoke tests passed!")

if __name__ == "__main__":
    import argparse, uvicorn
    p = argparse.ArgumentParser()
    p.add_argument("--test", action="store_true")
    a = p.parse_args()
    if a.test:
        asyncio.run(_run_smoke_tests())
    else:
        uvicorn.run("app_final:app", host="0.0.0.0", port=8000, reload=False)
