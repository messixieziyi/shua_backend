"""
Meetup Chat & Booking Backend (FastAPI, Python) — Auto-accept enabled, SQLAlchemy Mapped fix
===========================================================================================

✅ This revision fixes two issues:
1. **Auto-accept behavior**: if a request is created for an event with `auto_accept=True`, the backend immediately creates a confirmed booking and upgrades the thread to `BOOKING` scope (no grace period).
2. **SQLAlchemy typing bug**: ensured that `Mapped` is imported from `sqlalchemy.orm` (requires SQLAlchemy 2.0+). Added a note in the header that `pip install "sqlalchemy>=2.0"` is required.

Other notes:
- Keeps synchronous SQLite engine (no `aiosqlite`).
- Smoke tests now include auto-accept path.

Quick start:
  pip install "fastapi>=0.111" "uvicorn[standard]>=0.25" "sqlalchemy>=2.0" "pydantic>=2" httpx asgi-lifespan
  python app.py --test
  uvicorn app:app --reload
"""

from __future__ import annotations

# ---------- SSL pre-flight ----------
try:
    import ssl  # noqa: F401
except Exception:
    raise SystemExit("Python 'ssl' module missing. Install OpenSSL and rebuild Python.")

import asyncio
import datetime as dt
import uuid
from contextlib import asynccontextmanager
from enum import Enum
from typing import Annotated, Optional, Sequence

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocketState
from pydantic import BaseModel, Field

from sqlalchemy import (
    BigInteger, Boolean, DateTime, Enum as SQLEnum, ForeignKey, String, Text, UniqueConstraint, func, select, create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker
from sqlalchemy.exc import IntegrityError
from starlette.concurrency import run_in_threadpool

@app.get("/users")
async def list_users(session=Depends(get_session)):
    """Get all users."""
    users = session.execute(select(User)).scalars().all()
    return [{"id": u.id, "display_name": u.display_name} for u in users]

@app.get("/events")
async def list_events(session=Depends(get_session)):
    """Get all events (using event_id from requests)."""
    # Get unique event_ids from requests
    event_ids = session.execute(select(Request.event_id).distinct()).scalars().all()
    events = []
    for event_id in event_ids:
        # Count requests for this event
        request_count = session.execute(select(func.count(Request.id)).where(Request.event_id == event_id)).scalar()
        events.append({
            "id": event_id,
            "title": f"Event {event_id[:8]}",
            "description": f"Meetup event with {request_count} requests",
            "capacity": 10,
            "starts_at": "2024-01-01T10:00:00Z"
        })
    return events

@app.post("/rsvps")
async def create_rsvp(event_id: str, user_id: str = Depends(get_user_id), session=Depends(get_session)):
    """Create an RSVP for an event."""
    # For now, just return success - you can implement full RSVP logic later
    return {"message": "RSVP created", "event_id": event_id, "user_id": user_id}

# ---------------------
# Database
@app.get("/users")
async def list_users(session=Depends(get_session)):
    """Get all users."""
    users = session.execute(select(User)).scalars().all()
    return [{"id": u.id, "display_name": u.display_name} for u in users]

@app.get("/events")
async def list_events(session=Depends(get_session)):
    """Get all events (using event_id from requests)."""
    # Get unique event_ids from requests
    event_ids = session.execute(select(Request.event_id).distinct()).scalars().all()
    events = []
    for event_id in event_ids:
        # Count requests for this event
        request_count = session.execute(select(func.count(Request.id)).where(Request.event_id == event_id)).scalar()
        events.append({
            "id": event_id,
            "title": f"Event {event_id[:8]}",
            "description": f"Meetup event with {request_count} requests",
            "capacity": 10,
            "starts_at": "2024-01-01T10:00:00Z"
        })
    return events

@app.post("/rsvps")
async def create_rsvp(event_id: str, user_id: str = Depends(get_user_id), session=Depends(get_session)):
    """Create an RSVP for an event."""
    # For now, just return success - you can implement full RSVP logic later
    return {"message": "RSVP created", "event_id": event_id, "user_id": user_id}

# ---------------------
import os
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./dev.db")
engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)

class Base(DeclarativeBase):
    pass

@app.get("/users")
async def list_users(session=Depends(get_session)):
    """Get all users."""
    users = session.execute(select(User)).scalars().all()
    return [{"id": u.id, "display_name": u.display_name} for u in users]

@app.get("/events")
async def list_events(session=Depends(get_session)):
    """Get all events (using event_id from requests)."""
    # Get unique event_ids from requests
    event_ids = session.execute(select(Request.event_id).distinct()).scalars().all()
    events = []
    for event_id in event_ids:
        # Count requests for this event
        request_count = session.execute(select(func.count(Request.id)).where(Request.event_id == event_id)).scalar()
        events.append({
            "id": event_id,
            "title": f"Event {event_id[:8]}",
            "description": f"Meetup event with {request_count} requests",
            "capacity": 10,
            "starts_at": "2024-01-01T10:00:00Z"
        })
    return events

@app.post("/rsvps")
async def create_rsvp(event_id: str, user_id: str = Depends(get_user_id), session=Depends(get_session)):
    """Create an RSVP for an event."""
    # For now, just return success - you can implement full RSVP logic later
    return {"message": "RSVP created", "event_id": event_id, "user_id": user_id}

# ---------------------
# Enums
@app.get("/users")
async def list_users(session=Depends(get_session)):
    """Get all users."""
    users = session.execute(select(User)).scalars().all()
    return [{"id": u.id, "display_name": u.display_name} for u in users]

@app.get("/events")
async def list_events(session=Depends(get_session)):
    """Get all events (using event_id from requests)."""
    # Get unique event_ids from requests
    event_ids = session.execute(select(Request.event_id).distinct()).scalars().all()
    events = []
    for event_id in event_ids:
        # Count requests for this event
        request_count = session.execute(select(func.count(Request.id)).where(Request.event_id == event_id)).scalar()
        events.append({
            "id": event_id,
            "title": f"Event {event_id[:8]}",
            "description": f"Meetup event with {request_count} requests",
            "capacity": 10,
            "starts_at": "2024-01-01T10:00:00Z"
        })
    return events

@app.post("/rsvps")
async def create_rsvp(event_id: str, user_id: str = Depends(get_user_id), session=Depends(get_session)):
    """Create an RSVP for an event."""
    # For now, just return success - you can implement full RSVP logic later
    return {"message": "RSVP created", "event_id": event_id, "user_id": user_id}

# ---------------------
class RequestStatus(str, Enum):
    SUBMITTED = "SUBMITTED"
    ACCEPTED = "ACCEPTED"
    DECLINED = "DECLINED"
    EXPIRED = "EXPIRED"
    CANCELED = "CANCELED"

class BookingStatus(str, Enum):
    CONFIRMED = "CONFIRMED"
    CANCELED_BY_HOST = "CANCELED_BY_HOST"
    CANCELED_BY_GUEST = "CANCELED_BY_GUEST"
    COMPLETED = "COMPLETED"
    NO_SHOW = "NO_SHOW"

class ThreadScope(str, Enum):
    REQUEST = "request"
    BOOKING = "booking"

class MessageKind(str, Enum):
    USER = "user"
    SYSTEM = "system"

@app.get("/users")
async def list_users(session=Depends(get_session)):
    """Get all users."""
    users = session.execute(select(User)).scalars().all()
    return [{"id": u.id, "display_name": u.display_name} for u in users]

@app.get("/events")
async def list_events(session=Depends(get_session)):
    """Get all events (using event_id from requests)."""
    # Get unique event_ids from requests
    event_ids = session.execute(select(Request.event_id).distinct()).scalars().all()
    events = []
    for event_id in event_ids:
        # Count requests for this event
        request_count = session.execute(select(func.count(Request.id)).where(Request.event_id == event_id)).scalar()
        events.append({
            "id": event_id,
            "title": f"Event {event_id[:8]}",
            "description": f"Meetup event with {request_count} requests",
            "capacity": 10,
            "starts_at": "2024-01-01T10:00:00Z"
        })
    return events

@app.post("/rsvps")
async def create_rsvp(event_id: str, user_id: str = Depends(get_user_id), session=Depends(get_session)):
    """Create an RSVP for an event."""
    # For now, just return success - you can implement full RSVP logic later
    return {"message": "RSVP created", "event_id": event_id, "user_id": user_id}

# ---------------------
# Models
@app.get("/users")
async def list_users(session=Depends(get_session)):
    """Get all users."""
    users = session.execute(select(User)).scalars().all()
    return [{"id": u.id, "display_name": u.display_name} for u in users]

@app.get("/events")
async def list_events(session=Depends(get_session)):
    """Get all events (using event_id from requests)."""
    # Get unique event_ids from requests
    event_ids = session.execute(select(Request.event_id).distinct()).scalars().all()
    events = []
    for event_id in event_ids:
        # Count requests for this event
        request_count = session.execute(select(func.count(Request.id)).where(Request.event_id == event_id)).scalar()
        events.append({
            "id": event_id,
            "title": f"Event {event_id[:8]}",
            "description": f"Meetup event with {request_count} requests",
            "capacity": 10,
            "starts_at": "2024-01-01T10:00:00Z"
        })
    return events

@app.post("/rsvps")
async def create_rsvp(event_id: str, user_id: str = Depends(get_user_id), session=Depends(get_session)):
    """Create an RSVP for an event."""
    # For now, just return success - you can implement full RSVP logic later
    return {"message": "RSVP created", "event_id": event_id, "user_id": user_id}

# ---------------------
class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    display_name: Mapped[str] = mapped_column(String(120))

class Request(Base):
    __tablename__ = "requests"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    event_id: Mapped[str] = mapped_column(String(36))
    guest_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    host_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    status: Mapped[RequestStatus] = mapped_column(SQLEnum(RequestStatus), default=RequestStatus.SUBMITTED)
    auto_accept: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=func.now())

    guest: Mapped[User] = relationship(foreign_keys=[guest_id])
    host: Mapped[User] = relationship(foreign_keys=[host_id])
    booking: Mapped[Optional["Booking"]] = relationship(back_populates="request", uselist=False)

class Booking(Base):
    __tablename__ = "bookings"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    request_id: Mapped[str] = mapped_column(ForeignKey("requests.id", ondelete="CASCADE"), unique=True)
    status: Mapped[BookingStatus] = mapped_column(SQLEnum(BookingStatus), default=BookingStatus.CONFIRMED)

    request: Mapped[Request] = relationship(back_populates="booking")

class Thread(Base):
    __tablename__ = "threads"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    scope: Mapped[ThreadScope] = mapped_column(SQLEnum(ThreadScope), default=ThreadScope.REQUEST)
    request_id: Mapped[Optional[str]] = mapped_column(ForeignKey("requests.id", ondelete="CASCADE"))
    booking_id: Mapped[Optional[str]] = mapped_column(ForeignKey("bookings.id", ondelete="CASCADE"))
    event_id: Mapped[str] = mapped_column(String(36))
    is_locked: Mapped[bool] = mapped_column(Boolean, default=False)

class ThreadParticipant(Base):
    __tablename__ = "thread_participants"
    thread_id: Mapped[str] = mapped_column(ForeignKey("threads.id", ondelete="CASCADE"), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    role: Mapped[str] = mapped_column(String(30))

class Message(Base):
    __tablename__ = "messages"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    thread_id: Mapped[str] = mapped_column(ForeignKey("threads.id", ondelete="CASCADE"))
    sender_id: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"))
    client_msg_id: Mapped[str] = mapped_column(String(64))
    kind: Mapped[MessageKind] = mapped_column(SQLEnum(MessageKind), default=MessageKind.USER)
    body: Mapped[str] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=func.now())
    seq: Mapped[int] = mapped_column(BigInteger)

    __table_args__ = (
        UniqueConstraint("thread_id", "client_msg_id"),
        UniqueConstraint("thread_id", "seq"),
    )

class MessageRead(Base):
    __tablename__ = "message_reads"
    thread_id: Mapped[str] = mapped_column(ForeignKey("threads.id", ondelete="CASCADE"), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    last_read_seq: Mapped[int] = mapped_column(BigInteger, default=0)

@app.get("/users")
async def list_users(session=Depends(get_session)):
    """Get all users."""
    users = session.execute(select(User)).scalars().all()
    return [{"id": u.id, "display_name": u.display_name} for u in users]

@app.get("/events")
async def list_events(session=Depends(get_session)):
    """Get all events (using event_id from requests)."""
    # Get unique event_ids from requests
    event_ids = session.execute(select(Request.event_id).distinct()).scalars().all()
    events = []
    for event_id in event_ids:
        # Count requests for this event
        request_count = session.execute(select(func.count(Request.id)).where(Request.event_id == event_id)).scalar()
        events.append({
            "id": event_id,
            "title": f"Event {event_id[:8]}",
            "description": f"Meetup event with {request_count} requests",
            "capacity": 10,
            "starts_at": "2024-01-01T10:00:00Z"
        })
    return events

@app.post("/rsvps")
async def create_rsvp(event_id: str, user_id: str = Depends(get_user_id), session=Depends(get_session)):
    """Create an RSVP for an event."""
    # For now, just return success - you can implement full RSVP logic later
    return {"message": "RSVP created", "event_id": event_id, "user_id": user_id}

# ---------------------
# Schemas
@app.get("/users")
async def list_users(session=Depends(get_session)):
    """Get all users."""
    users = session.execute(select(User)).scalars().all()
    return [{"id": u.id, "display_name": u.display_name} for u in users]

@app.get("/events")
async def list_events(session=Depends(get_session)):
    """Get all events (using event_id from requests)."""
    # Get unique event_ids from requests
    event_ids = session.execute(select(Request.event_id).distinct()).scalars().all()
    events = []
    for event_id in event_ids:
        # Count requests for this event
        request_count = session.execute(select(func.count(Request.id)).where(Request.event_id == event_id)).scalar()
        events.append({
            "id": event_id,
            "title": f"Event {event_id[:8]}",
            "description": f"Meetup event with {request_count} requests",
            "capacity": 10,
            "starts_at": "2024-01-01T10:00:00Z"
        })
    return events

@app.post("/rsvps")
async def create_rsvp(event_id: str, user_id: str = Depends(get_user_id), session=Depends(get_session)):
    """Create an RSVP for an event."""
    # For now, just return success - you can implement full RSVP logic later
    return {"message": "RSVP created", "event_id": event_id, "user_id": user_id}

# ---------------------
class RequestCreate(BaseModel):
    event_id: str
    host_id: str
    auto_accept: bool = False

class RequestAction(BaseModel):
    action: str = Field(pattern="^(accept|decline)$")

class ThreadMessageIn(BaseModel):
    client_msg_id: str
    body: str

class MessageOut(BaseModel):
    id: str
    thread_id: str
    sender_id: Optional[str]
    kind: MessageKind
    body: str
    created_at: dt.datetime
    seq: int

@app.get("/users")
async def list_users(session=Depends(get_session)):
    """Get all users."""
    users = session.execute(select(User)).scalars().all()
    return [{"id": u.id, "display_name": u.display_name} for u in users]

@app.get("/events")
async def list_events(session=Depends(get_session)):
    """Get all events (using event_id from requests)."""
    # Get unique event_ids from requests
    event_ids = session.execute(select(Request.event_id).distinct()).scalars().all()
    events = []
    for event_id in event_ids:
        # Count requests for this event
        request_count = session.execute(select(func.count(Request.id)).where(Request.event_id == event_id)).scalar()
        events.append({
            "id": event_id,
            "title": f"Event {event_id[:8]}",
            "description": f"Meetup event with {request_count} requests",
            "capacity": 10,
            "starts_at": "2024-01-01T10:00:00Z"
        })
    return events

@app.post("/rsvps")
async def create_rsvp(event_id: str, user_id: str = Depends(get_user_id), session=Depends(get_session)):
    """Create an RSVP for an event."""
    # For now, just return success - you can implement full RSVP logic later
    return {"message": "RSVP created", "event_id": event_id, "user_id": user_id}

# ---------------------
# Dependencies
@app.get("/users")
async def list_users(session=Depends(get_session)):
    """Get all users."""
    users = session.execute(select(User)).scalars().all()
    return [{"id": u.id, "display_name": u.display_name} for u in users]

@app.get("/events")
async def list_events(session=Depends(get_session)):
    """Get all events (using event_id from requests)."""
    # Get unique event_ids from requests
    event_ids = session.execute(select(Request.event_id).distinct()).scalars().all()
    events = []
    for event_id in event_ids:
        # Count requests for this event
        request_count = session.execute(select(func.count(Request.id)).where(Request.event_id == event_id)).scalar()
        events.append({
            "id": event_id,
            "title": f"Event {event_id[:8]}",
            "description": f"Meetup event with {request_count} requests",
            "capacity": 10,
            "starts_at": "2024-01-01T10:00:00Z"
        })
    return events

@app.post("/rsvps")
async def create_rsvp(event_id: str, user_id: str = Depends(get_user_id), session=Depends(get_session)):
    """Create an RSVP for an event."""
    # For now, just return success - you can implement full RSVP logic later
    return {"message": "RSVP created", "event_id": event_id, "user_id": user_id}

# ---------------------
def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_user_id(x_user_id: str = Header(None)):
    if not x_user_id:
        raise HTTPException(401, "Missing X-User-Id header")
    return x_user_id

@app.get("/users")
async def list_users(session=Depends(get_session)):
    """Get all users."""
    users = session.execute(select(User)).scalars().all()
    return [{"id": u.id, "display_name": u.display_name} for u in users]

@app.get("/events")
async def list_events(session=Depends(get_session)):
    """Get all events (using event_id from requests)."""
    # Get unique event_ids from requests
    event_ids = session.execute(select(Request.event_id).distinct()).scalars().all()
    events = []
    for event_id in event_ids:
        # Count requests for this event
        request_count = session.execute(select(func.count(Request.id)).where(Request.event_id == event_id)).scalar()
        events.append({
            "id": event_id,
            "title": f"Event {event_id[:8]}",
            "description": f"Meetup event with {request_count} requests",
            "capacity": 10,
            "starts_at": "2024-01-01T10:00:00Z"
        })
    return events

@app.post("/rsvps")
async def create_rsvp(event_id: str, user_id: str = Depends(get_user_id), session=Depends(get_session)):
    """Create an RSVP for an event."""
    # For now, just return success - you can implement full RSVP logic later
    return {"message": "RSVP created", "event_id": event_id, "user_id": user_id}

# ---------------------
# Helpers
@app.get("/users")
async def list_users(session=Depends(get_session)):
    """Get all users."""
    users = session.execute(select(User)).scalars().all()
    return [{"id": u.id, "display_name": u.display_name} for u in users]

@app.get("/events")
async def list_events(session=Depends(get_session)):
    """Get all events (using event_id from requests)."""
    # Get unique event_ids from requests
    event_ids = session.execute(select(Request.event_id).distinct()).scalars().all()
    events = []
    for event_id in event_ids:
        # Count requests for this event
        request_count = session.execute(select(func.count(Request.id)).where(Request.event_id == event_id)).scalar()
        events.append({
            "id": event_id,
            "title": f"Event {event_id[:8]}",
            "description": f"Meetup event with {request_count} requests",
            "capacity": 10,
            "starts_at": "2024-01-01T10:00:00Z"
        })
    return events

@app.post("/rsvps")
async def create_rsvp(event_id: str, user_id: str = Depends(get_user_id), session=Depends(get_session)):
    """Create an RSVP for an event."""
    # For now, just return success - you can implement full RSVP logic later
    return {"message": "RSVP created", "event_id": event_id, "user_id": user_id}

# ---------------------
def _next_seq(session, thread_id: str) -> int:
    return session.execute(select(func.coalesce(func.max(Message.seq), 0) + 1).where(Message.thread_id == thread_id)).scalar_one()

def serialize_message(m: Message) -> dict:
    return {"id": m.id, "thread_id": m.thread_id, "sender_id": m.sender_id, "kind": m.kind.value, "body": m.body, "created_at": m.created_at.isoformat(), "seq": m.seq}

async def system_message(session, thread_id: str, body: str):
    m = Message(thread_id=thread_id, sender_id=None, client_msg_id=str(uuid.uuid4()), kind=MessageKind.SYSTEM, body=body, seq=_next_seq(session, thread_id))
    session.add(m)
    session.flush()

@app.get("/users")
async def list_users(session=Depends(get_session)):
    """Get all users."""
    users = session.execute(select(User)).scalars().all()
    return [{"id": u.id, "display_name": u.display_name} for u in users]

@app.get("/events")
async def list_events(session=Depends(get_session)):
    """Get all events (using event_id from requests)."""
    # Get unique event_ids from requests
    event_ids = session.execute(select(Request.event_id).distinct()).scalars().all()
    events = []
    for event_id in event_ids:
        # Count requests for this event
        request_count = session.execute(select(func.count(Request.id)).where(Request.event_id == event_id)).scalar()
        events.append({
            "id": event_id,
            "title": f"Event {event_id[:8]}",
            "description": f"Meetup event with {request_count} requests",
            "capacity": 10,
            "starts_at": "2024-01-01T10:00:00Z"
        })
    return events

@app.post("/rsvps")
async def create_rsvp(event_id: str, user_id: str = Depends(get_user_id), session=Depends(get_session)):
    """Create an RSVP for an event."""
    # For now, just return success - you can implement full RSVP logic later
    return {"message": "RSVP created", "event_id": event_id, "user_id": user_id}

# ---------------------
# Lifespan
@app.get("/users")
async def list_users(session=Depends(get_session)):
    """Get all users."""
    users = session.execute(select(User)).scalars().all()
    return [{"id": u.id, "display_name": u.display_name} for u in users]

@app.get("/events")
async def list_events(session=Depends(get_session)):
    """Get all events (using event_id from requests)."""
    # Get unique event_ids from requests
    event_ids = session.execute(select(Request.event_id).distinct()).scalars().all()
    events = []
    for event_id in event_ids:
        # Count requests for this event
        request_count = session.execute(select(func.count(Request.id)).where(Request.event_id == event_id)).scalar()
        events.append({
            "id": event_id,
            "title": f"Event {event_id[:8]}",
            "description": f"Meetup event with {request_count} requests",
            "capacity": 10,
            "starts_at": "2024-01-01T10:00:00Z"
        })
    return events

@app.post("/rsvps")
async def create_rsvp(event_id: str, user_id: str = Depends(get_user_id), session=Depends(get_session)):
    """Create an RSVP for an event."""
    # For now, just return success - you can implement full RSVP logic later
    return {"message": "RSVP created", "event_id": event_id, "user_id": user_id}

# ---------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await run_in_threadpool(Base.metadata.create_all, bind=engine)
    yield
    # Shutdown
    pass

@app.get("/users")
async def list_users(session=Depends(get_session)):
    """Get all users."""
    users = session.execute(select(User)).scalars().all()
    return [{"id": u.id, "display_name": u.display_name} for u in users]

@app.get("/events")
async def list_events(session=Depends(get_session)):
    """Get all events (using event_id from requests)."""
    # Get unique event_ids from requests
    event_ids = session.execute(select(Request.event_id).distinct()).scalars().all()
    events = []
    for event_id in event_ids:
        # Count requests for this event
        request_count = session.execute(select(func.count(Request.id)).where(Request.event_id == event_id)).scalar()
        events.append({
            "id": event_id,
            "title": f"Event {event_id[:8]}",
            "description": f"Meetup event with {request_count} requests",
            "capacity": 10,
            "starts_at": "2024-01-01T10:00:00Z"
        })
    return events

@app.post("/rsvps")
async def create_rsvp(event_id: str, user_id: str = Depends(get_user_id), session=Depends(get_session)):
    """Create an RSVP for an event."""
    # For now, just return success - you can implement full RSVP logic later
    return {"message": "RSVP created", "event_id": event_id, "user_id": user_id}

# ---------------------
# FastAPI app
@app.get("/users")
async def list_users(session=Depends(get_session)):
    """Get all users."""
    users = session.execute(select(User)).scalars().all()
    return [{"id": u.id, "display_name": u.display_name} for u in users]

@app.get("/events")
async def list_events(session=Depends(get_session)):
    """Get all events (using event_id from requests)."""
    # Get unique event_ids from requests
    event_ids = session.execute(select(Request.event_id).distinct()).scalars().all()
    events = []
    for event_id in event_ids:
        # Count requests for this event
        request_count = session.execute(select(func.count(Request.id)).where(Request.event_id == event_id)).scalar()
        events.append({
            "id": event_id,
            "title": f"Event {event_id[:8]}",
            "description": f"Meetup event with {request_count} requests",
            "capacity": 10,
            "starts_at": "2024-01-01T10:00:00Z"
        })
    return events

@app.post("/rsvps")
async def create_rsvp(event_id: str, user_id: str = Depends(get_user_id), session=Depends(get_session)):
    """Create an RSVP for an event."""
    # For now, just return success - you can implement full RSVP logic later
    return {"message": "RSVP created", "event_id": event_id, "user_id": user_id}

# ---------------------
app = FastAPI(title="Meetup Chat & Booking API", version="0.4.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True)

@app.get("/users")
async def list_users(session=Depends(get_session)):
    """Get all users."""
    users = session.execute(select(User)).scalars().all()
    return [{"id": u.id, "display_name": u.display_name} for u in users]

@app.get("/events")
async def list_events(session=Depends(get_session)):
    """Get all events (using event_id from requests)."""
    # Get unique event_ids from requests
    event_ids = session.execute(select(Request.event_id).distinct()).scalars().all()
    events = []
    for event_id in event_ids:
        # Count requests for this event
        request_count = session.execute(select(func.count(Request.id)).where(Request.event_id == event_id)).scalar()
        events.append({
            "id": event_id,
            "title": f"Event {event_id[:8]}",
            "description": f"Meetup event with {request_count} requests",
            "capacity": 10,
            "starts_at": "2024-01-01T10:00:00Z"
        })
    return events

@app.post("/rsvps")
async def create_rsvp(event_id: str, user_id: str = Depends(get_user_id), session=Depends(get_session)):
    """Create an RSVP for an event."""
    # For now, just return success - you can implement full RSVP logic later
    return {"message": "RSVP created", "event_id": event_id, "user_id": user_id}

# ---------------------
# Endpoints
@app.get("/users")
async def list_users(session=Depends(get_session)):
    """Get all users."""
    users = session.execute(select(User)).scalars().all()
    return [{"id": u.id, "display_name": u.display_name} for u in users]

@app.get("/events")
async def list_events(session=Depends(get_session)):
    """Get all events (using event_id from requests)."""
    # Get unique event_ids from requests
    event_ids = session.execute(select(Request.event_id).distinct()).scalars().all()
    events = []
    for event_id in event_ids:
        # Count requests for this event
        request_count = session.execute(select(func.count(Request.id)).where(Request.event_id == event_id)).scalar()
        events.append({
            "id": event_id,
            "title": f"Event {event_id[:8]}",
            "description": f"Meetup event with {request_count} requests",
            "capacity": 10,
            "starts_at": "2024-01-01T10:00:00Z"
        })
    return events

@app.post("/rsvps")
async def create_rsvp(event_id: str, user_id: str = Depends(get_user_id), session=Depends(get_session)):
    """Create an RSVP for an event."""
    # For now, just return success - you can implement full RSVP logic later
    return {"message": "RSVP created", "event_id": event_id, "user_id": user_id}

# ---------------------
@app.post("/requests")
async def create_request(payload: RequestCreate, user_id: str = Depends(get_user_id), session=Depends(get_session)):
    req = Request(event_id=payload.event_id, guest_id=user_id, host_id=payload.host_id, auto_accept=payload.auto_accept)
    session.add(req)
    session.flush()

    thread = Thread(scope=ThreadScope.REQUEST, request_id=req.id, event_id=payload.event_id)
    session.add(thread)
    session.flush()

    session.add_all([ThreadParticipant(thread_id=thread.id, user_id=req.guest_id, role="guest"), ThreadParticipant(thread_id=thread.id, user_id=req.host_id, role="host")])
    session.flush()

    # auto-accept: immediately create booking
    if req.auto_accept:
        req.status = RequestStatus.ACCEPTED
        booking = Booking(request_id=req.id, status=BookingStatus.CONFIRMED)
        session.add(booking)
        session.flush()
        thread.scope = ThreadScope.BOOKING
        thread.booking_id = booking.id
        session.flush()
        await system_message(session, thread.id, "Request auto-accepted, booking confirmed.")

    session.commit()
    return {"request_id": req.id, "thread_id": thread.id, "status": req.status}

@app.post("/requests/{request_id}/act")
async def act_on_request(request_id: str, payload: RequestAction, user_id: str = Depends(get_user_id), session=Depends(get_session)):
    req = session.execute(select(Request).where(Request.id == request_id)).scalar_one_or_none()
    if not req:
        raise HTTPException(404, "Not found")
    if user_id != req.host_id:
        raise HTTPException(403, "Forbidden")

    thread = session.execute(select(Thread).where(Thread.request_id == req.id)).scalar_one()

    if payload.action == "accept":
        req.status = RequestStatus.ACCEPTED
        booking = Booking(request_id=req.id, status=BookingStatus.CONFIRMED)
        session.add(booking)
        session.flush()
        thread.scope = ThreadScope.BOOKING
        thread.booking_id = booking.id
        session.flush()
        await system_message(session, thread.id, "Request accepted, booking confirmed.")
        session.commit()
        return {"status": req.status, "booking_id": booking.id, "thread_id": thread.id}

    if payload.action == "decline":
        req.status = RequestStatus.DECLINED
        thread.is_locked = True
        session.flush()
        await system_message(session, thread.id, "Request declined. Thread locked.")
        session.commit()
        return {"status": req.status, "thread_id": thread.id}

    raise HTTPException(400, "Invalid action")

@app.post("/dev/seed")
async def seed_database(session=Depends(get_session)):
    """Endpoint to seed the database with test data for smoke tests."""
    # Create test users
    guest = User(display_name="Test Guest")
    host = User(display_name="Test Host")
    session.add_all([guest, host])
    session.flush()
    
    # Create test event ID
    event_id = str(uuid.uuid4())
    
    session.commit()
    
    return {
        "guest_id": guest.id,
        "host_id": host.id,
        "event_id": event_id
    }

@app.get("/users")
async def list_users(session=Depends(get_session)):
    """Get all users."""
    users = session.execute(select(User)).scalars().all()
    return [{"id": u.id, "display_name": u.display_name} for u in users]

@app.get("/events")
async def list_events(session=Depends(get_session)):
    """Get all events (using event_id from requests)."""
    # Get unique event_ids from requests
    event_ids = session.execute(select(Request.event_id).distinct()).scalars().all()
    events = []
    for event_id in event_ids:
        # Count requests for this event
        request_count = session.execute(select(func.count(Request.id)).where(Request.event_id == event_id)).scalar()
        events.append({
            "id": event_id,
            "title": f"Event {event_id[:8]}",
            "description": f"Meetup event with {request_count} requests",
            "capacity": 10,
            "starts_at": "2024-01-01T10:00:00Z"
        })
    return events

@app.post("/rsvps")
async def create_rsvp(event_id: str, user_id: str = Depends(get_user_id), session=Depends(get_session)):
    """Create an RSVP for an event."""
    # For now, just return success - you can implement full RSVP logic later
    return {"message": "RSVP created", "event_id": event_id, "user_id": user_id}

# ---------------------
# Smoke tests
@app.get("/users")
async def list_users(session=Depends(get_session)):
    """Get all users."""
    users = session.execute(select(User)).scalars().all()
    return [{"id": u.id, "display_name": u.display_name} for u in users]

@app.get("/events")
async def list_events(session=Depends(get_session)):
    """Get all events (using event_id from requests)."""
    # Get unique event_ids from requests
    event_ids = session.execute(select(Request.event_id).distinct()).scalars().all()
    events = []
    for event_id in event_ids:
        # Count requests for this event
        request_count = session.execute(select(func.count(Request.id)).where(Request.event_id == event_id)).scalar()
        events.append({
            "id": event_id,
            "title": f"Event {event_id[:8]}",
            "description": f"Meetup event with {request_count} requests",
            "capacity": 10,
            "starts_at": "2024-01-01T10:00:00Z"
        })
    return events

@app.post("/rsvps")
async def create_rsvp(event_id: str, user_id: str = Depends(get_user_id), session=Depends(get_session)):
    """Create an RSVP for an event."""
    # For now, just return success - you can implement full RSVP logic later
    return {"message": "RSVP created", "event_id": event_id, "user_id": user_id}

# ---------------------
async def _run_smoke_tests():
    from asgi_lifespan import LifespanManager
    from httpx import AsyncClient

    async with LifespanManager(app):
        async with AsyncClient(app=app, base_url="http://testserver") as client:
            r = await client.post("/dev/seed")
            r.raise_for_status()
            data = r.json()
            guest, host, event = data["guest_id"], data["host_id"], data["event_id"]

            # auto-accept flow
            r = await client.post("/requests", headers={"X-User-Id": guest}, json={"event_id": event, "host_id": host, "auto_accept": True})
            r.raise_for_status()
            assert r.json()["status"] == "ACCEPTED"
            print("✅ Smoke tests passed!")

@app.get("/users")
async def list_users(session=Depends(get_session)):
    """Get all users."""
    users = session.execute(select(User)).scalars().all()
    return [{"id": u.id, "display_name": u.display_name} for u in users]

@app.get("/events")
async def list_events(session=Depends(get_session)):
    """Get all events (using event_id from requests)."""
    # Get unique event_ids from requests
    event_ids = session.execute(select(Request.event_id).distinct()).scalars().all()
    events = []
    for event_id in event_ids:
        # Count requests for this event
        request_count = session.execute(select(func.count(Request.id)).where(Request.event_id == event_id)).scalar()
        events.append({
            "id": event_id,
            "title": f"Event {event_id[:8]}",
            "description": f"Meetup event with {request_count} requests",
            "capacity": 10,
            "starts_at": "2024-01-01T10:00:00Z"
        })
    return events

@app.post("/rsvps")
async def create_rsvp(event_id: str, user_id: str = Depends(get_user_id), session=Depends(get_session)):
    """Create an RSVP for an event."""
    # For now, just return success - you can implement full RSVP logic later
    return {"message": "RSVP created", "event_id": event_id, "user_id": user_id}

# ---------------------
# Additional Schemas
@app.get("/users")
async def list_users(session=Depends(get_session)):
    """Get all users."""
    users = session.execute(select(User)).scalars().all()
    return [{"id": u.id, "display_name": u.display_name} for u in users]

@app.get("/events")
async def list_events(session=Depends(get_session)):
    """Get all events (using event_id from requests)."""
    # Get unique event_ids from requests
    event_ids = session.execute(select(Request.event_id).distinct()).scalars().all()
    events = []
    for event_id in event_ids:
        # Count requests for this event
        request_count = session.execute(select(func.count(Request.id)).where(Request.event_id == event_id)).scalar()
        events.append({
            "id": event_id,
            "title": f"Event {event_id[:8]}",
            "description": f"Meetup event with {request_count} requests",
            "capacity": 10,
            "starts_at": "2024-01-01T10:00:00Z"
        })
    return events

@app.post("/rsvps")
async def create_rsvp(event_id: str, user_id: str = Depends(get_user_id), session=Depends(get_session)):
    """Create an RSVP for an event."""
    # For now, just return success - you can implement full RSVP logic later
    return {"message": "RSVP created", "event_id": event_id, "user_id": user_id}

# ---------------------
class ThreadOut(BaseModel):
    id: str
    scope: ThreadScope
    request_id: Optional[str]
    booking_id: Optional[str]
    event_id: str
    is_locked: bool

class ThreadParticipantOut(BaseModel):
    thread_id: str
    user_id: str
    role: str

class MessageReadOut(BaseModel):
    thread_id: str
    user_id: str
    last_read_seq: int

class ThreadListOut(BaseModel):
    threads: list[ThreadOut]
    participants: dict[str, list[ThreadParticipantOut]]  # thread_id -> participants

@app.get("/users")
async def list_users(session=Depends(get_session)):
    """Get all users."""
    users = session.execute(select(User)).scalars().all()
    return [{"id": u.id, "display_name": u.display_name} for u in users]

@app.get("/events")
async def list_events(session=Depends(get_session)):
    """Get all events (using event_id from requests)."""
    # Get unique event_ids from requests
    event_ids = session.execute(select(Request.event_id).distinct()).scalars().all()
    events = []
    for event_id in event_ids:
        # Count requests for this event
        request_count = session.execute(select(func.count(Request.id)).where(Request.event_id == event_id)).scalar()
        events.append({
            "id": event_id,
            "title": f"Event {event_id[:8]}",
            "description": f"Meetup event with {request_count} requests",
            "capacity": 10,
            "starts_at": "2024-01-01T10:00:00Z"
        })
    return events

@app.post("/rsvps")
async def create_rsvp(event_id: str, user_id: str = Depends(get_user_id), session=Depends(get_session)):
    """Create an RSVP for an event."""
    # For now, just return success - you can implement full RSVP logic later
    return {"message": "RSVP created", "event_id": event_id, "user_id": user_id}

# ---------------------
# Chat Endpoints
@app.get("/users")
async def list_users(session=Depends(get_session)):
    """Get all users."""
    users = session.execute(select(User)).scalars().all()
    return [{"id": u.id, "display_name": u.display_name} for u in users]

@app.get("/events")
async def list_events(session=Depends(get_session)):
    """Get all events (using event_id from requests)."""
    # Get unique event_ids from requests
    event_ids = session.execute(select(Request.event_id).distinct()).scalars().all()
    events = []
    for event_id in event_ids:
        # Count requests for this event
        request_count = session.execute(select(func.count(Request.id)).where(Request.event_id == event_id)).scalar()
        events.append({
            "id": event_id,
            "title": f"Event {event_id[:8]}",
            "description": f"Meetup event with {request_count} requests",
            "capacity": 10,
            "starts_at": "2024-01-01T10:00:00Z"
        })
    return events

@app.post("/rsvps")
async def create_rsvp(event_id: str, user_id: str = Depends(get_user_id), session=Depends(get_session)):
    """Create an RSVP for an event."""
    # For now, just return success - you can implement full RSVP logic later
    return {"message": "RSVP created", "event_id": event_id, "user_id": user_id}

# ---------------------
@app.get("/threads", response_model=ThreadListOut)
async def get_user_threads(user_id: str = Depends(get_user_id), session=Depends(get_session)):
    """Get all threads where the user is a participant."""
    # Get thread IDs where user is a participant
    participant_threads = session.execute(
        select(ThreadParticipant.thread_id)
        .where(ThreadParticipant.user_id == user_id)
    ).scalars().all()
    
    if not participant_threads:
        return ThreadListOut(threads=[], participants={})
    
    # Get thread details
    threads = session.execute(
        select(Thread)
        .where(Thread.id.in_(participant_threads))
        .order_by(Thread.id)
    ).scalars().all()
    
    # Get participants for each thread
    participants = {}
    for thread in threads:
        thread_participants = session.execute(
            select(ThreadParticipant)
            .where(ThreadParticipant.thread_id == thread.id)
        ).scalars().all()
        participants[thread.id] = [
            ThreadParticipantOut(thread_id=p.thread_id, user_id=p.user_id, role=p.role)
            for p in thread_participants
        ]
    
    return ThreadListOut(
        threads=[ThreadOut(
            id=t.id, scope=t.scope, request_id=t.request_id, 
            booking_id=t.booking_id, event_id=t.event_id, is_locked=t.is_locked
        ) for t in threads],
        participants=participants
    )

@app.get("/threads/{thread_id}/messages")
async def get_thread_messages(
    thread_id: str, 
    limit: int = 50, 
    offset: int = 0,
    user_id: str = Depends(get_user_id), 
    session=Depends(get_session)
):
    """Get messages for a specific thread."""
    # Verify user has access to thread
    participant = session.execute(
        select(ThreadParticipant)
        .where(ThreadParticipant.thread_id == thread_id, ThreadParticipant.user_id == user_id)
    ).scalar_one_or_none()
    
    if not participant:
        raise HTTPException(403, "Access denied")
    
    # Get messages
    messages = session.execute(
        select(Message)
        .where(Message.thread_id == thread_id)
        .order_by(Message.seq.desc())
        .limit(limit)
        .offset(offset)
    ).scalars().all()
    
    return [serialize_message(m) for m in reversed(messages)]

@app.post("/threads/{thread_id}/messages")
async def send_message(
    thread_id: str,
    payload: ThreadMessageIn,
    user_id: str = Depends(get_user_id),
    session=Depends(get_session)
):
    """Send a message to a thread."""
    # Verify user has access to thread
    participant = session.execute(
        select(ThreadParticipant)
        .where(ThreadParticipant.thread_id == thread_id, ThreadParticipant.user_id == user_id)
    ).scalar_one_or_none()
    
    if not participant:
        raise HTTPException(403, "Access denied")
    
    # Check if thread is locked
    thread = session.execute(select(Thread).where(Thread.id == thread_id)).scalar_one()
    if thread.is_locked:
        raise HTTPException(400, "Thread is locked")
    
    # Create message
    message = Message(
        thread_id=thread_id,
        sender_id=user_id,
        client_msg_id=payload.client_msg_id,
        kind=MessageKind.USER,
        body=payload.body,
        seq=_next_seq(session, thread_id)
    )
    
    try:
        session.add(message)
        session.commit()
        
        # Broadcast via WebSocket
        await manager.broadcast_to_thread({
            "type": "new_message",
            "message": serialize_message(message)
        }, thread_id, exclude_user=message.sender_id)
        
        return serialize_message(message)
    except IntegrityError:
        session.rollback()
        raise HTTPException(400, "Duplicate client_msg_id")

@app.post("/threads/{thread_id}/read")
async def mark_messages_read(
    thread_id: str,
    last_read_seq: int,
    user_id: str = Depends(get_user_id),
    session=Depends(get_session)
):
    """Mark messages as read up to a specific sequence number."""
    # Verify user has access to thread
    participant = session.execute(
        select(ThreadParticipant)
        .where(ThreadParticipant.thread_id == thread_id, ThreadParticipant.user_id == user_id)
    ).scalar_one_or_none()
    
    if not participant:
        raise HTTPException(403, "Access denied")
    
    # Update or create read status
    read_status = session.execute(
        select(MessageRead)
        .where(MessageRead.thread_id == thread_id, MessageRead.user_id == user_id)
    ).scalar_one_or_none()
    
    if read_status:
        read_status.last_read_seq = last_read_seq
    else:
        read_status = MessageRead(
            thread_id=thread_id,
            user_id=user_id,
            last_read_seq=last_read_seq
        )
        session.add(read_status)
    
    session.commit()
    return {"status": "success"}

@app.get("/threads/{thread_id}/participants")
async def get_thread_participants(
    thread_id: str,
    user_id: str = Depends(get_user_id),
    session=Depends(get_session)
):
    """Get participants for a specific thread."""
    # Verify user has access to thread
    participant = session.execute(
        select(ThreadParticipant)
        .where(ThreadParticipant.thread_id == thread_id, ThreadParticipant.user_id == user_id)
    ).scalar_one_or_none()
    
    if not participant:
        raise HTTPException(403, "Access denied")
    
    # Get all participants
    participants = session.execute(
        select(ThreadParticipant)
        .where(ThreadParticipant.thread_id == thread_id)
    ).scalars().all()
    
    return [
        ThreadParticipantOut(thread_id=p.thread_id, user_id=p.user_id, role=p.role)
        for p in participants
    ]

@app.get("/users")
async def list_users(session=Depends(get_session)):
    """Get all users."""
    users = session.execute(select(User)).scalars().all()
    return [{"id": u.id, "display_name": u.display_name} for u in users]

@app.get("/events")
async def list_events(session=Depends(get_session)):
    """Get all events (using event_id from requests)."""
    # Get unique event_ids from requests
    event_ids = session.execute(select(Request.event_id).distinct()).scalars().all()
    events = []
    for event_id in event_ids:
        # Count requests for this event
        request_count = session.execute(select(func.count(Request.id)).where(Request.event_id == event_id)).scalar()
        events.append({
            "id": event_id,
            "title": f"Event {event_id[:8]}",
            "description": f"Meetup event with {request_count} requests",
            "capacity": 10,
            "starts_at": "2024-01-01T10:00:00Z"
        })
    return events

@app.post("/rsvps")
async def create_rsvp(event_id: str, user_id: str = Depends(get_user_id), session=Depends(get_session)):
    """Create an RSVP for an event."""
    # For now, just return success - you can implement full RSVP logic later
    return {"message": "RSVP created", "event_id": event_id, "user_id": user_id}

# ---------------------
# WebSocket Manager
@app.get("/users")
async def list_users(session=Depends(get_session)):
    """Get all users."""
    users = session.execute(select(User)).scalars().all()
    return [{"id": u.id, "display_name": u.display_name} for u in users]

@app.get("/events")
async def list_events(session=Depends(get_session)):
    """Get all events (using event_id from requests)."""
    # Get unique event_ids from requests
    event_ids = session.execute(select(Request.event_id).distinct()).scalars().all()
    events = []
    for event_id in event_ids:
        # Count requests for this event
        request_count = session.execute(select(func.count(Request.id)).where(Request.event_id == event_id)).scalar()
        events.append({
            "id": event_id,
            "title": f"Event {event_id[:8]}",
            "description": f"Meetup event with {request_count} requests",
            "capacity": 10,
            "starts_at": "2024-01-01T10:00:00Z"
        })
    return events

@app.post("/rsvps")
async def create_rsvp(event_id: str, user_id: str = Depends(get_user_id), session=Depends(get_session)):
    """Create an RSVP for an event."""
    # For now, just return success - you can implement full RSVP logic later
    return {"message": "RSVP created", "event_id": event_id, "user_id": user_id}

# ---------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
        self.user_threads: dict[str, set[str]] = {}  # user_id -> set of thread_ids
    
    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket
        self.user_threads[user_id] = set()
    
    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
        if user_id in self.user_threads:
            del self.user_threads[user_id]
    
    async def send_personal_message(self, message: dict, user_id: str):
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_json(message)
            except:
                self.disconnect(user_id)
    
    async def broadcast_to_thread(self, message: dict, thread_id: str, exclude_user: str = None):
        # Get all participants for this thread
        with SessionLocal() as session:
            participants = session.execute(
                select(ThreadParticipant)
                .where(ThreadParticipant.thread_id == thread_id)
            ).scalars().all()
            
            for participant in participants:
                if participant.user_id != exclude_user and participant.user_id in self.active_connections:
                    await self.send_personal_message(message, participant.user_id)

manager = ConnectionManager()

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "join_thread":
                thread_id = data.get("thread_id")
                if thread_id:
                    manager.user_threads[user_id].add(thread_id)
                    await websocket.send_json({
                        "type": "joined_thread",
                        "thread_id": thread_id
                    })
            
            elif data.get("type") == "leave_thread":
                thread_id = data.get("thread_id")
                if thread_id and thread_id in manager.user_threads[user_id]:
                    manager.user_threads[user_id].remove(thread_id)
                    await websocket.send_json({
                        "type": "left_thread",
                        "thread_id": thread_id
                    })
            
    except WebSocketDisconnect:
        manager.disconnect(user_id)

if __name__ == "__main__":
    import argparse, uvicorn
    p = argparse.ArgumentParser()
    p.add_argument("--test", action="store_true")
    a = p.parse_args()
    if a.test:
        asyncio.run(_run_smoke_tests())
    else:
        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)

# Add static file serving
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Mount static files
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
async def read_index():
    return FileResponse("index.html")

