"""
Meetup Chat & Booking Backend (FastAPI, Python) — Production Version for Railway
================================================================================

This version is optimized for Railway deployment with PostgreSQL database.
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

# ---------------------
# Database
# ---------------------
import os
from urllib.parse import urlparse

# Get database URL from Railway environment variable
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./dev.db")

# For Railway PostgreSQL, we need to handle the connection properly
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

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

# ---------------------
# Models
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

# ---------------------
# Schemas
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

# ---------------------
# Dependencies
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

# ---------------------
# Helpers
# ---------------------
def _next_seq(session, thread_id: str) -> int:
    return session.execute(select(func.coalesce(func.max(Message.seq), 0) + 1).where(Message.thread_id == thread_id)).scalar_one()

def serialize_message(m: Message) -> dict:
    return {"id": m.id, "thread_id": m.thread_id, "sender_id": m.sender_id, "kind": m.kind.value, "body": m.body, "created_at": m.created_at.isoformat(), "seq": m.seq}

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
app = FastAPI(title="Meetup Chat & Booking API", version="0.4.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True)

# ---------------------
# Endpoints
# ---------------------
@app.get("/")
async def root():
    return {"message": "Meetup Chat & Booking API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": dt.datetime.now().isoformat()}

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

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app_production:app", host="0.0.0.0", port=port, reload=False)
