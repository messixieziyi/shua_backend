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
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, EmailStr

from sqlalchemy import (
    BigInteger, Boolean, DateTime, Enum as SQLEnum, ForeignKey, String, Text, UniqueConstraint, func, select, create_engine, text, delete, JSON, Date
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker
from sqlalchemy.exc import IntegrityError
from starlette.concurrency import run_in_threadpool

# Authentication imports
from passlib.context import CryptContext
from argon2 import PasswordHasher
from jose import JWTError, jwt
from datetime import timedelta

# Cloudinary imports
from cloudinary_config import validate_and_upload_image, delete_image, is_cloudinary_configured

# ---------------------
# Database
# ---------------------
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Try to get DATABASE_URL from environment, fallback to SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./dev.db")

# Configure PostgreSQL SSL for Railway
if DATABASE_URL.startswith("postgresql://"):
    # Add SSL configuration for Railway PostgreSQL
    if "?" in DATABASE_URL:
        DATABASE_URL += "&sslmode=require"
    else:
        DATABASE_URL += "?sslmode=require"

print(f"🔗 Using database: {DATABASE_URL[:50]}...")

# Create engine with connection pooling and retry logic
engine = create_engine(
    DATABASE_URL, 
    echo=False, 
    future=True,
    pool_pre_ping=True,  # Verify connections before use
    pool_recycle=300,    # Recycle connections every 5 minutes
    connect_args={"connect_timeout": 10} if DATABASE_URL.startswith("postgresql://") else {}
)
print("✅ Database engine created successfully")
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)

# ---------------------
# Authentication Setup
# ---------------------
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production-please-make-it-secure")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
security = HTTPBearer()

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

class EventStatus(str, Enum):
    ACTIVE = "ACTIVE"
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

class NotificationType(str, Enum):
    """Extensible notification types for the system"""
    EVENT_JOIN_REQUEST = "event_join_request"  # Someone requested to join your event
    EVENT_JOIN_ACCEPTED = "event_join_accepted"  # Your request to join was accepted
    EVENT_JOIN_DECLINED = "event_join_declined"  # Your request to join was declined
    EVENT_JOINED = "event_joined"  # Someone joined your event (auto-accept or accepted)
    EVENT_LEFT = "event_left"  # Someone left your event
    EVENT_CANCELED = "event_canceled"  # Event you're in was canceled
    NEW_MESSAGE = "new_message"  # New message in a chat thread
    # Future notification types can be added here:
    # EVENT_REMINDER = "event_reminder"
    # EVENT_STARTING_SOON = "event_starting_soon"
    # FRIEND_REQUEST = "friend_request"
    # etc.

# ---------------------
# Models
# ---------------------
class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    display_name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    profile_picture: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Base64 encoded image
    bio: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    gallery_image_1: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Base64 encoded image
    gallery_image_2: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Base64 encoded image
    gallery_image_3: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Base64 encoded image
    gallery_image_4: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Base64 encoded image
    gallery_image_5: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Base64 encoded image
    gallery_image_6: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Base64 encoded image
    first_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    last_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    gender: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    birthday: Mapped[Optional[dt.date]] = mapped_column(Date, nullable=True)
    sports: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=func.now())

class Event(Base):
    __tablename__ = "events"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    starts_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    capacity: Mapped[int] = mapped_column(BigInteger, default=10)
    activity_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    location: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    address: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=func.now())
    image_1: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Base64 encoded image
    image_2: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Base64 encoded image
    image_3: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Base64 encoded image
    auto_accept: Mapped[Optional[bool]] = mapped_column(Boolean, default=False, nullable=True)  # Free to join if True
    status: Mapped[EventStatus] = mapped_column(SQLEnum(EventStatus), default=EventStatus.ACTIVE)
    cancellation_deadline_hours: Mapped[int] = mapped_column(BigInteger, default=24)
    level_requirement: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # Stores generic level ID: "level_1" to "level_5"

class Request(Base):
    __tablename__ = "requests"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    event_id: Mapped[str] = mapped_column(ForeignKey("events.id"))
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
    request_id: Mapped[Optional[str]] = mapped_column(String(36))
    booking_id: Mapped[Optional[str]] = mapped_column(String(36))
    event_id: Mapped[str] = mapped_column(ForeignKey("events.id"), unique=True)  # One group chat per event
    is_locked: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=func.now())

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

class Tag(Base):
    __tablename__ = "tags"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(50))  # Removed unique constraint
    sport_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # Sport-specific tags
    color: Mapped[Optional[str]] = mapped_column(String(7), nullable=True)  # Hex color code
    description: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=func.now())
    
    __table_args__ = (
        UniqueConstraint('name', 'sport_type', name='uix_tag_name_sport'),
    )

class EventTag(Base):
    __tablename__ = "event_tags"
    event_id: Mapped[str] = mapped_column(ForeignKey("events.id", ondelete="CASCADE"), primary_key=True)
    tag_id: Mapped[str] = mapped_column(ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=func.now())

class EventLike(Base):
    __tablename__ = "event_likes"
    event_id: Mapped[str] = mapped_column(ForeignKey("events.id", ondelete="CASCADE"), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=func.now())
    
    __table_args__ = (
        UniqueConstraint('event_id', 'user_id', name='uix_event_user_like'),
    )

class Notification(Base):
    __tablename__ = "notifications"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    type: Mapped[NotificationType] = mapped_column(SQLEnum(NotificationType), nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    is_read: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=func.now(), index=True)
    
    # Optional reference fields for linking notifications to entities
    # These allow notifications to link to events, threads, users, etc.
    event_id: Mapped[Optional[str]] = mapped_column(ForeignKey("events.id", ondelete="CASCADE"), nullable=True)
    thread_id: Mapped[Optional[str]] = mapped_column(ForeignKey("threads.id", ondelete="CASCADE"), nullable=True)
    request_id: Mapped[Optional[str]] = mapped_column(ForeignKey("requests.id", ondelete="CASCADE"), nullable=True)
    related_user_id: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    
    # Additional metadata as JSON for extensibility
    extra_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)


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

class TagCreate(BaseModel):
    name: str
    sport_type: Optional[str] = None
    color: Optional[str] = None
    description: Optional[str] = None

class TagOut(BaseModel):
    id: str
    name: str
    color: Optional[str]
    description: Optional[str]
    created_at: dt.datetime

class EventTagCreate(BaseModel):
    tag_ids: list[str]

# Notification Schemas
class NotificationOut(BaseModel):
    id: str
    user_id: str
    type: NotificationType
    title: str
    body: str
    is_read: bool
    created_at: dt.datetime
    event_id: Optional[str] = None
    thread_id: Optional[str] = None
    request_id: Optional[str] = None
    related_user_id: Optional[str] = None
    metadata: Optional[dict] = None

class NotificationListOut(BaseModel):
    notifications: list[NotificationOut]
    unread_count: int
    total_count: int

class NotificationMarkRead(BaseModel):
    notification_ids: list[str]

# Image Upload Schemas
class ImageUpload(BaseModel):
    image_data: str  # Base64 encoded image
    image_type: str  # MIME type like 'image/jpeg', 'image/png'

class ProfilePictureUpdate(BaseModel):
    profile_picture: Optional[str] = None  # Base64 encoded image

class UserProfileUpdate(BaseModel):
    display_name: Optional[str] = None
    bio: Optional[str] = None
    gallery_image_1: Optional[str] = None  # Base64 encoded image
    gallery_image_2: Optional[str] = None  # Base64 encoded image
    gallery_image_3: Optional[str] = None  # Base64 encoded image
    gallery_image_4: Optional[str] = None  # Base64 encoded image
    gallery_image_5: Optional[str] = None  # Base64 encoded image
    gallery_image_6: Optional[str] = None  # Base64 encoded image
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    gender: Optional[str] = None
    birthday: Optional[dt.date] = None
    sports: Optional[list] = None

class EventImagesUpdate(BaseModel):
    image_1: Optional[str] = None  # Base64 encoded image
    image_2: Optional[str] = None  # Base64 encoded image  
    image_3: Optional[str] = None  # Base64 encoded image

class EventUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    starts_at: Optional[str] = None  # ISO format datetime string
    capacity: Optional[int] = None
    activity_type: Optional[str] = None
    location: Optional[str] = None
    address: Optional[str] = None
    image_1: Optional[str] = None  # Base64 encoded image
    image_2: Optional[str] = None  # Base64 encoded image
    image_3: Optional[str] = None  # Base64 encoded image
    tag_ids: Optional[list[str]] = None  # List of tag IDs
    cancellation_deadline_hours: Optional[int] = None
    level_requirement: Optional[str] = None  # Level ID: "level_1" to "level_5"

# ---------------------
# Authentication Schemas
# ---------------------
class UserRegister(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)
    display_name: str = Field(min_length=1, max_length=120)
    first_name: Optional[str] = None
    last_name: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user_id: str
    display_name: str
    email: str

class UserOut(BaseModel):
    id: str
    email: str
    display_name: str
    profile_picture: Optional[str] = None
    bio: Optional[str] = None
    gallery_image_1: Optional[str] = None
    gallery_image_2: Optional[str] = None
    gallery_image_3: Optional[str] = None
    gallery_image_4: Optional[str] = None
    gallery_image_5: Optional[str] = None
    gallery_image_6: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    gender: Optional[str] = None
    birthday: Optional[dt.date] = None
    sports: Optional[list] = None
    created_at: Optional[str] = None

# ---------------------
# Authentication Utilities
# ---------------------
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = dt.datetime.utcnow() + expires_delta
    else:
        expire = dt.datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str) -> Optional[dict]:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError as e:
        print(f"JWT decode error: {str(e)}")
        print(f"Token preview: {token[:20]}..." if len(token) > 20 else f"Token: {token}")
        print(f"SECRET_KEY configured: {'Yes' if SECRET_KEY != 'your-secret-key-change-in-production-please-make-it-secure' else 'Using default (INSECURE!)'}")
        return None

# ---------------------
# Image Upload Utilities
# ---------------------
import base64
import re

def validate_base64_image(image_data: str, max_size_mb: int = 5) -> tuple[bool, str]:
    """
    Validate base64 encoded image data or image URL.
    
    Note: This function is kept for backward compatibility but now delegates
    to Cloudinary for actual uploads. Use validate_and_upload_image for new code.
    """
    try:
        # Allow HTTP/HTTPS URLs (for existing images from Unsplash, Cloudinary, etc.)
        if image_data.startswith('http://') or image_data.startswith('https://'):
            return True, "Valid image URL"
        
        # Check if it's a valid base64 data URL
        if not image_data.startswith('data:image/'):
            return False, "Invalid image format. Must be a data URL or valid image URL."
        
        # Extract the base64 part
        header, data = image_data.split(',', 1)
        
        # Validate MIME type
        mime_match = re.match(r'data:image/(jpeg|jpg|png|gif|webp|heic|heif)', header)
        if not mime_match:
            return False, "Unsupported image type. Only JPEG, PNG, GIF, WebP, and HEIC are allowed."
        
        # Decode and check size
        try:
            decoded = base64.b64decode(data)
            size_mb = len(decoded) / (1024 * 1024)
            
            if size_mb > max_size_mb:
                return False, f"Image too large. Maximum size is {max_size_mb}MB."
                
            return True, "Valid image"
            
        except Exception:
            return False, "Invalid base64 encoding."
            
    except Exception as e:
        return False, f"Image validation error: {str(e)}"

# ---------------------
# Dependencies
# ---------------------
def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    session = Depends(get_session)
) -> User:
    """Get the current authenticated user from JWT token."""
    token = credentials.credentials
    payload = decode_access_token(token)
    
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user_id: str = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = session.execute(select(User).where(User.id == user_id)).scalar_one_or_none()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user

def get_user_id(x_user_id: str = Header(None)):
    """Legacy dependency - kept for backward compatibility but deprecated."""
    if not x_user_id:
        raise HTTPException(401, "Missing X-User-Id header")
    return x_user_id

# ---------------------
# Helpers
# ---------------------
def _next_seq(session, thread_id: str) -> int:
    return session.execute(select(func.coalesce(func.max(Message.seq), 0) + 1).where(Message.thread_id == thread_id)).scalar_one()

def serialize_message(m: Message, session=None) -> dict:
    sender_name = None
    if m.sender_id and session:
        # Get sender's display name
        user = session.execute(select(User).where(User.id == m.sender_id)).scalar_one_or_none()
        if user:
            sender_name = user.display_name
    
    return {
        "id": m.id, 
        "thread_id": m.thread_id, 
        "sender_id": m.sender_id, 
        "sender_name": sender_name,
        "kind": m.kind.value, 
        "body": m.body, 
        "created_at": m.created_at.isoformat(), 
        "seq": m.seq
    }

# ---------------------
# Notification Helper Functions
# ---------------------
async def create_notification(
    session,
    user_id: str,
    notification_type: NotificationType,
    title: str,
    body: str,
    event_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    request_id: Optional[str] = None,
    related_user_id: Optional[str] = None,
    metadata: Optional[dict] = None
) -> Notification:
    """Create a notification and send it via WebSocket if user is connected."""
    notification = Notification(
        user_id=user_id,
        type=notification_type,
        title=title,
        body=body,
        event_id=event_id,
        thread_id=thread_id,
        request_id=request_id,
        related_user_id=related_user_id,
        extra_data=metadata,  # Use extra_data field name to match model
        is_read=False
    )
    session.add(notification)
    session.flush()
    
    # Send notification via WebSocket if user is connected
    await manager.send_personal_message({
        "type": "new_notification",
        "notification": {
            "id": notification.id,
            "user_id": notification.user_id,
            "type": notification.type.value,
            "title": notification.title,
            "body": notification.body,
            "is_read": notification.is_read,
            "created_at": notification.created_at.isoformat(),
            "event_id": notification.event_id,
            "thread_id": notification.thread_id,
            "request_id": notification.request_id,
            "related_user_id": notification.related_user_id,
            "metadata": notification.extra_data
        }
    }, user_id)
    
    return notification

async def system_message(session, thread_id: str, body: str):
    try:
        m = Message(thread_id=thread_id, sender_id=None, client_msg_id=str(uuid.uuid4()), kind=MessageKind.SYSTEM, body=body, seq=_next_seq(session, thread_id))
        session.add(m)
        session.flush() # Should be safe as we are in a transaction
        
        # Try to broadcast if manager is available
        try:
            await manager.broadcast_to_thread({
                "type": "new_message",
                "thread_id": thread_id,
                "message": serialize_message(m, session)
            }, thread_id)
        except Exception as e:
            print(f"⚠️ Failed to broadcast system message: {e}")
            
    except Exception as e:
        print(f"❌ Failed to create system message: {e}")
        # Don't re-raise to prevent main action failure

# ---------------------
# Lifespan
# ---------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting application...")
    try:
        # Test database connection first
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ Database connection successful")
        
        # Create tables
        await run_in_threadpool(Base.metadata.create_all, bind=engine)
        print("✅ Database tables created successfully")
        
    except Exception as e:
        print(f"⚠️  Database connection failed during startup: {e}")
        print("🔄 Application will continue - database will be initialized on first use")
    
    print("🎉 Application startup complete")
    yield
    # Shutdown
    print("🛑 Application shutting down...")
    pass

# ---------------------
# FastAPI app
# ---------------------
app = FastAPI(title="Meetup Chat & Booking API", version="0.4.0", lifespan=lifespan)
# CORS configuration
# Allow Vercel and localhost origins
# Using regex pattern to match Vercel domains and localhost
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost:\d+|.*\.vercel\.app)",
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    allow_credentials=True,
    expose_headers=["*"]
)

# ---------------------
# Authentication Endpoints
# ---------------------
@app.post("/auth/register", response_model=Token)
async def register(user_data: UserRegister, session=Depends(get_session)):
    """Register a new user."""
    try:
        # Check if user with email already exists
        existing_user = session.execute(
            select(User).where(User.email == user_data.email)
        ).scalar_one_or_none()
        
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create new user
        hashed_password = get_password_hash(user_data.password)
        new_user = User(
            email=user_data.email,
            display_name=user_data.display_name,
            hashed_password=hashed_password,
            first_name=user_data.first_name,
            last_name=user_data.last_name
        )
        session.add(new_user)
        session.commit()
        session.refresh(new_user)
        
        # Create access token
        access_token = create_access_token(data={"sub": new_user.id})
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            user_id=new_user.id,
            display_name=new_user.display_name,
            email=new_user.email
        )
    except HTTPException:
        session.rollback()
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register user: {str(e)}"
        )

@app.post("/auth/login", response_model=Token)
async def login(credentials: UserLogin, session=Depends(get_session)):
    """Login with email and password."""
    try:
        # Find user by email
        user = session.execute(
            select(User).where(User.email == credentials.email)
        ).scalar_one_or_none()
        
        if not user or not verify_password(credentials.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create access token
        access_token = create_access_token(data={"sub": user.id})
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            user_id=user.id,
            display_name=user.display_name,
            email=user.email
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to login: {str(e)}"
        )

@app.post("/auth/logout")
async def logout(current_user: User = Depends(get_current_user)):
    """Logout (client should discard the token)."""
    # For JWT tokens, logout is typically handled client-side by discarding the token
    # If you need server-side token revocation, you'd need to maintain a blacklist
    return {"message": "Successfully logged out"}

@app.get("/auth/me", response_model=UserOut)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current authenticated user information."""
    return UserOut(
        id=current_user.id,
        email=current_user.email,
        display_name=current_user.display_name,
        profile_picture=current_user.profile_picture,
        bio=current_user.bio,
        gallery_image_1=current_user.gallery_image_1,
        gallery_image_2=current_user.gallery_image_2,
        gallery_image_3=current_user.gallery_image_3,
        gallery_image_4=current_user.gallery_image_4,
        gallery_image_5=current_user.gallery_image_5,
        gallery_image_6=current_user.gallery_image_6,
        first_name=current_user.first_name,
        last_name=current_user.last_name,
        gender=current_user.gender,
        birthday=current_user.birthday,
        sports=current_user.sports,
        created_at=current_user.created_at.isoformat() if current_user.created_at else None
    )

# ---------------------
# User Endpoints
# ---------------------
@app.get("/users")
async def get_current_user_info(current_user: User = Depends(get_current_user), session=Depends(get_session)):
    """Get current user information. Requires authentication."""
    return {
        "id": current_user.id, 
        "display_name": current_user.display_name, 
        "email": current_user.email,
        "profile_picture": current_user.profile_picture,
        "bio": current_user.bio,
        "gallery_image_1": current_user.gallery_image_1,
        "gallery_image_2": current_user.gallery_image_2,
        "gallery_image_3": current_user.gallery_image_3,
        "gallery_image_4": current_user.gallery_image_4,
        "gallery_image_5": current_user.gallery_image_5,
        "gallery_image_6": current_user.gallery_image_6,
        "first_name": current_user.first_name,
        "last_name": current_user.last_name,
        "gender": current_user.gender,
        "birthday": current_user.birthday,
        "sports": current_user.sports,
        "created_at": current_user.created_at.isoformat() if current_user.created_at else None
    }

@app.get("/users/{user_id}")
async def get_user_public_profile(
    user_id: str,
    current_user: User = Depends(get_current_user),
    session=Depends(get_session)
):
    """Get a user's public profile. Requires authentication."""
    user = session.execute(select(User).where(User.id == user_id)).scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Return public profile info (no email)
    return {
        "id": user.id,
        "display_name": user.display_name,
        "profile_picture": user.profile_picture,
        "bio": user.bio,
        "gallery_image_1": user.gallery_image_1,
        "gallery_image_2": user.gallery_image_2,
        "gallery_image_3": user.gallery_image_3,
        "gallery_image_4": user.gallery_image_4,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "sports": user.sports,
        "created_at": user.created_at.isoformat() if user.created_at else None
    }

@app.get("/users/{user_id}/events")
async def get_user_hosting_events(
    user_id: str,
    current_user: User = Depends(get_current_user),
    session=Depends(get_session)
):
    """Get events hosted by a specific user. Requires authentication."""
    # Verify user exists
    user = session.execute(select(User).where(User.id == user_id)).scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get all events created by this user
    events = session.execute(
        select(Event).where(Event.created_by == user_id).where(Event.status == "ACTIVE")
    ).scalars().all()
    
    result = []
    for event in events:
        # Get tags
        event_tags = session.execute(
            select(Tag).join(EventTag).where(EventTag.event_id == event.id)
        ).scalars().all()
        
        # Get occupied spots count
        occupied_count = session.execute(
            select(func.count(Booking.id))
            .where(Booking.event_id == event.id)
            .where(Booking.status == "CONFIRMED")
        ).scalar() or 0
        
        result.append({
            "id": event.id,
            "title": event.title,
            "description": event.description,
            "capacity": event.capacity,
            "starts_at": event.starts_at.isoformat(),
            "activity_type": event.activity_type,
            "location": event.location,
            "address": event.address,
            "created_by": event.created_by,
            "host_name": user.display_name,
            "available_spots": event.capacity - occupied_count,
            "occupied_spots": occupied_count,
            "status": event.status,
            "cancellation_deadline_hours": event.cancellation_deadline_hours,
            "images": {
                "image_1": event.image_1,
                "image_2": event.image_2,
                "image_3": event.image_3
            },
            "tags": [
                {
                    "id": tag.id,
                    "name": tag.name,
                    "color": tag.color,
                    "description": tag.description
                }
                for tag in event_tags
            ]
        })
    
    return result

# ---------------------
# Image Upload Endpoints
# ---------------------
@app.post("/users/profile-picture")
async def update_profile_picture(
    profile_data: ProfilePictureUpdate,
    current_user: User = Depends(get_current_user),
    session=Depends(get_session)
):
    """Update user's profile picture. Requires authentication."""
    try:
        if profile_data.profile_picture:
            # Upload to Cloudinary and get URL
            success, message, new_url = validate_and_upload_image(
                profile_data.profile_picture,
                old_image_url=current_user.profile_picture,
                folder="shua/profile_pictures",
                max_size_mb=2
            )
            
            if not success:
                raise HTTPException(status_code=400, detail=message)
            
            # Update user's profile picture with Cloudinary URL
            current_user.profile_picture = new_url
        else:
            # If None or empty, remove profile picture
            if current_user.profile_picture:
                delete_image(current_user.profile_picture)
            current_user.profile_picture = None
        
        session.commit()
        
        return {
            "message": "Profile picture updated successfully",
            "profile_picture": current_user.profile_picture
        }
    except HTTPException:
        session.rollback()
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update profile picture: {str(e)}")

@app.put("/users/profile")
async def update_user_profile(
    profile_data: UserProfileUpdate,
    current_user: User = Depends(get_current_user),
    session=Depends(get_session)
):
    """Update user's profile (name, bio, gallery images). Requires authentication."""
    try:
        # Update display name if provided
        if profile_data.display_name is not None:
            current_user.display_name = profile_data.display_name
        
        # Update bio if provided
        if profile_data.bio is not None:
            current_user.bio = profile_data.bio
            
        # Update extended profile fields
        if profile_data.first_name is not None:
            current_user.first_name = profile_data.first_name
        if profile_data.last_name is not None:
            current_user.last_name = profile_data.last_name
        if profile_data.gender is not None:
            current_user.gender = profile_data.gender
        if profile_data.birthday is not None:
            current_user.birthday = profile_data.birthday
        if profile_data.sports is not None:
            current_user.sports = profile_data.sports
        
        # Validate and update gallery images if provided
        for field_name, image_data in [
            ("gallery_image_1", profile_data.gallery_image_1),
            ("gallery_image_2", profile_data.gallery_image_2),
            ("gallery_image_3", profile_data.gallery_image_3),
            ("gallery_image_4", profile_data.gallery_image_4),
            ("gallery_image_5", profile_data.gallery_image_5),
            ("gallery_image_6", profile_data.gallery_image_6)
        ]:
            if image_data is not None:
                # Get current image URL
                old_image_url = getattr(current_user, field_name)
                
                # Upload to Cloudinary and get URL
                success, message, new_url = validate_and_upload_image(
                    image_data,
                    old_image_url=old_image_url,
                    folder="shua/gallery",
                    max_size_mb=5
                )
                
                if not success:
                    raise HTTPException(status_code=400, detail=f"{field_name}: {message}")
                
                # Update with Cloudinary URL (or None if image was removed)
                setattr(current_user, field_name, new_url)
        
        session.commit()
        
        return {
            "message": "Profile updated successfully",
            "user": {
                "id": current_user.id,
                "display_name": current_user.display_name,
                "bio": current_user.bio,
                "gallery_image_1": current_user.gallery_image_1,
                "gallery_image_2": current_user.gallery_image_2,
                "gallery_image_3": current_user.gallery_image_3,
                "gallery_image_3": current_user.gallery_image_3,
                "gallery_image_4": current_user.gallery_image_4,
                "gallery_image_5": current_user.gallery_image_5,
                "gallery_image_6": current_user.gallery_image_6,
                "first_name": current_user.first_name,
                "last_name": current_user.last_name,
                "gender": current_user.gender,
                "birthday": current_user.birthday,
                "sports": current_user.sports
            }
        }
    except HTTPException:
        session.rollback()
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update profile: {str(e)}")

@app.post("/events/{event_id}/images")
async def update_event_images(
    event_id: str,
    images_data: EventImagesUpdate,
    current_user: User = Depends(get_current_user),
    session=Depends(get_session)
):
    """Update event images (max 3). Requires authentication and event ownership."""
    try:
        # Get the event and verify ownership
        event = session.execute(select(Event).where(Event.id == event_id)).scalar_one_or_none()
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
        
        if event.created_by != current_user.id:
            raise HTTPException(status_code=403, detail="Only event creator can update images")
        
        # Validate and upload images if provided
        for field_name, image_data in [
            ("image_1", images_data.image_1),
            ("image_2", images_data.image_2), 
            ("image_3", images_data.image_3)
        ]:
            if image_data is not None:
                # Get current image URL
                old_image_url = getattr(event, field_name)
                
                # Upload to Cloudinary and get URL
                success, message, new_url = validate_and_upload_image(
                    image_data,
                    old_image_url=old_image_url,
                    folder="shua/events",
                    max_size_mb=3
                )
                
                if not success:
                    raise HTTPException(status_code=400, detail=f"{field_name}: {message}")
                
                # Update with Cloudinary URL (or None if image was removed)
                setattr(event, field_name, new_url)
            
        session.commit()
        
        return {
            "message": "Event images updated successfully",
            "event_id": event_id,
            "images": {
                "image_1": event.image_1,
                "image_2": event.image_2,
                "image_3": event.image_3
            }
        }
    except HTTPException:
        session.rollback()
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update event images: {str(e)}")

# POST /users endpoint removed - users should be created through /auth/register

def get_current_user_optional(
    authorization: str = Header(None, alias="Authorization"),
    session = Depends(get_session)
) -> Optional[User]:
    """Get the current user if authenticated, otherwise return None."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    
    try:
        token = authorization.split(" ")[1]
        payload = decode_access_token(token)
        if payload is None:
            return None
        
        user_id: str = payload.get("sub")
        if user_id is None:
            return None
        
        user = session.execute(select(User).where(User.id == user_id)).scalar_one_or_none()
        return user
    except Exception:
        return None

@app.get("/events")
async def list_events(
    session=Depends(get_session), 
    tag_filter: Optional[str] = None,
    activity_filter: Optional[str] = None,
    level_filter: Optional[str] = None,
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """Get all events, optionally filtered by tag, activity type, or level. Public endpoint - no authentication required."""
    try:
        query = select(Event)
        
        if tag_filter:
            # Filter events by tag name
            query = query.join(EventTag).join(Tag).where(Tag.name.ilike(f"%{tag_filter}%"))
        
        if activity_filter:
            # Filter events by activity type (sport name)
            query = query.where(Event.activity_type.ilike(f"%{activity_filter}%"))
        
        if level_filter:
            # Filter events by level requirement (level_1 to level_5)
            query = query.where(Event.level_requirement == level_filter)
        
        events = session.execute(query).scalars().all()
    except Exception as e:
        # Check if it's a "table doesn't exist" error
        error_msg = str(e).lower()
        if 'does not exist' in error_msg or 'no such table' in error_msg or 'relation' in error_msg:
            # Try to create tables
            try:
                print(f"⚠️  Tables missing, creating them now... Error: {e}")
                Base.metadata.create_all(bind=engine)
                # Return empty list after creating tables
                return []
            except Exception as create_error:
                print(f"❌ Failed to create tables: {create_error}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Database tables not initialized. Error: {str(create_error)}"
                )
        else:
            # Other database error
            print(f"❌ Database error in /events: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Database error: {str(e)}"
            )
    
    result = []
    for event in events:
        # Get host/creator information
        creator = session.execute(select(User).where(User.id == event.created_by)).scalar_one_or_none()
        
        # Get tags for this event
        event_tags = session.execute(
            select(Tag).join(EventTag).where(EventTag.event_id == event.id)
        ).scalars().all()
        
        # Get like count
        like_count = session.execute(
            select(func.count(EventLike.user_id)).where(EventLike.event_id == event.id)
        ).scalar_one()
        
        # Check if current user liked this event
        is_liked_by_user = False
        if current_user:
            is_liked_by_user = session.execute(
                select(EventLike).where(
                    EventLike.event_id == event.id,
                    EventLike.user_id == current_user.id
                )
            ).scalar_one_or_none() is not None
        
        result.append({
            "id": event.id,
            "title": event.title,
            "description": event.description,
            "capacity": event.capacity,
            "starts_at": event.starts_at.isoformat(),
            "time": event.starts_at.isoformat(),  # For backward compatibility
            "activity_type": event.activity_type,
            "sport_type": event.activity_type,  # For backward compatibility
            "location": event.location,
            "address": event.address,
            "created_by": event.created_by,
            "host_name": creator.display_name if creator else "Unknown Host",
            "available_spots": event.capacity,
            "occupied_spots": 0,  # TODO: Calculate from bookings
            "level_needed": "All Levels",  # Deprecated: use level_requirement instead
            "level_requirement": event.level_requirement,
            "auto_accept": event.auto_accept if event.auto_accept is not None else False,
            "status": event.status.value if event.status else "ACTIVE",
            "cancellation_deadline_hours": event.cancellation_deadline_hours,
            "like_count": like_count,
            "is_liked_by_user": is_liked_by_user,
            "images": {
                "image_1": event.image_1,
                "image_2": event.image_2,
                "image_3": event.image_3
            },
            "tags": [
                {
                    "id": tag.id,
                    "name": tag.name,
                    "color": tag.color,
                    "description": tag.description
                }
                for tag in event_tags
            ]
        })
    
    return result

@app.get("/events/liked")
async def get_liked_events(
    current_user: User = Depends(get_current_user),
    session=Depends(get_session)
):
    """Get all events liked by the current user."""
    try:
        # Join events with event_likes
        query = select(Event).join(EventLike).where(EventLike.user_id == current_user.id)
        events = session.execute(query).scalars().all()
        
        result = []
        for event in events:
            # Get host/creator information
            creator = session.execute(select(User).where(User.id == event.created_by)).scalar_one_or_none()
            
            # Get tags for this event
            event_tags = session.execute(
                select(Tag).join(EventTag).where(EventTag.event_id == event.id)
            ).scalars().all()
            
            # Get like count
            like_count = session.execute(
                select(func.count(EventLike.user_id)).where(EventLike.event_id == event.id)
            ).scalar_one()
            
            result.append({
                "id": event.id,
                "title": event.title,
                "description": event.description,
                "capacity": event.capacity,
                "starts_at": event.starts_at.isoformat(),
                "time": event.starts_at.isoformat(),
                "activity_type": event.activity_type,
                "sport_type": event.activity_type,
                "location": event.location,
                "address": event.address,
                "created_by": event.created_by,
                "host_name": creator.display_name if creator else "Unknown Host",
                "available_spots": event.capacity,
                "occupied_spots": 0,
                "level_needed": "All Levels",
                "auto_accept": event.auto_accept if event.auto_accept is not None else False,
                "status": event.status.value if event.status else "ACTIVE",
                "cancellation_deadline_hours": event.cancellation_deadline_hours,
                "like_count": like_count,
                "is_liked_by_user": True,
                "images": {
                    "image_1": event.image_1,
                    "image_2": event.image_2,
                    "image_3": event.image_3
                },
                "tags": [
                    {
                        "id": tag.id,
                        "name": tag.name,
                        "color": tag.color,
                        "description": tag.description
                    }
                    for tag in event_tags
                ]
            })
        
        return result
    except Exception as e:
        print(f"❌ Error fetching liked events: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch liked events: {str(e)}")

@app.get("/events/my")
async def get_my_events(
    current_user: User = Depends(get_current_user),
    session=Depends(get_session)
):
    """Get all events created by the current user. Requires authentication."""
    try:
        # Get events created by the current user
        user_events = session.execute(
            select(Event)
            .where(Event.created_by == current_user.id)
            .order_by(Event.starts_at.asc())
        ).scalars().all()
        
        result = []
        for event in user_events:
            # Get tags for this event
            event_tags = session.execute(
                select(Tag).join(EventTag).where(EventTag.event_id == event.id)
            ).scalars().all()
            
            result.append({
                "id": event.id,
                "title": event.title,
                "description": event.description,
                "capacity": event.capacity,
                "starts_at": event.starts_at.isoformat(),
                "time": event.starts_at.isoformat(),  # For backward compatibility
                "activity_type": event.activity_type,
                "sport_type": event.activity_type,  # For backward compatibility
                "location": event.location,
                "address": event.address,
                "created_by": event.created_by,
                "host_name": current_user.display_name,  # Current user is the host
                "available_spots": event.capacity,
                "occupied_spots": 0,  # TODO: Calculate from bookings
                "level_needed": "All Levels",  # TODO: Add to Event model if needed
                "auto_accept": event.auto_accept if event.auto_accept is not None else False,
                "images": {
                    "image_1": event.image_1,
                    "image_2": event.image_2,
                    "image_3": event.image_3
                },
                "tags": [
                    {
                        "id": tag.id,
                        "name": tag.name,
                        "color": tag.color,
                        "description": tag.description
                    }
                    for tag in event_tags
                ]
            })
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get user events: {str(e)}")

@app.get("/events/{event_id}")
async def get_event_by_id(
    event_id: str, 
    current_user: Optional[User] = Depends(get_current_user_optional),
    session=Depends(get_session)
):
    """Get a specific event by ID. Public endpoint - authentication optional."""
    try:
        event = session.execute(select(Event).where(Event.id == event_id)).scalar_one_or_none()
        
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
        
        # Get host/creator information
        creator = session.execute(select(User).where(User.id == event.created_by)).scalar_one_or_none()
        
        # Get tags for this event
        event_tags = session.execute(
            select(Tag).join(EventTag).where(EventTag.event_id == event.id)
        ).scalars().all()
        
        # Get like count
        like_count = session.execute(
            select(func.count(EventLike.user_id)).where(EventLike.event_id == event.id)
        ).scalar_one()
        
        # Check if current user liked this event
        is_liked_by_user = False
        if current_user:
            is_liked_by_user = session.execute(
                select(EventLike).where(
                    EventLike.event_id == event.id,
                    EventLike.user_id == current_user.id
                )
            ).scalar_one_or_none() is not None

        return {
            "id": event.id,
            "title": event.title,
            "description": event.description,
            "capacity": event.capacity,
            "starts_at": event.starts_at.isoformat(),
            "time": event.starts_at.isoformat(),  # For backward compatibility
            "activity_type": event.activity_type,
            "sport_type": event.activity_type,  # For backward compatibility
            "location": event.location,
            "address": event.address,
            "created_by": event.created_by,
            "host_name": creator.display_name if creator else "Unknown Host",
            "available_spots": event.capacity,
            "occupied_spots": 0,  # TODO: Calculate from bookings
            "level_needed": "All Levels",  # Deprecated: use level_requirement instead
            "level_requirement": event.level_requirement,
            "auto_accept": event.auto_accept if event.auto_accept is not None else False,
            "status": event.status.value if event.status else "ACTIVE",
            "cancellation_deadline_hours": event.cancellation_deadline_hours,
            "like_count": like_count,
            "is_liked_by_user": is_liked_by_user,
            "images": {
                "image_1": event.image_1,
                "image_2": event.image_2,
                "image_3": event.image_3
            },
            "tags": [
                {
                    "id": tag.id,
                    "name": tag.name,
                    "color": tag.color,
                    "description": tag.description
                }
                for tag in event_tags
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get event: {str(e)}")

@app.post("/events")
async def create_event(
    event_data: dict, 
    current_user: User = Depends(get_current_user),
    session=Depends(get_session)
):
    """Create a new event. Requires authentication."""
    try:
        title = event_data.get("title")
        description = event_data.get("description")
        starts_at_str = event_data.get("starts_at")
        capacity = event_data.get("capacity", 10)
        activity_type = event_data.get("activity_type")
        location = event_data.get("location")
        address = event_data.get("address")
        cancellation_deadline_hours = event_data.get("cancellation_deadline_hours", 24)
        level_requirement = event_data.get("level_requirement")  # Optional: "level_1" to "level_5"
        # Force use current user as creator for security
        created_by = current_user.id
        
        if not title:
            raise HTTPException(status_code=400, detail="title is required")
        
        if not starts_at_str:
            raise HTTPException(status_code=400, detail="starts_at is required")
        
        # Parse the datetime
        try:
            starts_at = dt.datetime.fromisoformat(starts_at_str.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid starts_at format")
        
        # Create new event
        new_event = Event(
            title=title,
            description=description,
            starts_at=starts_at,
            capacity=capacity,
            activity_type=activity_type,
            location=location,
            address=address,
            created_by=created_by,
            cancellation_deadline_hours=cancellation_deadline_hours,
            level_requirement=level_requirement
        )
        session.add(new_event)
        session.flush()  # Flush to get the event ID
        
        # Add tags if provided
        tag_ids = event_data.get("tag_ids", [])
        if tag_ids:
            for tag_id in tag_ids:
                # Verify tag exists
                tag = session.execute(select(Tag).where(Tag.id == tag_id)).scalar_one_or_none()
                if tag:
                    event_tag = EventTag(event_id=new_event.id, tag_id=tag_id)
                    session.add(event_tag)
        
        session.commit()
        session.refresh(new_event)
        
        # Get tags for response
        event_tags = session.execute(
            select(Tag).join(EventTag).where(EventTag.event_id == new_event.id)
        ).scalars().all()
        
        return {
            "id": new_event.id,
            "title": new_event.title,
            "description": new_event.description,
            "capacity": new_event.capacity,
            "starts_at": new_event.starts_at.isoformat(),
            "activity_type": new_event.activity_type,
            "location": new_event.location,
            "address": new_event.address,
            "created_by": new_event.created_by,
            "status": new_event.status,
            "cancellation_deadline_hours": new_event.cancellation_deadline_hours,
            "level_requirement": new_event.level_requirement,
            "images": {
                "image_1": new_event.image_1,
                "image_2": new_event.image_2,
                "image_3": new_event.image_3
            },
            "tags": [
                {
                    "id": tag.id,
                    "name": tag.name,
                    "color": tag.color,
                    "description": tag.description
                }
                for tag in event_tags
            ]
        }
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create event: {str(e)}")

@app.put("/events/{event_id}")
async def update_event(
    event_id: str,
    event_data: EventUpdate,
    current_user: User = Depends(get_current_user),
    session=Depends(get_session)
):
    """Update an event. Only the creator can update. Requires authentication."""
    try:
        # Get the event
        event = session.execute(select(Event).where(Event.id == event_id)).scalar_one_or_none()
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
        
        # Check if current user is the creator
        if event.created_by != current_user.id:
            raise HTTPException(status_code=403, detail="Only the event creator can update this event")
        
        # Update basic fields if provided
        if event_data.title is not None:
            event.title = event_data.title
        if event_data.description is not None:
            event.description = event_data.description
        if event_data.capacity is not None:
            event.capacity = event_data.capacity
        if event_data.activity_type is not None:
            event.activity_type = event_data.activity_type
        if event_data.location is not None:
            event.location = event_data.location
        if event_data.address is not None:
            event.address = event_data.address
        if event_data.level_requirement is not None:
            event.level_requirement = event_data.level_requirement
        
        # Update datetime if provided
        if event_data.starts_at is not None:
            try:
                starts_at = dt.datetime.fromisoformat(event_data.starts_at.replace('Z', '+00:00'))
                event.starts_at = starts_at
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid starts_at format")
        
        # Validate and update images if provided
        for field_name, image_data in [
            ("image_1", event_data.image_1),
            ("image_2", event_data.image_2),
            ("image_3", event_data.image_3)
        ]:
            if image_data is not None:
                # Get current image URL
                old_image_url = getattr(event, field_name)
                
                # Upload to Cloudinary and get URL
                success, message, new_url = validate_and_upload_image(
                    image_data,
                    old_image_url=old_image_url,
                    folder="shua/events",
                    max_size_mb=5
                )
                
                if not success:
                    raise HTTPException(status_code=400, detail=f"{field_name}: {message}")
                
                # Update with Cloudinary URL (or None if image was removed)
                setattr(event, field_name, new_url)
        
        # Update tags if provided
        if event_data.tag_ids is not None:
            # Remove existing tags
            session.execute(
                delete(EventTag).where(EventTag.event_id == event_id)
            )
            
            # Add new tags
            for tag_id in event_data.tag_ids:
                # Verify tag exists
                tag = session.execute(select(Tag).where(Tag.id == tag_id)).scalar_one_or_none()
                if tag:
                    event_tag = EventTag(event_id=event_id, tag_id=tag_id)
                    session.add(event_tag)
        
        session.commit()
        session.refresh(event)
        
        # Get tags for response
        event_tags = session.execute(
            select(Tag).join(EventTag).where(EventTag.event_id == event.id)
        ).scalars().all()
        
        return {
            "message": "Event updated successfully",
            "event": {
                "id": event.id,
                "title": event.title,
                "description": event.description,
                "capacity": event.capacity,
                "starts_at": event.starts_at.isoformat(),
                "activity_type": event.activity_type,
                "location": event.location,
                "address": event.address,
                "created_by": event.created_by,
                "level_requirement": event.level_requirement,
                "images": {
                    "image_1": event.image_1,
                    "image_2": event.image_2,
                    "image_3": event.image_3
                },
                "tags": [
                    {
                        "id": tag.id,
                        "name": tag.name,
                        "color": tag.color,
                        "description": tag.description
                    }
                    for tag in event_tags
                ]
            }
        }
    except HTTPException:
        session.rollback()
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update event: {str(e)}")

async def cancel_event(
    event_id: str,
    current_user: User = Depends(get_current_user),
    session=Depends(get_session)
):
    """Cancel an event. Only the creator can cancel. Requires authentication."""
    try:
        # Get the event
        event = session.execute(select(Event).where(Event.id == event_id)).scalar_one_or_none()
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
        
        # Check if current user is the creator
        if event.created_by != current_user.id:
            raise HTTPException(status_code=403, detail="Only the event creator can cancel this event")
        
        if event.status == EventStatus.CANCELED:
            raise HTTPException(status_code=400, detail="Event is already canceled")
            
        # Update status
        event.status = EventStatus.CANCELED
        
        # Notify participants
        thread = session.execute(
            select(Thread).where(Thread.event_id == event_id)
        ).scalar_one_or_none()
        
        if thread:
            thread.is_locked = True
            await system_message(session, thread.id, f"⚠️ Event has been canceled by the host.")
        
        # Get all participants (users with accepted requests for this event)
        accepted_requests = session.execute(
            select(Request)
            .where(Request.event_id == event_id)
            .where(Request.status == RequestStatus.ACCEPTED)
        ).scalars().all()
        
        # Send cancellation notifications to all participants (excluding the host)
        for req in accepted_requests:
            if req.guest_id != current_user.id:  # Don't notify the host who canceled
                await create_notification(
                    session,
                    user_id=req.guest_id,
                    notification_type=NotificationType.EVENT_CANCELED,
                    title=f"Event canceled: '{event.title}'",
                    body=f"The event '{event.title}' has been canceled by the host",
                    event_id=event_id,
                    request_id=req.id,
                    thread_id=thread.id if thread else None
                )
            
        session.commit()
        
        return {"message": "Event canceled successfully", "event_id": event.id}
    except HTTPException:
        session.rollback()
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to cancel event: {str(e)}")

@app.post("/events/{event_id}/like")
async def like_event(
    event_id: str,
    current_user: User = Depends(get_current_user),
    session=Depends(get_session)
):
    """Like an event."""
    try:
        # Check if event exists
        event = session.execute(select(Event).where(Event.id == event_id)).scalar_one_or_none()
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
            
        # Check if already liked
        existing_like = session.execute(
            select(EventLike).where(
                EventLike.event_id == event_id,
                EventLike.user_id == current_user.id
            )
        ).scalar_one_or_none()
        
        if existing_like:
            # Already liked, just return current count
            pass
        else:
            # Create new like
            new_like = EventLike(event_id=event_id, user_id=current_user.id)
            session.add(new_like)
            session.commit()
            
        # Get updated like count
        like_count = session.execute(
            select(func.count(EventLike.user_id)).where(EventLike.event_id == event_id)
        ).scalar_one()
        
        return {"message": "Event liked", "like_count": like_count}
    except HTTPException:
        session.rollback()
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to like event: {str(e)}")

@app.delete("/events/{event_id}/like")
async def unlike_event(
    event_id: str,
    current_user: User = Depends(get_current_user),
    session=Depends(get_session)
):
    """Unlike an event."""
    try:
        # Check if event exists
        event = session.execute(select(Event).where(Event.id == event_id)).scalar_one_or_none()
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
            
        # Delete like if exists
        session.execute(
            delete(EventLike).where(
                EventLike.event_id == event_id,
                EventLike.user_id == current_user.id
            )
        )
        session.commit()
            
        # Get updated like count
        like_count = session.execute(
            select(func.count(EventLike.user_id)).where(EventLike.event_id == event_id)
        ).scalar_one()
        
        return {"message": "Event unliked", "like_count": like_count}
    except HTTPException:
        session.rollback()
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to unlike event: {str(e)}")



@app.post("/rsvps")
async def create_rsvp(
    event_id: str, 
    current_user: User = Depends(get_current_user),
    session=Depends(get_session)
):
    """Create an RSVP for an event. Requires authentication."""
    # For now, just return success - you can implement full RSVP logic later
    return {"message": "RSVP created", "event_id": event_id, "user_id": current_user.id}
# Endpoints
# ---------------------
@app.get("/requests")
async def get_requests(
    current_user: User = Depends(get_current_user),
    session=Depends(get_session)
):
    """Get all requests where the user is either a guest or host. Requires authentication."""
    try:
        # Get requests where user is guest or host
        user_requests = session.execute(
            select(Request)
            .where((Request.guest_id == current_user.id) | (Request.host_id == current_user.id))
            .order_by(Request.created_at.desc())
        ).scalars().all()
        
        # Convert to response format with user names
        result = []
        for req in user_requests:
            guest = session.execute(select(User).where(User.id == req.guest_id)).scalar_one_or_none()
            host = session.execute(select(User).where(User.id == req.host_id)).scalar_one_or_none()
            
            result.append({
                "id": req.id,
                "event_id": req.event_id,
                "user_id": req.guest_id,
                "host_id": req.host_id,
                "status": req.status,
                "user_name": guest.display_name if guest else "Unknown User",
                "host_name": host.display_name if host else "Unknown Host",
                "event_title": "Event",  # We could join with events table if needed
                "created_at": req.created_at.isoformat()
            })
        
        return result
    except Exception as e:
        raise HTTPException(500, f"Failed to get requests: {str(e)}")

@app.get("/requests/all")
async def get_all_requests(
    current_user: User = Depends(get_current_user),
    session=Depends(get_session)
):
    """Get all requests where user is involved (guest or host). Requires authentication."""
    try:
        # Get requests where user is guest or host
        all_requests = session.execute(
            select(Request)
            .where((Request.guest_id == current_user.id) | (Request.host_id == current_user.id))
            .order_by(Request.created_at.desc())
        ).scalars().all()
        
        # Convert to response format with user names
        result = []
        for req in all_requests:
            guest = session.execute(select(User).where(User.id == req.guest_id)).scalar_one_or_none()
            host = session.execute(select(User).where(User.id == req.host_id)).scalar_one_or_none()
            
            result.append({
                "id": req.id,
                "event_id": req.event_id,
                "user_id": req.guest_id,
                "host_id": req.host_id,
                "status": req.status,
                "user_name": guest.display_name if guest else "Unknown User",
                "host_name": host.display_name if host else "Unknown Host",
                "event_title": "Event",  # We could join with events table if needed
                "created_at": req.created_at.isoformat()
            })
        
        return result
    except Exception as e:
        raise HTTPException(500, f"Failed to get all requests: {str(e)}")

@app.post("/requests")
async def create_request(
    payload: RequestCreate, 
    current_user: User = Depends(get_current_user),
    session=Depends(get_session)
):
    """Create a request for an event. Requires authentication."""
    try:
        # Validate that the event exists
        event = session.execute(select(Event).where(Event.id == payload.event_id)).scalar_one_or_none()
        if not event:
            raise HTTPException(404, "Event not found")
        
        # Use current user as guest (requester)
        guest_id = current_user.id
        
        # Check if user already has a request for this event
        existing_request = session.execute(
            select(Request).where(
                Request.event_id == payload.event_id,
                Request.guest_id == current_user.id
            )
        ).scalar_one_or_none()
        
        if existing_request:
            # If existing request is canceled, declined, or expired, allow "re-requesting" by updating it
            if existing_request.status in [RequestStatus.CANCELED, RequestStatus.DECLINED, RequestStatus.EXPIRED]:
                 # Reuse the existing request object
                 req = existing_request
                 req.status = RequestStatus.SUBMITTED
                 req.created_at = dt.datetime.now(dt.timezone.utc) # Update timestamp
                 
                 # Determine auto-accept strictly on the server side
                 should_auto_accept = bool(payload.auto_accept) or bool(getattr(event, "auto_accept", False))
                 req.auto_accept = should_auto_accept

                 # If auto-accept is enabled or event is auto-accept
                 if req.auto_accept:
                    req.status = RequestStatus.ACCEPTED
                    
                    # Create booking (check if one exists first to avoid duplicates/errors)
                    existing_booking = session.execute(select(Booking).where(Booking.request_id == req.id)).scalar_one_or_none()
                    if existing_booking:
                        existing_booking.status = BookingStatus.CONFIRMED
                    else:
                        booking = Booking(request_id=req.id, status=BookingStatus.CONFIRMED)
                        session.add(booking)
                    
                    session.flush()

                    # Find or create the event group chat
                    thread = session.execute(
                        select(Thread).where(Thread.event_id == payload.event_id)
                    ).scalar_one_or_none()
                    
                    if not thread:
                        # Create new group chat for this event
                        thread = Thread(scope=ThreadScope.BOOKING, event_id=payload.event_id)
                        session.add(thread)
                        session.flush()
                        
                        # Add the host as the first participant
                        host_participant = ThreadParticipant(thread_id=thread.id, user_id=event.created_by, role="host")
                        session.add(host_participant)
                        
                        await system_message(session, thread.id, f"Welcome to the {event.title} event chat!")
                    
                    # Add the requesting user to the group chat
                    existing_participant = session.execute(
                        select(ThreadParticipant).where(
                            ThreadParticipant.thread_id == thread.id,
                            ThreadParticipant.user_id == current_user.id
                        )
                    ).scalar_one_or_none()
                    
                    if not existing_participant:
                        guest_participant = ThreadParticipant(thread_id=thread.id, user_id=current_user.id, role="guest")
                        session.add(guest_participant)
                        
                        guest_name = current_user.display_name
                        await system_message(session, thread.id, f"{guest_name} has joined the event!")
                 
                 session.commit()
                 
                 # Return same structure as new request
                 thread = session.execute(select(Thread).where(Thread.event_id == payload.event_id)).scalar_one_or_none()
                 return {
                    "request_id": req.id,
                    "thread_id": thread.id if thread else None,
                    "status": req.status
                 }
            else:
                raise HTTPException(409, "You already have a request for this event")
        
        # Determine auto-accept strictly on the server side
        should_auto_accept = bool(payload.auto_accept) or bool(getattr(event, "auto_accept", False))
        
        # Create the request - use event creator as host
        req = Request(event_id=payload.event_id, guest_id=current_user.id, host_id=event.created_by, auto_accept=should_auto_accept)
        session.add(req)
        session.flush()

        # Note: Thread will be created when first request is accepted, not when request is made
        # This ensures the group chat only exists when there are actual participants

        # auto-accept: immediately create booking and add to group chat
        if req.auto_accept:
            req.status = RequestStatus.ACCEPTED
            booking = Booking(request_id=req.id, status=BookingStatus.CONFIRMED)
            session.add(booking)
            session.flush()
            
            # Find or create the event group chat
            thread = session.execute(
                select(Thread).where(Thread.event_id == payload.event_id)
            ).scalar_one_or_none()
            
            if not thread:
                # Create new group chat for this event
                thread = Thread(scope=ThreadScope.BOOKING, event_id=payload.event_id)
                session.add(thread)
                session.flush()
                
                # Add the host as the first participant
                host_participant = ThreadParticipant(thread_id=thread.id, user_id=event.created_by, role="host")
                session.add(host_participant)
                
                await system_message(session, thread.id, f"Welcome to the {event.title} event chat!")
            
            # Add the requesting user to the group chat
            existing_participant = session.execute(
                select(ThreadParticipant).where(
                    ThreadParticipant.thread_id == thread.id,
                    ThreadParticipant.user_id == current_user.id
                )
            ).scalar_one_or_none()
            
            if not existing_participant:
                guest_participant = ThreadParticipant(thread_id=thread.id, user_id=current_user.id, role="guest")
                session.add(guest_participant)
                
                guest_name = current_user.display_name
                await system_message(session, thread.id, f"{guest_name} has joined the event!")
            
            session.flush()

        # Create notifications
        if req.auto_accept:
            # Notify host that someone joined their event
            await create_notification(
                session,
                user_id=req.host_id,
                notification_type=NotificationType.EVENT_JOINED,
                title=f"{current_user.display_name} joined your event",
                body=f"{current_user.display_name} joined '{event.title}'",
                event_id=req.event_id,
                request_id=req.id,
                related_user_id=current_user.id
            )
        else:
            # Notify host that someone requested to join
            await create_notification(
                session,
                user_id=req.host_id,
                notification_type=NotificationType.EVENT_JOIN_REQUEST,
                title=f"New join request for '{event.title}'",
                body=f"{current_user.display_name} wants to join your event",
                event_id=req.event_id,
                request_id=req.id,
                related_user_id=current_user.id
            )
        
        session.commit()
        thread_id = thread.id if req.auto_accept else None
        return {"request_id": req.id, "thread_id": thread_id, "status": req.status}
    
    except HTTPException:
        session.rollback()
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(500, f"Failed to create request: {str(e)}")

@app.post("/requests/{request_id}/act")
async def act_on_request(
    request_id: str, 
    payload: RequestAction, 
    current_user: User = Depends(get_current_user),
    session=Depends(get_session)
):
    """Accept or decline a request. Requires authentication."""
    try:
        req = session.execute(select(Request).where(Request.id == request_id)).scalar_one_or_none()
        if not req:
            raise HTTPException(404, "Request not found")
        if current_user.id != req.host_id:
            raise HTTPException(403, "Only the host can approve or decline requests")

        if payload.action == "accept":
            # Check if request is already processed
            if req.status != RequestStatus.SUBMITTED:
                raise HTTPException(400, f"Request is already {req.status.lower()}")
            
            # Check if booking already exists (e.g., from auto-accept or race condition)
            booking = session.execute(
                select(Booking).where(Booking.request_id == req.id)
            ).scalar_one_or_none()
            
            if not booking:
                booking = Booking(request_id=req.id, status=BookingStatus.CONFIRMED)
                session.add(booking)
                session.flush()
            
            req.status = RequestStatus.ACCEPTED
            
            # Find or create the event group chat
            thread = session.execute(
                select(Thread).where(Thread.event_id == req.event_id)
            ).scalar_one_or_none()
            
            # Get event details for better messages
            event = session.execute(select(Event).where(Event.id == req.event_id)).scalar_one_or_none()
            event_title = event.title if event else "this event"
            
            if not thread:
                # Create new group chat for this event (first accepted participant)
                thread = Thread(scope=ThreadScope.BOOKING, event_id=req.event_id)
                session.add(thread)
                session.flush()
                
                # Add the host as the first participant
                host_participant = ThreadParticipant(thread_id=thread.id, user_id=req.host_id, role="host")
                session.add(host_participant)
                
                await system_message(session, thread.id, f"Welcome to the {event_title} event chat!")
            else:
                # If thread exists but was locked (e.g., from a previous decline), unlock it
                # Threads should only be locked when the event is canceled
                if thread.is_locked:
                    # Check if event is still active - only unlock if event is not canceled
                    if event and event.status != EventStatus.CANCELED:
                        thread.is_locked = False
                        await system_message(session, thread.id, "Chat is now available.")
            
            # Get guest name for notifications and messages
            guest = session.execute(select(User).where(User.id == req.guest_id)).scalar_one_or_none()
            guest_name = guest.display_name if guest else "A participant"
            
            # Add the accepted user to the group chat
            existing_participant = session.execute(
                select(ThreadParticipant).where(
                    ThreadParticipant.thread_id == thread.id,
                    ThreadParticipant.user_id == req.guest_id
                )
            ).scalar_one_or_none()
            
            if not existing_participant:
                guest_participant = ThreadParticipant(thread_id=thread.id, user_id=req.guest_id, role="guest")
                session.add(guest_participant)
                await system_message(session, thread.id, f"{guest_name} has joined the event!")
            
            # Notify guest that their request was accepted
            await create_notification(
                session,
                user_id=req.guest_id,
                notification_type=NotificationType.EVENT_JOIN_ACCEPTED,
                title=f"Request accepted for '{event_title}'",
                body=f"Your request to join '{event_title}' has been accepted!",
                event_id=req.event_id,
                request_id=req.id,
                thread_id=thread.id
            )
            
            # Notify host that someone joined (if not already notified)
            await create_notification(
                session,
                user_id=req.host_id,
                notification_type=NotificationType.EVENT_JOINED,
                title=f"{guest_name} joined your event",
                body=f"{guest_name} joined '{event_title}'",
                event_id=req.event_id,
                request_id=req.id,
                thread_id=thread.id,
                related_user_id=req.guest_id
            )
            
            session.flush()
            session.commit()
            return {"status": req.status, "booking_id": booking.id, "thread_id": thread.id}

        elif payload.action == "decline":
            # Check if request is already processed
            if req.status != RequestStatus.SUBMITTED:
                raise HTTPException(400, f"Request is already {req.status.lower()}")
            
            req.status = RequestStatus.DECLINED
            
            # Get event details for notification
            event = session.execute(select(Event).where(Event.id == req.event_id)).scalar_one_or_none()
            event_title = event.title if event else "this event"
            
            # Find thread if exists
            thread = session.execute(
                select(Thread).where(Thread.event_id == req.event_id)
            ).scalar_one_or_none()
            
            # Note: We don't lock the thread when a request is declined because:
            # 1. The declined user is not a participant, so they can't chat anyway
            # 2. Other participants should still be able to chat
            # 3. If the user rejoins and gets accepted later, they should be able to chat
            # Threads are only locked when the event itself is canceled
            
            # Notify guest that their request was declined
            await create_notification(
                session,
                user_id=req.guest_id,
                notification_type=NotificationType.EVENT_JOIN_DECLINED,
                title=f"Request declined for '{event_title}'",
                body=f"Your request to join '{event_title}' was declined",
                event_id=req.event_id,
                request_id=req.id
            )
            
            session.commit()
            return {"status": req.status, "thread_id": thread.id if thread else None}

        else:
            raise HTTPException(400, "Invalid action. Must be 'accept' or 'decline'")
    
    except HTTPException:
        session.rollback()
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(500, f"Failed to process request action: {str(e)}")

@app.post("/requests/{request_id}/cancel")
async def cancel_request(
    request_id: str,
    current_user: User = Depends(get_current_user),
    session=Depends(get_session)
):
    """Cancel a request or leave an event. Requires authentication."""
    try:
        req = session.execute(select(Request).where(Request.id == request_id)).scalar_one_or_none()
        if not req:
            raise HTTPException(404, "Request not found")
            
        if req.guest_id != current_user.id:
            raise HTTPException(403, "Only the requester can cancel their request")
            
        if req.status in [RequestStatus.CANCELED, RequestStatus.DECLINED, RequestStatus.EXPIRED]:
            raise HTTPException(400, f"Request is already {req.status.lower()}")
            
        # If already accepted (booked), check deadline
        if req.status == RequestStatus.ACCEPTED:
            event = session.execute(select(Event).where(Event.id == req.event_id)).scalar_one_or_none()
            if not event:
                raise HTTPException(404, "Event not found")
                
            # Check deadline
            if event.starts_at:
                # Check if starts_at is offset-aware
                starts_at = event.starts_at
                if starts_at.tzinfo is None:
                    starts_at = starts_at.replace(tzinfo=dt.timezone.utc)
                
                deadline = starts_at - timedelta(hours=event.cancellation_deadline_hours)
                now = dt.datetime.now(dt.timezone.utc)
                
                if now > deadline:
                    raise HTTPException(400, f"Cannot cancel less than {event.cancellation_deadline_hours} hours before event start")
            
            # Update booking status
            booking = session.execute(select(Booking).where(Booking.request_id == req.id)).scalar_one_or_none()
            if booking:
                booking.status = BookingStatus.CANCELED_BY_GUEST
                
            # Notify group chat
            thread = session.execute(
                select(Thread).where(Thread.event_id == req.event_id)
            ).scalar_one_or_none()
            
            if thread:
                # Remove from participants
                participant = session.execute(
                    select(ThreadParticipant).where(
                        ThreadParticipant.thread_id == thread.id,
                        ThreadParticipant.user_id == current_user.id
                    )
                ).scalar_one_or_none()
                
                if participant:
                    session.delete(participant)
                    
                    # Get user name for message
                    user = session.execute(select(User).where(User.id == current_user.id)).scalar_one_or_none()
                    user_name = user.display_name if user else "A participant"
                    
                    # Send system message BEFORE committing transaction
                    # We'll queue it to be sent after commit or handle it here?
                    # system_message is async and likely does db writes. 
                    # We should probably not delete participant yet if we want them to trigger a message?
                    # No, system message is sent by system (no sender needed).
                    await system_message(session, thread.id, f"{user_name} has left the event.")
            
            # Notify host that someone left
            event = session.execute(select(Event).where(Event.id == req.event_id)).scalar_one_or_none()
            if event:
                await create_notification(
                    session,
                    user_id=req.host_id,
                    notification_type=NotificationType.EVENT_LEFT,
                    title=f"{user_name} left your event",
                    body=f"{user_name} left '{event.title}'",
                    event_id=req.event_id,
                    request_id=req.id,
                    thread_id=thread.id if thread else None,
                    related_user_id=current_user.id
                )

        req.status = RequestStatus.CANCELED
        session.commit()
        
        return {"message": "Request canceled successfully", "status": req.status}
        
    except HTTPException:
        session.rollback()
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(500, f"Failed to cancel request: {str(e)}")

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

@app.post("/dev/seed-tags")
async def seed_tags(session=Depends(get_session)):
    """Endpoint to seed the database with sport-specific tags."""
    sample_tags = [
        # Tennis tags
        {"name": "Competitive", "sport_type": "Tennis", "color": "#dc2626", "description": "Competitive tennis match"},
        {"name": "Casual", "sport_type": "Tennis", "color": "#10b981", "description": "Casual tennis play"},
        {"name": "Singles", "sport_type": "Tennis", "color": "#3b82f6", "description": "Singles match"},
        {"name": "Doubles", "sport_type": "Tennis", "color": "#8b5cf6", "description": "Doubles match"},
        {"name": "Beginner-Friendly", "sport_type": "Tennis", "color": "#10b981", "description": "Welcoming to beginners"},
        {"name": "Advanced Players", "sport_type": "Tennis", "color": "#ef4444", "description": "For advanced players"},
        
        # Badminton tags
        {"name": "Competitive", "sport_type": "Badminton", "color": "#dc2626", "description": "Competitive badminton match"},
        {"name": "Casual", "sport_type": "Badminton", "color": "#10b981", "description": "Casual badminton play"},
        {"name": "Singles", "sport_type": "Badminton", "color": "#3b82f6", "description": "Singles match"},
        {"name": "Doubles", "sport_type": "Badminton", "color": "#8b5cf6", "description": "Doubles match"},
        {"name": "Beginner-Friendly", "sport_type": "Badminton", "color": "#10b981", "description": "Welcoming to beginners"},
        {"name": "Advanced Players", "sport_type": "Badminton", "color": "#ef4444", "description": "For advanced players"},
        
        # Golf tags
        {"name": "9 Holes", "sport_type": "Golf", "color": "#10b981", "description": "9 hole round"},
        {"name": "18 Holes", "sport_type": "Golf", "color": "#3b82f6", "description": "Full 18 hole round"},
        {"name": "Driving Range", "sport_type": "Golf", "color": "#f59e0b", "description": "Practice at driving range"},
        {"name": "Beginner-Friendly", "sport_type": "Golf", "color": "#10b981", "description": "Welcoming to beginners"},
        {"name": "Competitive", "sport_type": "Golf", "color": "#dc2626", "description": "Competitive round"},
        
        # Bouldering tags
        {"name": "Indoor", "sport_type": "Bouldering", "color": "#3b82f6", "description": "Indoor climbing gym"},
        {"name": "Outdoor", "sport_type": "Bouldering", "color": "#059669", "description": "Outdoor bouldering"},
        {"name": "Beginner-Friendly", "sport_type": "Bouldering", "color": "#10b981", "description": "Welcoming to beginners"},
        {"name": "Advanced Routes", "sport_type": "Bouldering", "color": "#ef4444", "description": "Advanced difficulty routes"},
        {"name": "Top Rope", "sport_type": "Bouldering", "color": "#8b5cf6", "description": "Top rope climbing"},
        
        # Pickleball tags
        {"name": "Competitive", "sport_type": "Pickleball", "color": "#dc2626", "description": "Competitive pickleball"},
        {"name": "Casual", "sport_type": "Pickleball", "color": "#10b981", "description": "Casual pickleball play"},
        {"name": "Doubles", "sport_type": "Pickleball", "color": "#8b5cf6", "description": "Doubles match"},
        {"name": "Beginner-Friendly", "sport_type": "Pickleball", "color": "#10b981", "description": "Welcoming to beginners"},
        
        # Gym tags
        {"name": "Strength Training", "sport_type": "Gym", "color": "#dc2626", "description": "Strength and resistance training"},
        {"name": "Cardio", "sport_type": "Gym", "color": "#f59e0b", "description": "Cardio workout"},
        {"name": "CrossFit", "sport_type": "Gym", "color": "#ef4444", "description": "CrossFit workout"},
        {"name": "Beginner-Friendly", "sport_type": "Gym", "color": "#10b981", "description": "Welcoming to beginners"},
        {"name": "Partner Workout", "sport_type": "Gym", "color": "#8b5cf6", "description": "Partner or buddy workout"},
        
        # Hiking tags
        {"name": "Easy Trail", "sport_type": "Hiking", "color": "#10b981", "description": "Easy difficulty trail"},
        {"name": "Moderate Trail", "sport_type": "Hiking", "color": "#f59e0b", "description": "Moderate difficulty trail"},
        {"name": "Difficult Trail", "sport_type": "Hiking", "color": "#ef4444", "description": "Difficult/challenging trail"},
        {"name": "Scenic", "sport_type": "Hiking", "color": "#3b82f6", "description": "Scenic views"},
        {"name": "Dog-Friendly", "sport_type": "Hiking", "color": "#8b5cf6", "description": "Dogs welcome"},
        
        # Ski/Snowboarding tags
        {"name": "Beginner Slopes", "sport_type": "Ski/Snowboarding", "color": "#10b981", "description": "Beginner/green slopes"},
        {"name": "Intermediate", "sport_type": "Ski/Snowboarding", "color": "#3b82f6", "description": "Intermediate/blue slopes"},
        {"name": "Advanced", "sport_type": "Ski/Snowboarding", "color": "#ef4444", "description": "Advanced/black slopes"},
        {"name": "Backcountry", "sport_type": "Ski/Snowboarding", "color": "#dc2626", "description": "Backcountry skiing/boarding"},
        {"name": "Park", "sport_type": "Ski/Snowboarding", "color": "#8b5cf6", "description": "Terrain park"},
        
        # Basketball tags
        {"name": "Pickup Game", "sport_type": "Basketball", "color": "#dc2626", "description": "Casual pickup basketball game"},
        {"name": "Competitive", "sport_type": "Basketball", "color": "#ef4444", "description": "Competitive basketball"},
        {"name": "3-on-3", "sport_type": "Basketball", "color": "#3b82f6", "description": "3-on-3 basketball"},
        {"name": "5-on-5", "sport_type": "Basketball", "color": "#8b5cf6", "description": "Full court 5-on-5"},
        {"name": "Shooting Practice", "sport_type": "Basketball", "color": "#10b981", "description": "Shooting and practice"},
        {"name": "Beginner-Friendly", "sport_type": "Basketball", "color": "#10b981", "description": "Welcoming to beginners"},
    ]
    
    created_tags = []
    for tag_data in sample_tags:
        # Check if tag already exists with same name and sport_type
        query = select(Tag).where(Tag.name == tag_data["name"])
        sport_type = tag_data.get("sport_type")
        if sport_type:
            query = query.where(Tag.sport_type == sport_type)
        else:
            query = query.where(Tag.sport_type.is_(None))
        
        existing_tag = session.execute(query).scalar_one_or_none()
        
        if not existing_tag:
            new_tag = Tag(
                name=tag_data["name"],
                sport_type=tag_data.get("sport_type"),
                color=tag_data["color"],
                description=tag_data["description"]
            )
            session.add(new_tag)
            created_tags.append(f"{tag_data['name']} ({tag_data.get('sport_type', 'General')})")
    
    session.commit()
    
    return {
        "message": f"Created {len(created_tags)} sport-specific tags",
        "created_tags": created_tags,
        "total_tags": len(sample_tags)
    }

@app.post("/dev/create-tables")
async def create_tables(session=Depends(get_session)):
    """Manually create all database tables."""
    try:
        Base.metadata.create_all(bind=engine)
        return {"status": "success", "message": "Tables created successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/dev/check-db")
async def check_database(session=Depends(get_session)):
    """Check database connection and show current tables."""
    try:
        # Test connection
        session.execute(select(1))
        
        # Get table names - PostgreSQL syntax
        tables = session.execute(select(text("tablename")).select_from(text("pg_tables")).where(text("schemaname='public'"))).scalars().all()
        
        return {
            "status": "connected",
            "database_url": DATABASE_URL,
            "tables": list(tables)
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "database_url": DATABASE_URL}

# ---------------------
# Smoke tests
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

# ---------------------
# Additional Schemas
# ---------------------
class EventSummaryOut(BaseModel):
    id: str
    title: str
    description: Optional[str]
    starts_at: str
    location: Optional[str]
    activity_type: Optional[str]

class LastMessageOut(BaseModel):
    id: str
    sender_id: Optional[str]
    sender_name: Optional[str]
    body: str
    created_at: str
    kind: MessageKind

class ThreadOut(BaseModel):
    id: str
    scope: ThreadScope
    event_id: str
    is_locked: bool
    created_at: str
    event: Optional[EventSummaryOut] = None
    last_message: Optional[LastMessageOut] = None
    unread_count: int = 0
    participant_count: int = 0

class ThreadParticipantOut(BaseModel):
    thread_id: str
    user_id: str
    user_name: str
    role: str

class MessageReadOut(BaseModel):
    thread_id: str
    user_id: str
    last_read_seq: int

class ThreadListOut(BaseModel):
    threads: list[ThreadOut]
    participants: dict[str, list[ThreadParticipantOut]]  # thread_id -> participants

# ---------------------
# Chat Endpoints
# ---------------------
@app.get("/threads", response_model=ThreadListOut)
async def get_user_threads(
    current_user: User = Depends(get_current_user),
    session=Depends(get_session)
):
    """Get all threads where the user is a participant with event details and last messages. Requires authentication."""
    # Get thread IDs where user is a participant
    participant_threads = session.execute(
        select(ThreadParticipant.thread_id)
        .where(ThreadParticipant.user_id == current_user.id)
    ).scalars().all()
    
    if not participant_threads:
        return ThreadListOut(threads=[], participants={})
    
    # Get thread details with events
    threads = session.execute(
        select(Thread)
        .where(Thread.id.in_(participant_threads))
        .order_by(Thread.id)
    ).scalars().all()
    
    # Build enhanced thread list
    enhanced_threads = []
    participants = {}
    
    for thread in threads:
        # Get event details
        event = session.execute(
            select(Event).where(Event.id == thread.event_id)
        ).scalar_one_or_none()
        
        event_summary = None
        if event:
            event_summary = EventSummaryOut(
                id=event.id,
                title=event.title,
                description=event.description,
                starts_at=event.starts_at.isoformat(),
                location=event.location,
                activity_type=event.activity_type
            )
        
        # Get last message
        last_message_record = session.execute(
            select(Message)
            .where(Message.thread_id == thread.id)
            .order_by(Message.seq.desc())
            .limit(1)
        ).scalar_one_or_none()
        
        last_message = None
        if last_message_record:
            # Get sender name
            sender_name = None
            if last_message_record.sender_id:
                sender = session.execute(
                    select(User).where(User.id == last_message_record.sender_id)
                ).scalar_one_or_none()
                sender_name = sender.display_name if sender else "Unknown User"
            
            last_message = LastMessageOut(
                id=last_message_record.id,
                sender_id=last_message_record.sender_id,
                sender_name=sender_name,
                body=last_message_record.body,
                created_at=last_message_record.created_at.isoformat(),
                kind=last_message_record.kind
            )
        
        # Get unread count
        user_read_status = session.execute(
            select(MessageRead)
            .where(MessageRead.thread_id == thread.id, MessageRead.user_id == current_user.id)
        ).scalar_one_or_none()
        
        last_read_seq = user_read_status.last_read_seq if user_read_status else 0
        unread_count = session.execute(
            select(func.count(Message.id))
            .where(Message.thread_id == thread.id, Message.seq > last_read_seq)
        ).scalar_one()
        
        # Get participants for this thread
        thread_participants = session.execute(
            select(ThreadParticipant, User)
            .join(User, ThreadParticipant.user_id == User.id)
            .where(ThreadParticipant.thread_id == thread.id)
        ).all()
        
        participants[thread.id] = [
            ThreadParticipantOut(
                thread_id=p.ThreadParticipant.thread_id, 
                user_id=p.ThreadParticipant.user_id, 
                user_name=p.User.display_name,
                role=p.ThreadParticipant.role
            )
            for p in thread_participants
        ]
        
        enhanced_threads.append(ThreadOut(
            id=thread.id,
            scope=thread.scope,
            event_id=thread.event_id,
            is_locked=thread.is_locked,
            created_at=thread.created_at.isoformat(),
            event=event_summary,
            last_message=last_message,
            unread_count=unread_count,
            participant_count=len(thread_participants)
        ))
    
    # Sort threads by last message time (most recent first)
    enhanced_threads.sort(
        key=lambda t: t.last_message.created_at if t.last_message else "1970-01-01T00:00:00",
        reverse=True
    )
    
    return ThreadListOut(
        threads=enhanced_threads,
        participants=participants
    )

@app.get("/threads/{thread_id}/messages")
async def get_thread_messages(
    thread_id: str, 
    limit: int = 50, 
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    session=Depends(get_session)
):
    """Get messages for a specific thread. Requires authentication."""
    # Verify user has access to thread
    participant = session.execute(
        select(ThreadParticipant)
        .where(ThreadParticipant.thread_id == thread_id, ThreadParticipant.user_id == current_user.id)
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
    
    return [serialize_message(m, session) for m in reversed(messages)]

@app.post("/threads/{thread_id}/messages")
async def send_message(
    thread_id: str,
    payload: ThreadMessageIn,
    current_user: User = Depends(get_current_user),
    session=Depends(get_session)
):
    """Send a message to a thread. Requires authentication."""
    try:
        # Get thread and verify it exists
        thread = session.execute(select(Thread).where(Thread.id == thread_id)).scalar_one_or_none()
        if not thread:
            raise HTTPException(404, "Thread not found")
        
        # Verify user has access to thread
        participant = session.execute(
            select(ThreadParticipant)
            .where(ThreadParticipant.thread_id == thread_id, ThreadParticipant.user_id == current_user.id)
        ).scalar_one_or_none()
        
        if not participant:
            raise HTTPException(403, "Access denied - you are not a participant in this thread")
        
        # Check if thread is locked
        if thread.is_locked:
            raise HTTPException(400, "Thread is locked")
        
        # Additional access control: Check if user is host or accepted participant
        if thread.scope == ThreadScope.REQUEST and thread.request_id:
            # For request threads, check if user is the host or has an accepted request
            request = session.execute(
                select(Request).where(Request.id == thread.request_id)
            ).scalar_one_or_none()
            
            if not request:
                raise HTTPException(404, "Request not found")
            
            # Allow host or accepted guest
            if current_user.id != request.host_id and request.status != RequestStatus.ACCEPTED:
                raise HTTPException(403, "Only the host or accepted participants can send messages")
        
        elif thread.scope == ThreadScope.BOOKING and thread.request_id:
            # For booking threads, check if user is the host or has an accepted request
            request = session.execute(
                select(Request).where(Request.id == thread.request_id)
            ).scalar_one_or_none()
            
            if not request:
                raise HTTPException(404, "Request not found")
            
            # Allow host or accepted guest
            if current_user.id != request.host_id and request.status != RequestStatus.ACCEPTED:
                raise HTTPException(403, "Only the host or accepted participants can send messages")
        
        # Create message
        message = Message(
            thread_id=thread_id,
            sender_id=current_user.id,
            client_msg_id=payload.client_msg_id,
            kind=MessageKind.USER,
            body=payload.body,
            seq=_next_seq(session, thread_id)
        )
        
        session.add(message)
        session.flush()
        
        # Get event details for notification
        event = session.execute(select(Event).where(Event.id == thread.event_id)).scalar_one_or_none()
        event_title = event.title if event else "Event"
        
        # Get all participants except the sender
        participants = session.execute(
            select(ThreadParticipant)
            .where(ThreadParticipant.thread_id == thread_id)
            .where(ThreadParticipant.user_id != current_user.id)
        ).scalars().all()
        
        # Create notifications for all other participants
        for participant in participants:
            await create_notification(
                session,
                user_id=participant.user_id,
                notification_type=NotificationType.NEW_MESSAGE,
                title=f"New message in '{event_title}'",
                body=f"{current_user.display_name}: {payload.body[:100]}{'...' if len(payload.body) > 100 else ''}",
                event_id=thread.event_id,
                thread_id=thread_id,
                related_user_id=current_user.id,
                metadata={"message_id": message.id}
            )
        
        session.commit()
        
        # Broadcast via WebSocket
        await manager.broadcast_to_thread({
            "type": "new_message",
            "message": serialize_message(message, session)
        }, thread_id, exclude_user=message.sender_id)
        
        return serialize_message(message, session)
    
    except HTTPException:
        session.rollback()
        raise
    except IntegrityError:
        session.rollback()
        raise HTTPException(400, "Duplicate client_msg_id")
    except Exception as e:
        session.rollback()
        raise HTTPException(500, f"Failed to send message: {str(e)}")

class MarkReadRequest(BaseModel):
    last_read_seq: int

@app.post("/threads/{thread_id}/read")
async def mark_messages_read(
    thread_id: str,
    request: MarkReadRequest,
    current_user: User = Depends(get_current_user),
    session=Depends(get_session)
):
    """Mark messages as read up to a specific sequence number. Requires authentication."""
    # Verify user has access to thread
    participant = session.execute(
        select(ThreadParticipant)
        .where(ThreadParticipant.thread_id == thread_id, ThreadParticipant.user_id == current_user.id)
    ).scalar_one_or_none()
    
    if not participant:
        raise HTTPException(403, "Access denied")
    
    # Update or create read status
    read_status = session.execute(
        select(MessageRead)
        .where(MessageRead.thread_id == thread_id, MessageRead.user_id == current_user.id)
    ).scalar_one_or_none()
    
    if read_status:
        read_status.last_read_seq = request.last_read_seq
    else:
        read_status = MessageRead(
            thread_id=thread_id,
            user_id=current_user.id,
            last_read_seq=request.last_read_seq
        )
        session.add(read_status)
    
    session.commit()
    return {"status": "success"}

@app.get("/threads/{thread_id}/participants")
async def get_thread_participants(
    thread_id: str,
    current_user: User = Depends(get_current_user),
    session=Depends(get_session)
):
    """Get participants for a specific thread. Requires authentication."""
    # Verify user has access to thread
    participant = session.execute(
        select(ThreadParticipant)
        .where(ThreadParticipant.thread_id == thread_id, ThreadParticipant.user_id == current_user.id)
    ).scalar_one_or_none()
    
    if not participant:
        raise HTTPException(403, "Access denied")
    
    # Get all participants with user details
    participants = session.execute(
        select(ThreadParticipant, User)
        .join(User, ThreadParticipant.user_id == User.id)
        .where(ThreadParticipant.thread_id == thread_id)
    ).all()
    
    return [
        ThreadParticipantOut(
            thread_id=p.ThreadParticipant.thread_id, 
            user_id=p.ThreadParticipant.user_id, 
            user_name=p.User.display_name,
            role=p.ThreadParticipant.role
        )
        for p in participants
    ]

@app.get("/threads/{thread_id}/details")
async def get_thread_details(
    thread_id: str,
    current_user: User = Depends(get_current_user),
    session=Depends(get_session)
):
    """Get detailed information about a thread including event details. Requires authentication."""
    # Verify user has access to thread
    participant = session.execute(
        select(ThreadParticipant)
        .where(ThreadParticipant.thread_id == thread_id, ThreadParticipant.user_id == current_user.id)
    ).scalar_one_or_none()
    
    if not participant:
        raise HTTPException(403, "Access denied")
    
    # Get thread details
    thread = session.execute(
        select(Thread).where(Thread.id == thread_id)
    ).scalar_one_or_none()
    
    if not thread:
        raise HTTPException(404, "Thread not found")
    
    # Get event details
    event = session.execute(
        select(Event).where(Event.id == thread.event_id)
    ).scalar_one_or_none()
    
    event_summary = None
    if event:
        # Get event creator details
        creator = session.execute(
            select(User).where(User.id == event.created_by)
        ).scalar_one_or_none()
        
        # Get event tags
        event_tags = session.execute(
            select(Tag).join(EventTag).where(EventTag.event_id == event.id)
        ).scalars().all()
        
        event_summary = {
            "id": event.id,
            "title": event.title,
            "description": event.description,
            "starts_at": event.starts_at.isoformat(),
            "location": event.location,
            "address": event.address,
            "activity_type": event.activity_type,
            "capacity": event.capacity,
            "created_by": event.created_by,
            "creator_name": creator.display_name if creator else "Unknown",
            "tags": [
                {
                    "id": tag.id,
                    "name": tag.name,
                    "color": tag.color,
                    "description": tag.description
                }
                for tag in event_tags
            ]
        }
    
    
    # Get all requests for this event to show request/booking details
    event_requests = session.execute(
        select(Request).where(Request.event_id == thread.event_id)
    ).scalars().all()
    
    requests_details = []
    for req in event_requests:
        guest = session.execute(select(User).where(User.id == req.guest_id)).scalar_one_or_none()
        host = session.execute(select(User).where(User.id == req.host_id)).scalar_one_or_none()
        
        booking_info = None
        if req.status == RequestStatus.ACCEPTED:
            booking = session.execute(
                select(Booking).where(Booking.request_id == req.id)
            ).scalar_one_or_none()
            if booking:
                booking_info = {
                    "id": booking.id,
                    "status": booking.status
                }
        
        requests_details.append({
            "id": req.id,
            "status": req.status,
            "guest_id": req.guest_id,
            "guest_name": guest.display_name if guest else "Unknown",
            "host_id": req.host_id,
            "host_name": host.display_name if host else "Unknown",
            "created_at": req.created_at.isoformat(),
            "booking": booking_info
        })
    
    return {
        "thread": {
            "id": thread.id,
            "scope": thread.scope,
            "is_locked": thread.is_locked,
            "event_id": thread.event_id,
            "created_at": thread.created_at.isoformat()
        },
        "event": event_summary,
        "requests": requests_details
    }

# ---------------------
# Notification API Endpoints
# ---------------------
@app.get("/notifications", response_model=NotificationListOut)
async def get_notifications(
    limit: int = 50,
    offset: int = 0,
    unread_only: bool = False,
    current_user: User = Depends(get_current_user),
    session=Depends(get_session)
):
    """Get notifications for the current user. Requires authentication."""
    query = select(Notification).where(Notification.user_id == current_user.id)
    
    if unread_only:
        query = query.where(Notification.is_read == False)
    
    query = query.order_by(Notification.created_at.desc()).limit(limit).offset(offset)
    
    notifications = session.execute(query).scalars().all()
    
    # Get unread count
    unread_count = session.execute(
        select(func.count(Notification.id))
        .where(Notification.user_id == current_user.id, Notification.is_read == False)
    ).scalar() or 0
    
    # Get total count
    total_count = session.execute(
        select(func.count(Notification.id))
        .where(Notification.user_id == current_user.id)
    ).scalar() or 0
    
    return {
        "notifications": [
            {
                "id": n.id,
                "user_id": n.user_id,
                "type": n.type,
                "title": n.title,
                "body": n.body,
                "is_read": n.is_read,
                "created_at": n.created_at,
                "event_id": n.event_id,
                "thread_id": n.thread_id,
                "request_id": n.request_id,
                "related_user_id": n.related_user_id,
                "metadata": n.extra_data
            }
            for n in notifications
        ],
        "unread_count": unread_count,
        "total_count": total_count
    }

@app.get("/notifications/unread-count")
async def get_unread_count(
    current_user: User = Depends(get_current_user),
    session=Depends(get_session)
):
    """Get unread notification count for the current user. Requires authentication."""
    count = session.execute(
        select(func.count(Notification.id))
        .where(Notification.user_id == current_user.id, Notification.is_read == False)
    ).scalar() or 0
    
    return {"unread_count": count}

@app.post("/notifications/mark-read")
async def mark_notifications_read(
    payload: NotificationMarkRead,
    current_user: User = Depends(get_current_user),
    session=Depends(get_session)
):
    """Mark notifications as read. Requires authentication."""
    # Verify all notifications belong to the current user
    notifications = session.execute(
        select(Notification)
        .where(
            Notification.id.in_(payload.notification_ids),
            Notification.user_id == current_user.id
        )
    ).scalars().all()
    
    for notification in notifications:
        notification.is_read = True
    
    session.commit()
    return {"message": f"Marked {len(notifications)} notification(s) as read"}

@app.post("/notifications/{notification_id}/mark-read")
async def mark_single_notification_read(
    notification_id: str,
    current_user: User = Depends(get_current_user),
    session=Depends(get_session)
):
    """Mark a single notification as read. Requires authentication."""
    notification = session.execute(
        select(Notification).where(
            Notification.id == notification_id,
            Notification.user_id == current_user.id
        )
    ).scalar_one_or_none()
    
    if not notification:
        raise HTTPException(404, "Notification not found")
    
    notification.is_read = True
    session.commit()
    return {"message": "Notification marked as read"}

@app.delete("/notifications/{notification_id}")
async def delete_notification(
    notification_id: str,
    current_user: User = Depends(get_current_user),
    session=Depends(get_session)
):
    """Delete a notification. Requires authentication."""
    notification = session.execute(
        select(Notification).where(
            Notification.id == notification_id,
            Notification.user_id == current_user.id
        )
    ).scalar_one_or_none()
    
    if not notification:
        raise HTTPException(404, "Notification not found")
    
    session.delete(notification)
    session.commit()
    return {"message": "Notification deleted"}

@app.delete("/notifications")
async def delete_all_notifications(
    read_only: bool = False,
    current_user: User = Depends(get_current_user),
    session=Depends(get_session)
):
    """Delete all notifications (optionally only read ones). Requires authentication."""
    query = select(Notification).where(Notification.user_id == current_user.id)
    
    if read_only:
        query = query.where(Notification.is_read == True)
    
    notifications = session.execute(query).scalars().all()
    count = len(notifications)
    
    for notification in notifications:
        session.delete(notification)
    
    session.commit()
    return {"message": f"Deleted {count} notification(s)"}

# ---------------------
# WebSocket Manager
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

# Old duplicate code removed - using main block at end of file

# Add static file serving
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Mount static files
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
async def read_index():
    return FileResponse("index.html")

@app.get("/health")
async def health_check():
    """Simple health check endpoint for Railway"""
    return {"status": "healthy", "message": "App is running"}

# ---------------------
# Tag Management Endpoints
# ---------------------

@app.get("/tags")
async def list_tags(
    sport: Optional[str] = None,
    session=Depends(get_session)
):
    """Get all available tags, optionally filtered by sport. Public endpoint (no authentication required)."""
    if sport:
        # Filter tags by sport_type (case-insensitive matching)
        # Handle variations like "Ski/Snowboard" vs "Ski/Snowboarding"
        sport_normalized = sport.strip()
        tags = session.execute(
            select(Tag).where(
                func.lower(Tag.sport_type) == func.lower(sport_normalized)
            ).order_by(Tag.name)
        ).scalars().all()
        
        # If no exact match, try partial match for variations
        if not tags and "/" in sport_normalized:
            # Try matching first part (e.g., "Ski" matches "Ski/Snowboarding")
            sport_part = sport_normalized.split("/")[0]
            tags = session.execute(
                select(Tag).where(
                    func.lower(Tag.sport_type).like(func.lower(f"{sport_part}%"))
                ).order_by(Tag.name)
            ).scalars().all()
    else:
        # Return all tags
        tags = session.execute(select(Tag).order_by(Tag.name)).scalars().all()
    
    return [
        {
            "id": tag.id,
            "name": tag.name,
            "sport_type": tag.sport_type,
            "color": tag.color,
            "description": tag.description,
            "created_at": tag.created_at.isoformat()
        }
        for tag in tags
    ]

@app.post("/tags")
async def create_tag(
    tag_data: TagCreate, 
    session=Depends(get_session)
):
    """Create a new tag. Public endpoint (no authentication required) to allow tag creation during registration."""
    try:
        # Check if tag with same name and sport_type already exists
        query = select(Tag).where(Tag.name == tag_data.name)
        if tag_data.sport_type:
            query = query.where(Tag.sport_type == tag_data.sport_type)
        else:
            query = query.where(Tag.sport_type.is_(None))
        
        existing_tag = session.execute(query).scalar_one_or_none()
        
        if existing_tag:
            raise HTTPException(
                status_code=400, 
                detail=f"Tag '{tag_data.name}' already exists for this sport"
            )
        
        # Generate a random color if not provided
        import random
        default_colors = ["#dc2626", "#10b981", "#3b82f6", "#8b5cf6", "#f59e0b", "#ef4444", "#059669"]
        tag_color = tag_data.color or random.choice(default_colors)
        
        new_tag = Tag(
            name=tag_data.name,
            sport_type=tag_data.sport_type,
            color=tag_color,
            description=tag_data.description
        )
        session.add(new_tag)
        session.commit()
        session.refresh(new_tag)
        
        return {
            "id": new_tag.id,
            "name": new_tag.name,
            "sport_type": new_tag.sport_type,
            "color": new_tag.color,
            "description": new_tag.description,
            "created_at": new_tag.created_at.isoformat()
        }
    except HTTPException:
        session.rollback()
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create tag: {str(e)}")

@app.delete("/tags/{tag_id}")
async def delete_tag(
    tag_id: str, 
    current_user: User = Depends(get_current_user),
    session=Depends(get_session)
):
    """Delete a tag. Requires authentication."""
    try:
        tag = session.execute(select(Tag).where(Tag.id == tag_id)).scalar_one_or_none()
        if not tag:
            raise HTTPException(status_code=404, detail="Tag not found")
        
        session.delete(tag)
        session.commit()
        
        return {"message": "Tag deleted successfully"}
    except HTTPException:
        session.rollback()
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete tag: {str(e)}")

@app.post("/events/{event_id}/tags")
async def add_tags_to_event(
    event_id: str, 
    tag_data: EventTagCreate,
    current_user: User = Depends(get_current_user),
    session=Depends(get_session)
):
    """Add tags to an event. Requires authentication."""
    try:
        # Verify event exists and user is the creator
        event = session.execute(select(Event).where(Event.id == event_id)).scalar_one_or_none()
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
        
        if event.created_by != current_user.id:
            raise HTTPException(status_code=403, detail="Only event creator can manage tags")
        
        # Add each tag
        for tag_id in tag_data.tag_ids:
            # Verify tag exists
            tag = session.execute(select(Tag).where(Tag.id == tag_id)).scalar_one_or_none()
            if not tag:
                continue  # Skip non-existent tags
            
            # Check if relationship already exists
            existing = session.execute(
                select(EventTag).where(
                    EventTag.event_id == event_id,
                    EventTag.tag_id == tag_id
                )
            ).scalar_one_or_none()
            
            if not existing:
                event_tag = EventTag(event_id=event_id, tag_id=tag_id)
                session.add(event_tag)
        
        session.commit()
        
        # Return updated event with tags
        event_tags = session.execute(
            select(Tag).join(EventTag).where(EventTag.event_id == event_id)
        ).scalars().all()
        
        return {
            "message": "Tags added successfully",
            "event_id": event_id,
            "tags": [
                {
                    "id": tag.id,
                    "name": tag.name,
                    "color": tag.color,
                    "description": tag.description
                }
                for tag in event_tags
            ]
        }
    except HTTPException:
        session.rollback()
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to add tags: {str(e)}")

@app.delete("/events/{event_id}/tags/{tag_id}")
async def remove_tag_from_event(
    event_id: str, 
    tag_id: str,
    current_user: User = Depends(get_current_user),
    session=Depends(get_session)
):
    """Remove a tag from an event. Requires authentication."""
    try:
        # Verify event exists and user is the creator
        event = session.execute(select(Event).where(Event.id == event_id)).scalar_one_or_none()
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
        
        if event.created_by != current_user.id:
            raise HTTPException(status_code=403, detail="Only event creator can manage tags")
        
        # Find the relationship
        event_tag = session.execute(
            select(EventTag).where(
                EventTag.event_id == event_id,
                EventTag.tag_id == tag_id
            )
        ).scalar_one_or_none()
        
        if not event_tag:
            raise HTTPException(status_code=404, detail="Tag not found on this event")
        
        session.delete(event_tag)
        session.commit()
        
        return {"message": "Tag removed from event successfully"}
    except HTTPException:
        session.rollback()
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to remove tag: {str(e)}")

# ---------------------
# Smoke tests
# ---------------------
async def _run_smoke_tests():
    from asgi_lifespan import LifespanManager
    from httpx import AsyncClient

    async with LifespanManager(app):
        async with AsyncClient(app=app, base_url="http://testserver") as client:
            # Test registration
            r = await client.post("/auth/register", json={
                "email": "test@example.com",
                "password": "password123",
                "display_name": "Test User"
            })
            r.raise_for_status()
            data = r.json()
            token = data["access_token"]
            
            # Test authenticated endpoint
            r = await client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
            r.raise_for_status()
            print("✅ Smoke tests passed!")

if __name__ == "__main__":
    import argparse, uvicorn
    p = argparse.ArgumentParser()
    p.add_argument("--test", action="store_true")
    p.add_argument("--port", type=int, default=8000, help="Port to run the server on (default: 8000)")
    a = p.parse_args()
    if a.test:
        asyncio.run(_run_smoke_tests())
    else:
        port = int(os.getenv("PORT", a.port))
        uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
