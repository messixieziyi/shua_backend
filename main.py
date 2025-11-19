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
    BigInteger, Boolean, DateTime, Enum as SQLEnum, ForeignKey, String, Text, UniqueConstraint, func, select, create_engine, text, delete,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker
from sqlalchemy.exc import IntegrityError
from starlette.concurrency import run_in_threadpool

# Authentication imports
from passlib.context import CryptContext
from argon2 import PasswordHasher
from jose import JWTError, jwt
from datetime import timedelta

# ---------------------
# Database
# ---------------------
import os

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
    email: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    profile_picture: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Base64 encoded image
    bio: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    gallery_image_1: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Base64 encoded image
    gallery_image_2: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Base64 encoded image
    gallery_image_3: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Base64 encoded image
    gallery_image_4: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Base64 encoded image
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
    name: Mapped[str] = mapped_column(String(50), unique=True)
    color: Mapped[Optional[str]] = mapped_column(String(7), nullable=True)  # Hex color code
    description: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=func.now())

class EventTag(Base):
    __tablename__ = "event_tags"
    event_id: Mapped[str] = mapped_column(ForeignKey("events.id", ondelete="CASCADE"), primary_key=True)
    tag_id: Mapped[str] = mapped_column(ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=func.now())

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

# ---------------------
# Authentication Schemas
# ---------------------
class UserRegister(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)
    display_name: str = Field(min_length=1, max_length=120)

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
    except JWTError:
        return None

# ---------------------
# Image Upload Utilities
# ---------------------
import base64
import re

def validate_base64_image(image_data: str, max_size_mb: int = 5) -> tuple[bool, str]:
    """Validate base64 encoded image data or image URL."""
    try:
        # Allow HTTP/HTTPS URLs (for existing images from Unsplash, etc.)
        if image_data.startswith('http://') or image_data.startswith('https://'):
            return True, "Valid image URL"
        
        # Check if it's a valid base64 data URL
        if not image_data.startswith('data:image/'):
            return False, "Invalid image format. Must be a data URL or valid image URL."
        
        # Extract the base64 part
        header, data = image_data.split(',', 1)
        
        # Validate MIME type
        mime_match = re.match(r'data:image/(jpeg|jpg|png|gif|webp)', header)
        if not mime_match:
            return False, "Unsupported image type. Only JPEG, PNG, GIF, and WebP are allowed."
        
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
            hashed_password=hashed_password
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
        "created_at": current_user.created_at.isoformat() if current_user.created_at else None
    }

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
            # Validate the image
            is_valid, message = validate_base64_image(profile_data.profile_picture, max_size_mb=2)
            if not is_valid:
                raise HTTPException(status_code=400, detail=message)
        
        # Update user's profile picture
        current_user.profile_picture = profile_data.profile_picture
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
        
        # Validate and update gallery images if provided
        for field_name, image_data in [
            ("gallery_image_1", profile_data.gallery_image_1),
            ("gallery_image_2", profile_data.gallery_image_2),
            ("gallery_image_3", profile_data.gallery_image_3),
            ("gallery_image_4", profile_data.gallery_image_4)
        ]:
            if image_data is not None:
                if image_data:  # Not empty string
                    is_valid, message = validate_base64_image(image_data, max_size_mb=5)
                    if not is_valid:
                        raise HTTPException(status_code=400, detail=f"{field_name}: {message}")
                setattr(current_user, field_name, image_data)
        
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
                "gallery_image_4": current_user.gallery_image_4
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
        
        # Validate images if provided
        for field_name, image_data in [
            ("image_1", images_data.image_1),
            ("image_2", images_data.image_2), 
            ("image_3", images_data.image_3)
        ]:
            if image_data:
                is_valid, message = validate_base64_image(image_data, max_size_mb=3)
                if not is_valid:
                    raise HTTPException(status_code=400, detail=f"{field_name}: {message}")
        
        # Update event images
        if images_data.image_1 is not None:
            event.image_1 = images_data.image_1
        if images_data.image_2 is not None:
            event.image_2 = images_data.image_2
        if images_data.image_3 is not None:
            event.image_3 = images_data.image_3
            
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
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """Get all events, optionally filtered by tag or activity type. Public endpoint - no authentication required."""
    try:
        query = select(Event)
        
        if tag_filter:
            # Filter events by tag name
            query = query.join(EventTag).join(Tag).where(Tag.name.ilike(f"%{tag_filter}%"))
        
        if activity_filter:
            # Filter events by activity type (sport name)
            query = query.where(Event.activity_type.ilike(f"%{activity_filter}%"))
        
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
            created_by=created_by
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
                if image_data:  # Not empty string
                    is_valid, message = validate_base64_image(image_data, max_size_mb=5)
                    if not is_valid:
                        raise HTTPException(status_code=400, detail=f"{field_name}: {message}")
                setattr(event, field_name, image_data)
        
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
            raise HTTPException(400, "You already have a request for this event")
        
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
            
            req.status = RequestStatus.ACCEPTED
            booking = Booking(request_id=req.id, status=BookingStatus.CONFIRMED)
            session.add(booking)
            session.flush()
            
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
                
                # Get guest name for system message
                guest = session.execute(select(User).where(User.id == req.guest_id)).scalar_one_or_none()
                guest_name = guest.display_name if guest else "A participant"
                await system_message(session, thread.id, f"{guest_name} has joined the event!")
            
            session.flush()
            session.commit()
            return {"status": req.status, "booking_id": booking.id, "thread_id": thread.id}

        elif payload.action == "decline":
            # Check if request is already processed
            if req.status != RequestStatus.SUBMITTED:
                raise HTTPException(400, f"Request is already {req.status.lower()}")
            
            req.status = RequestStatus.DECLINED
            thread.is_locked = True
            session.flush()
            await system_message(session, thread.id, "Request declined. Thread locked.")
            session.commit()
            return {"status": req.status, "thread_id": thread.id}

        else:
            raise HTTPException(400, "Invalid action. Must be 'accept' or 'decline'")
    
    except HTTPException:
        session.rollback()
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(500, f"Failed to process request action: {str(e)}")

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
    """Endpoint to seed the database with sample tags."""
    sample_tags = [
        {"name": "Beginner", "color": "#10b981", "description": "Suitable for beginners"},
        {"name": "Advanced", "color": "#ef4444", "description": "For experienced participants"},
        {"name": "Outdoor", "color": "#059669", "description": "Outdoor activities"},
        {"name": "Indoor", "color": "#3b82f6", "description": "Indoor activities"},
        {"name": "Morning", "color": "#f59e0b", "description": "Morning sessions"},
        {"name": "Evening", "color": "#8b5cf6", "description": "Evening sessions"},
        {"name": "Weekend", "color": "#ec4899", "description": "Weekend events"},
        {"name": "Free", "color": "#10b981", "description": "Free events"},
        {"name": "Paid", "color": "#f59e0b", "description": "Paid events"},
        {"name": "Group", "color": "#6366f1", "description": "Group activities"},
        {"name": "Solo", "color": "#8b5cf6", "description": "Individual activities"},
        {"name": "Fitness", "color": "#ef4444", "description": "Fitness related"},
        {"name": "Social", "color": "#ec4899", "description": "Social events"},
        {"name": "Competitive", "color": "#dc2626", "description": "Competitive events"},
        {"name": "Casual", "color": "#6b7280", "description": "Casual activities"}
    ]
    
    created_tags = []
    for tag_data in sample_tags:
        # Check if tag already exists
        existing_tag = session.execute(
            select(Tag).where(Tag.name == tag_data["name"])
        ).scalar_one_or_none()
        
        if not existing_tag:
            new_tag = Tag(
                name=tag_data["name"],
                color=tag_data["color"],
                description=tag_data["description"]
            )
            session.add(new_tag)
            created_tags.append(tag_data["name"])
    
    session.commit()
    
    return {
        "message": f"Created {len(created_tags)} sample tags",
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
    current_user: User = Depends(get_current_user),
    session=Depends(get_session)
):
    """Get all available tags. Requires authentication."""
    tags = session.execute(select(Tag).order_by(Tag.name)).scalars().all()
    return [
        {
            "id": tag.id,
            "name": tag.name,
            "color": tag.color,
            "description": tag.description,
            "created_at": tag.created_at.isoformat()
        }
        for tag in tags
    ]

@app.post("/tags")
async def create_tag(
    tag_data: TagCreate, 
    current_user: User = Depends(get_current_user),
    session=Depends(get_session)
):
    """Create a new tag. Requires authentication."""
    try:
        # Check if tag with same name already exists
        existing_tag = session.execute(
            select(Tag).where(Tag.name == tag_data.name)
        ).scalar_one_or_none()
        
        if existing_tag:
            raise HTTPException(status_code=400, detail="Tag with this name already exists")
        
        new_tag = Tag(
            name=tag_data.name,
            color=tag_data.color,
            description=tag_data.description
        )
        session.add(new_tag)
        session.commit()
        session.refresh(new_tag)
        
        return {
            "id": new_tag.id,
            "name": new_tag.name,
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
    a = p.parse_args()
    if a.test:
        asyncio.run(_run_smoke_tests())
    else:
        uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=False)
