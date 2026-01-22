"""
Migration script to add notifications table to the database.
Run this script to add the notifications table to your database.
"""

import sys
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database URL
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./dev.db")

# Configure PostgreSQL SSL for Railway
if DATABASE_URL.startswith("postgresql://"):
    if "?" in DATABASE_URL:
        DATABASE_URL += "&sslmode=require"
    else:
        DATABASE_URL += "?sslmode=require"

print(f"🔗 Using database: {DATABASE_URL[:50]}...")

# Create engine
engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(bind=engine)

def migrate():
    """Add notifications table to the database."""
    session = SessionLocal()
    try:
        # Check if table already exists
        if DATABASE_URL.startswith("sqlite"):
            result = session.execute(text(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='notifications'"
            )).fetchone()
            table_exists = result is not None
        else:
            result = session.execute(text(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'notifications')"
            )).fetchone()
            table_exists = result[0] if result else False

        if table_exists:
            print("✅ Notifications table already exists. Skipping migration.")
            return

        print("📝 Creating notifications table...")

        if DATABASE_URL.startswith("sqlite"):
            # SQLite syntax
            session.execute(text("""
                CREATE TABLE notifications (
                    id VARCHAR(36) PRIMARY KEY,
                    user_id VARCHAR(36) NOT NULL,
                    type VARCHAR(50) NOT NULL,
                    title VARCHAR(200) NOT NULL,
                    body TEXT NOT NULL,
                    is_read BOOLEAN NOT NULL DEFAULT 0,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    event_id VARCHAR(36),
                    thread_id VARCHAR(36),
                    request_id VARCHAR(36),
                    related_user_id VARCHAR(36),
                    extra_data TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE,
                    FOREIGN KEY (thread_id) REFERENCES threads(id) ON DELETE CASCADE,
                    FOREIGN KEY (request_id) REFERENCES requests(id) ON DELETE CASCADE,
                    FOREIGN KEY (related_user_id) REFERENCES users(id) ON DELETE SET NULL
                )
            """))
            
            # Create indexes
            session.execute(text("CREATE INDEX idx_notifications_user_id ON notifications(user_id)"))
            session.execute(text("CREATE INDEX idx_notifications_is_read ON notifications(is_read)"))
            session.execute(text("CREATE INDEX idx_notifications_created_at ON notifications(created_at)"))
        else:
            # PostgreSQL syntax
            session.execute(text("""
                CREATE TYPE notificationtype AS ENUM (
                    'event_join_request',
                    'event_join_accepted',
                    'event_join_declined',
                    'event_joined',
                    'event_left',
                    'event_canceled',
                    'new_message'
                )
            """))
            
            session.execute(text("""
                CREATE TABLE notifications (
                    id VARCHAR(36) PRIMARY KEY,
                    user_id VARCHAR(36) NOT NULL,
                    type notificationtype NOT NULL,
                    title VARCHAR(200) NOT NULL,
                    body TEXT NOT NULL,
                    is_read BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    event_id VARCHAR(36),
                    thread_id VARCHAR(36),
                    request_id VARCHAR(36),
                    related_user_id VARCHAR(36),
                    extra_data JSONB,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE,
                    FOREIGN KEY (thread_id) REFERENCES threads(id) ON DELETE CASCADE,
                    FOREIGN KEY (request_id) REFERENCES requests(id) ON DELETE CASCADE,
                    FOREIGN KEY (related_user_id) REFERENCES users(id) ON DELETE SET NULL
                )
            """))
            
            # Create indexes
            session.execute(text("CREATE INDEX idx_notifications_user_id ON notifications(user_id)"))
            session.execute(text("CREATE INDEX idx_notifications_is_read ON notifications(is_read)"))
            session.execute(text("CREATE INDEX idx_notifications_created_at ON notifications(created_at)"))

        session.commit()
        print("✅ Notifications table created successfully!")
        
    except Exception as e:
        session.rollback()
        print(f"❌ Error creating notifications table: {e}")
        sys.exit(1)
    finally:
        session.close()

if __name__ == "__main__":
    migrate()
