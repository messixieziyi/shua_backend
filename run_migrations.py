#!/usr/bin/env python3
"""
Automated database migration script for Shua backend.
This script runs all necessary migrations to bring the database schema up to date.
It's designed to be idempotent - safe to run multiple times.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

def get_database_connection():
    """Get database connection based on DATABASE_URL environment variable."""
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./dev.db")
    
    if DATABASE_URL.startswith("postgresql://"):
        # PostgreSQL connection
        import psycopg2
        from urllib.parse import urlparse
        
        result = urlparse(DATABASE_URL)
        conn = psycopg2.connect(
            database=result.path[1:],
            user=result.username,
            password=result.password,
            host=result.hostname,
            port=result.port,
            sslmode='require'
        )
        return conn, 'postgresql'
    else:
        # SQLite connection
        import sqlite3
        db_file = DATABASE_URL.replace("sqlite:///", "")
        db_path = Path(__file__).parent / db_file
        conn = sqlite3.connect(str(db_path))
        return conn, 'sqlite'

def column_exists(cursor, table_name, column_name, db_type):
    """Check if a column exists in a table."""
    if db_type == 'sqlite':
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [column[1] for column in cursor.fetchall()]
        return column_name in columns
    else:  # postgresql
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name=%s AND column_name=%s
        """, (table_name, column_name))
        return cursor.fetchone() is not None

def run_migrations():
    """Run all database migrations."""
    print("🔄 Starting database migrations...")
    
    try:
        conn, db_type = get_database_connection()
        cursor = conn.cursor()
        print(f"✅ Connected to {db_type} database")
        
        migrations_run = []
        
        # Migration 1: Add created_at to threads table
        if not column_exists(cursor, 'threads', 'created_at', db_type):
            print("➕ Migration 1: Adding created_at column to threads table...")
            default_timestamp = datetime.now().isoformat()
            
            if db_type == 'sqlite':
                cursor.execute(f"""
                    ALTER TABLE threads 
                    ADD COLUMN created_at TIMESTAMP DEFAULT '{default_timestamp}'
                """)
                cursor.execute(f"""
                    UPDATE threads 
                    SET created_at = '{default_timestamp}' 
                    WHERE created_at IS NULL
                """)
            else:  # postgresql
                cursor.execute("""
                    ALTER TABLE threads 
                    ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                """)
            
            migrations_run.append("threads.created_at")
        
        # Migration 2: Add profile fields to users table
        profile_fields = [
            ('first_name', 'VARCHAR(100)'),
            ('last_name', 'VARCHAR(100)'),
            ('gender', 'VARCHAR(20)'),
            ('birthday', 'DATE'),
            ('sports', 'JSON' if db_type == 'postgresql' else 'TEXT'),
            ('gallery_image_5', 'TEXT'),
            ('gallery_image_6', 'TEXT'),
        ]
        
        for field_name, field_type in profile_fields:
            if not column_exists(cursor, 'users', field_name, db_type):
                print(f"➕ Migration 2: Adding {field_name} column to users table...")
                cursor.execute(f"""
                    ALTER TABLE users 
                    ADD COLUMN {field_name} {field_type}
                """)
                migrations_run.append(f"users.{field_name}")
        
        # Migration 3: Add sport_type to tags table
        if not column_exists(cursor, 'tags', 'sport_type', db_type):
            print("➕ Migration 3: Adding sport_type column to tags table...")
            cursor.execute("""
                ALTER TABLE tags 
                ADD COLUMN sport_type VARCHAR(50)
            """)
            migrations_run.append("tags.sport_type")
            
        # Migration 4: Create event_likes table
        if db_type == 'postgresql':
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'event_likes'
                )
            """)
            table_exists = cursor.fetchone()[0]
        else:  # sqlite
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='event_likes'
            """)
            table_exists = cursor.fetchone() is not None

        if not table_exists:
            print("➕ Migration 4: Creating event_likes table...")
            if db_type == 'sqlite':
                cursor.execute("""
                    CREATE TABLE event_likes (
                        event_id VARCHAR(36) NOT NULL,
                        user_id VARCHAR(36) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (event_id, user_id),
                        FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE,
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                    )
                """)
            else:  # postgresql
                cursor.execute("""
                    CREATE TABLE event_likes (
                        event_id VARCHAR(36) NOT NULL,
                        user_id VARCHAR(36) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (event_id, user_id),
                        FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE,
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                        CONSTRAINT uix_event_user_like UNIQUE (event_id, user_id)
                    )
                """)
            migrations_run.append("event_likes table")
        
        # Migration 5: Add level_requirement to events table
        if not column_exists(cursor, 'events', 'level_requirement', db_type):
            print("➕ Migration 5: Adding level_requirement column to events table...")
            cursor.execute("""
                ALTER TABLE events 
                ADD COLUMN level_requirement VARCHAR(20)
            """)
            migrations_run.append("events.level_requirement")
        
        # Commit all migrations
        conn.commit()
        
        if migrations_run:
            print(f"✅ Successfully ran {len(migrations_run)} migrations:")
            for migration in migrations_run:
                print(f"   - {migration}")
        else:
            print("✅ Database is already up to date - no migrations needed")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        if 'conn' in locals():
            conn.rollback()
            conn.close()
        return False

if __name__ == "__main__":
    success = run_migrations()
    sys.exit(0 if success else 1)
