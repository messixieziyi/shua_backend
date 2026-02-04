#!/usr/bin/env python3
"""
Database migration script to add level_requirement field to events table.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import main
sys.path.insert(0, str(Path(__file__).parent))

def migrate_database():
    """Add level_requirement column to events table if it doesn't exist."""
    from main import engine, Base, Event
    from sqlalchemy import inspect, text
    
    print("🔄 Migrating database: Adding level_requirement field to events table...")
    
    try:
        # Check if column exists
        inspector = inspect(engine)
        columns = [col['name'] for col in inspector.get_columns('events')]
        
        if 'level_requirement' in columns:
            print("✅ Database is already up to date! level_requirement column exists.")
            return
        
        # Get database URL to determine database type
        database_url = os.getenv('DATABASE_URL', 'sqlite:///./dev.db')
        
        print(f"➕ Adding level_requirement column to events table...")
        
        with engine.connect() as conn:
            if 'sqlite' in database_url.lower():
                # SQLite migration
                conn.execute(text("ALTER TABLE events ADD COLUMN level_requirement VARCHAR(20)"))
            else:
                # PostgreSQL migration
                conn.execute(text("ALTER TABLE events ADD COLUMN level_requirement VARCHAR(20)"))
            
            conn.commit()
        
        print("✅ Migration completed successfully! Added level_requirement column.")
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        # If column already exists, that's okay
        if 'already exists' in str(e).lower() or 'duplicate' in str(e).lower():
            print("✅ Column already exists, migration not needed.")
        else:
            raise

if __name__ == "__main__":
    migrate_database()
