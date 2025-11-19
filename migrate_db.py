#!/usr/bin/env python3
"""
Database migration script to add created_at column to threads table.
This handles the schema change without losing existing data.
"""

import sqlite3
import os
from datetime import datetime
from pathlib import Path

def migrate_database():
    """Add created_at column to threads table if it doesn't exist."""
    db_file = Path(__file__).parent / "dev.db"
    
    if not db_file.exists():
        print("ℹ️  No existing database found. New database will be created with correct schema.")
        return
    
    print(f"🔄 Migrating database: {db_file}")
    
    try:
        # Connect to database
        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()
        
        # Check if created_at column exists
        cursor.execute("PRAGMA table_info(threads)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'created_at' not in columns:
            print("➕ Adding created_at column to threads table...")
            
            # Add the column with a default value
            default_timestamp = datetime.now().isoformat()
            cursor.execute(f"""
                ALTER TABLE threads 
                ADD COLUMN created_at TIMESTAMP DEFAULT '{default_timestamp}'
            """)
            
            # Update existing rows with current timestamp
            cursor.execute(f"""
                UPDATE threads 
                SET created_at = '{default_timestamp}' 
                WHERE created_at IS NULL
            """)
            
            conn.commit()
            print("✅ Migration completed successfully!")
        else:
            print("✅ Database is already up to date!")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        if 'conn' in locals():
            conn.close()
        raise

if __name__ == "__main__":
    migrate_database()
