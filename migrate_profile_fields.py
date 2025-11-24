#!/usr/bin/env python3
"""
Database migration script to add extended profile fields to users table.
"""

import sqlite3
import os
from pathlib import Path

def migrate_database():
    """Add extended profile columns to users table if they don't exist."""
    db_file = Path(__file__).parent / "dev.db"
    
    if not db_file.exists():
        print("ℹ️  No existing database found. New database will be created with correct schema on startup.")
        return
    
    print(f"🔄 Migrating database: {db_file}")
    
    try:
        # Connect to database
        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()
        
        # Check existing columns
        cursor.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in cursor.fetchall()]
        
        new_columns = {
            "first_name": "VARCHAR(100)",
            "last_name": "VARCHAR(100)",
            "gender": "VARCHAR(20)",
            "birthday": "DATE",
            "sports": "JSON",
            "gallery_image_5": "TEXT",
            "gallery_image_6": "TEXT"
        }
        
        added_count = 0
        for col_name, col_type in new_columns.items():
            if col_name not in columns:
                print(f"➕ Adding {col_name} column to users table...")
                try:
                    cursor.execute(f"ALTER TABLE users ADD COLUMN {col_name} {col_type}")
                    added_count += 1
                except sqlite3.OperationalError as e:
                    print(f"⚠️  Error adding {col_name}: {e}")
        
        if added_count > 0:
            conn.commit()
            print(f"✅ Migration completed successfully! Added {added_count} columns.")
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
