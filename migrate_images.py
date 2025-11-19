#!/usr/bin/env python3
"""
Database migration script to add image fields to users and events tables.
Run this script to update your existing database with the new image columns.
"""

import sqlite3
import os
import sys

def migrate_database():
    """Add image columns to existing database tables."""
    
    # Get database path
    db_path = os.getenv("DATABASE_PATH", "./dev.db")
    
    if not os.path.exists(db_path):
        print(f"Database file not found: {db_path}")
        print("Creating new database with image columns...")
        # If database doesn't exist, the main app will create it with all columns
        return True
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print(f"Migrating database: {db_path}")
        
        # Check if profile_picture column exists in users table
        cursor.execute("PRAGMA table_info(users)")
        users_columns = [column[1] for column in cursor.fetchall()]
        
        if 'profile_picture' not in users_columns:
            print("Adding profile_picture column to users table...")
            cursor.execute("ALTER TABLE users ADD COLUMN profile_picture TEXT")
            print("✅ Added profile_picture column to users table")
        else:
            print("✅ profile_picture column already exists in users table")
        
        # Check if image columns exist in events table
        cursor.execute("PRAGMA table_info(events)")
        events_columns = [column[1] for column in cursor.fetchall()]
        
        image_columns = ['image_1', 'image_2', 'image_3']
        for col in image_columns:
            if col not in events_columns:
                print(f"Adding {col} column to events table...")
                cursor.execute(f"ALTER TABLE events ADD COLUMN {col} TEXT")
                print(f"✅ Added {col} column to events table")
            else:
                print(f"✅ {col} column already exists in events table")
        
        conn.commit()
        conn.close()
        
        print("\n🎉 Database migration completed successfully!")
        return True
        
    except sqlite3.Error as e:
        print(f"❌ Database migration failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during migration: {e}")
        return False

if __name__ == "__main__":
    print("🔄 Starting database migration for image upload feature...")
    print("=" * 50)
    
    success = migrate_database()
    
    if success:
        print("\n✅ Migration completed! You can now use image upload features.")
        print("💡 Tip: Restart your backend server to ensure all changes are loaded.")
        sys.exit(0)
    else:
        print("\n❌ Migration failed! Please check the error messages above.")
        sys.exit(1)
