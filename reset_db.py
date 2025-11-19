#!/usr/bin/env python3
"""
Simple script to reset the database by deleting the SQLite file and recreating tables.
Use this when the database schema has changed and you need a fresh start.
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path to import main
sys.path.insert(0, str(Path(__file__).parent))

def reset_database():
    """Reset the database by removing the SQLite file."""
    db_file = Path(__file__).parent / "dev.db"
    
    if db_file.exists():
        print(f"🗑️  Removing existing database: {db_file}")
        db_file.unlink()
        print("✅ Database file removed")
    else:
        print("ℹ️  No existing database file found")
    
    print("🔄 Database will be recreated when the server starts")
    print("💡 You can now start the server with: python main.py")

if __name__ == "__main__":
    reset_database()
