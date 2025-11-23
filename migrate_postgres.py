import os
import sys
from sqlalchemy import create_engine, text

def migrate_postgres():
    """
    Migrate PostgreSQL database to add missing columns to events table.
    """
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("❌ DATABASE_URL environment variable not set.")
        sys.exit(1)

    # Fix for sqlalchemy needing postgresql:// instead of postgres://
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)

    print(f"🔄 Connecting to database...")
    
    try:
        engine = create_engine(database_url)
        with engine.connect() as conn:
            # Check if 'status' column exists in 'events' table
            # PostgreSQL query to check columns
            check_sql = text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name='events' AND column_name='status';
            """)
            result = conn.execute(check_sql).fetchone()
            
            if not result:
                print("➕ Adding 'status' column to events table...")
                conn.execute(text("ALTER TABLE events ADD COLUMN status VARCHAR(20) DEFAULT 'ACTIVE'"))
                conn.commit()
                print("✅ Added 'status' column.")
            else:
                print("✅ 'status' column already exists.")

            # Check if 'cancellation_deadline_hours' column exists
            check_sql = text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name='events' AND column_name='cancellation_deadline_hours';
            """)
            result = conn.execute(check_sql).fetchone()
            
            if not result:
                print("➕ Adding 'cancellation_deadline_hours' column to events table...")
                conn.execute(text("ALTER TABLE events ADD COLUMN cancellation_deadline_hours INTEGER DEFAULT 24"))
                conn.commit()
                print("✅ Added 'cancellation_deadline_hours' column.")
            else:
                print("✅ 'cancellation_deadline_hours' column already exists.")

            print("🎉 Migration completed successfully!")

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    migrate_postgres()
