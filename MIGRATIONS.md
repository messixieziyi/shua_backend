# Database Migration Guide

## Overview

This project uses an automated database migration system that runs on every deployment. The migrations are designed to be **idempotent** - meaning they can be run multiple times safely without causing issues.

## How It Works

### 1. Migration Script (`run_migrations.py`)

The main migration script automatically:
- Detects whether you're using SQLite (local) or PostgreSQL (production)
- Checks which migrations have already been applied
- Runs only the necessary migrations
- Supports both SQLite and PostgreSQL databases

### 2. Automatic Execution on Deployment

#### Railway/Heroku (Procfile)
The `Procfile` has been configured to run migrations before starting the server:

```
release: python3 run_migrations.py
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

The `release` phase runs migrations, and only if successful, the `web` phase starts the server.

#### Local Development
For local development, you can run migrations manually:

```bash
python3 run_migrations.py
```

Or use the startup script:

```bash
./startup.sh
```

## Adding New Migrations

When you need to add a new database schema change:

### 1. Edit `run_migrations.py`

Add your migration to the `run_migrations()` function. Example:

```python
# Migration 3: Add new column
if not column_exists(cursor, 'events', 'new_field', db_type):
    print("➕ Migration 3: Adding new_field column to events table...")
    cursor.execute("""
        ALTER TABLE events 
        ADD COLUMN new_field VARCHAR(100)
    """)
    migrations_run.append("events.new_field")
```

### 2. Test Locally

```bash
# Run migrations
python3 run_migrations.py

# Start the server
python3 main.py
```

### 3. Commit and Deploy

```bash
git add run_migrations.py
git commit -m "Add migration for new_field"
git push
```

The migration will run automatically on deployment.

## Current Migrations

### Migration 1: Threads Table
- **Column**: `created_at`
- **Type**: TIMESTAMP
- **Purpose**: Track when conversation threads are created

### Migration 2: Users Table Profile Fields
- **Columns**: 
  - `first_name` (VARCHAR)
  - `last_name` (VARCHAR)
  - `gender` (VARCHAR)
  - `birthday` (DATE)
  - `sports` (JSON/TEXT)
  - `gallery_image_5` (TEXT)
  - `gallery_image_6` (TEXT)
- **Purpose**: Extended user profile information

### Migration 3: Tags Table Sport Type
- **Column**: `sport_type`
- **Type**: VARCHAR(50)
- **Purpose**: Associate tags with specific sports for filtered tag selection
- **Nullable**: Yes (supports general tags that apply to all sports)

## Database Support

The migration system supports:
- **SQLite** (local development)
- **PostgreSQL** (production on Railway/Heroku)

The script automatically detects which database you're using based on the `DATABASE_URL` environment variable.

## Troubleshooting

### Migration Failed on Deployment

1. Check the deployment logs for the error message
2. Fix the migration script
3. Redeploy

### Need to Rollback a Migration

Since migrations are additive (only adding columns), you typically don't need to rollback. If you must:

1. Manually connect to the database
2. Run `ALTER TABLE table_name DROP COLUMN column_name`
3. Update the migration script to remove that migration

### Testing Migrations Locally

```bash
# Backup your database first
cp dev.db dev.db.backup

# Run migrations
python3 run_migrations.py

# If something goes wrong, restore
mv dev.db.backup dev.db
```

## Best Practices

1. **Always test migrations locally first**
2. **Make migrations idempotent** - check if changes already exist
3. **Never delete or modify existing migrations** - only add new ones
4. **Backup production database before major migrations**
5. **Keep migrations small and focused** - one change per migration
6. **Add descriptive print statements** for debugging

## Environment Variables

- `DATABASE_URL`: Database connection string (auto-detected)
  - SQLite: `sqlite:///./dev.db` (default)
  - PostgreSQL: `postgresql://user:pass@host:port/db?sslmode=require`

## Files

- `run_migrations.py` - Main migration script
- `startup.sh` - Local development startup script
- `Procfile` - Railway/Heroku deployment configuration
- `migrate_db.py` - Legacy migration (deprecated)
- `migrate_profile_fields.py` - Legacy migration (deprecated)
