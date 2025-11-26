#!/bin/bash
# Startup script for Shua backend
# This script runs migrations before starting the server

set -e  # Exit on error

echo "🚀 Starting Shua Backend..."

# Run database migrations
echo "📦 Running database migrations..."
python3 run_migrations.py

if [ $? -eq 0 ]; then
    echo "✅ Migrations completed successfully"
else
    echo "❌ Migrations failed"
    exit 1
fi

# Start the server
echo "🌐 Starting FastAPI server..."
if [ "$ENVIRONMENT" = "production" ]; then
    # Production: Use gunicorn with uvicorn workers
    exec gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT:-8000}
else
    # Development: Use uvicorn with reload
    exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --reload
fi
