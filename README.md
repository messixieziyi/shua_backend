# 🎉 Meetup Service - Event Management System

A complete meetup service backend with approval-based chat functionality, built with FastAPI and SQLAlchemy.

## 📁 Project Structure

### Core Files
- `app.py` - Original FastAPI backend (basic version)
- `app_final.py` - Complete FastAPI backend with full database schema
- `app_production.py` - Production-ready version for Railway deployment
- `simple_app.py` - Simplified working backend for testing
- `split_interface_approval.html` - Main web interface with approval workflow

### Database
- `dev.db` - SQLite database (original app)
- `meetup_final.db` - SQLite database (complete schema)

### Environment
- `venv/` - Python virtual environment

## 🚀 Quick Start

### 1. Activate Environment
```bash
source venv/bin/activate
```

### 2. Start Server
```bash
# Simple version (recommended for testing)
python simple_app.py

# Complete version (full database)
python app_final.py

# Production version (Railway deployment)
python app_production.py
```

### 3. Open Interface
Open `split_interface_approval.html` in your browser

## 🎯 Features

### Event Management
- **Users**: 5 sample users with profiles
- **Events**: 5 sample events (Tennis, Basketball, Yoga, Golf, Running)
- **RSVPs**: Users can RSVP to events
- **Requests**: Join requests with approval workflow

### Approval Workflow
- **Participants**: Must request to join events
- **Organizers**: Approve/decline requests
- **Chat Access**: Only after approval
- **Real-time Updates**: UI updates immediately

### Interface
- **Split Design**: Participants (left) | Organizers (right)
- **Status Tracking**: Pending, Approved, Declined
- **Chat System**: Real-time messaging simulation
- **Quick Actions**: Pre-built message templates

## 🔧 API Endpoints

### Simple App (`simple_app.py`)
- `GET /users` - List users
- `GET /events` - List events
- `POST /requests` - Create join request
- `POST /rsvps` - Create RSVP

### Complete App (`app_final.py`)
- All simple app endpoints
- `POST /dev/seed` - Seed database
- Full database schema with relationships

### Production App (`app_production.py`)
- All endpoints from complete app
- PostgreSQL support for Railway
- Health check endpoint
- Production optimizations

## 🚀 Railway Deployment

This project is configured for deployment on Railway with:
- `requirements.txt` - Python dependencies
- `railway.toml` - Railway configuration
- `Procfile` - Process definition
- `app_production.py` - Production-ready app
- `.env.example` - Environment variables template

See `DEPLOYMENT.md` for detailed deployment instructions.

## 🎮 How to Use

1. **Start Server**: `python simple_app.py`
2. **Open Interface**: Open `split_interface_approval.html`
3. **Left Side (Participants)**:
   - Select a user
   - Select an event
   - Click "Request to Join Event"
   - Wait for approval
   - Chat and RSVP after approval
4. **Right Side (Organizers)**:
   - Select an organizer
   - Select an event
   - View pending requests
   - Approve/decline requests
   - Chat with approved participants

## 📊 Sample Data

- **5 Users**: Sarah, Mike, Emma, Alex, Lisa
- **5 Events**: Tennis, Basketball, Yoga, Golf, Running
- **Realistic Data**: Locations, times, capacities

## 🛠️ Technical Stack

- **Backend**: FastAPI, SQLAlchemy, SQLite/PostgreSQL
- **Frontend**: HTML, CSS, JavaScript
- **Database**: SQLite (local) / PostgreSQL (production)
- **API**: RESTful endpoints
- **Deployment**: Railway

## 🎯 Ready to Use!

The system is fully functional and ready for testing. The approval workflow ensures proper event management with controlled access to chat features.
