# 🎉 Meetup Service - Complete Event Management Platform

A comprehensive meetup service backend with real-time chat, approval workflows, and advanced tagging system. Built with FastAPI, SQLAlchemy, and modern web technologies.

## 🚀 Live Deployment

**Production URL:** [Deployed on Railway](https://sweet-creativity-production.up.railway.app/)

## 📁 Project Structure

### Core Backend Files
- `main.py` - **Primary FastAPI application** (use this for production)
- `app.py` - Alternative FastAPI implementation
- `index.html` - **Complete web interface** with 5-panel design

### Database
- `dev.db` - SQLite development database
- `test.db` - SQLite test database
- PostgreSQL - Production database (Railway)

### Configuration
- `requirements.txt` - Python dependencies
- `railway.toml` - Railway deployment configuration
- `Procfile` - Process definition for Railway

## 🎯 Current Features

### 🏷️ **Advanced Tagging System**
- **Tag Management**: Create, edit, delete tags with custom colors
- **Event Tagging**: Assign multiple tags to events
- **Tag Filtering**: Filter events by tags in real-time
- **Visual Tags**: Color-coded badges with automatic contrast
- **Sample Tags**: 15 pre-built categories (Beginner, Advanced, Outdoor, etc.)

### 👥 **User Management**
- **User Creation**: Add users with display names and emails
- **User Profiles**: Complete user information system
- **Random Generation**: Quick user creation with realistic data

### 📅 **Event Management**
- **Event Creation**: Full event creation with all details
- **Event Details**: Title, description, date/time, capacity, location, address
- **Activity Types**: Predefined categories (Tennis, Basketball, Yoga, etc.)
- **Event Overview**: Comprehensive event listing with participants

### 💬 **Real-time Chat System**
- **WebSocket Support**: Real-time messaging
- **Thread-based Chat**: Separate chat threads per event
- **Access Control**: Only approved participants can chat
- **Message Types**: User messages and system notifications
- **Read Status**: Track message read status

### ✅ **Approval Workflow**
- **Join Requests**: Users must request to join events
- **Host Approval**: Event creators approve/decline requests
- **Status Tracking**: SUBMITTED, ACCEPTED, DECLINED, EXPIRED, CANCELED
- **Auto-accept**: Optional automatic approval for events
- **Thread Management**: Chat threads upgrade from REQUEST to BOOKING scope

### 🎨 **Modern Web Interface**
- **5-Panel Design**: Organized interface for different functions
- **Responsive Layout**: Works on desktop and mobile
- **Real-time Updates**: Live data synchronization
- **Interactive Elements**: Drag-and-drop, color pickers, filters

## 🔧 API Endpoints

### Core Endpoints
```
GET    /                    - Main web interface
GET    /health             - Health check
GET    /users              - List all users
POST   /users              - Create new user
GET    /events             - List events (with tag filtering)
POST   /events             - Create new event
```

### Request Management
```
GET    /requests           - Get user's requests
GET    /requests/all       - Get all requests (admin)
POST   /requests           - Create join request
POST   /requests/{id}/act  - Approve/decline request
```

### Chat System
```
GET    /threads            - Get user's chat threads
GET    /threads/{id}/messages - Get thread messages
POST   /threads/{id}/messages - Send message
POST   /threads/{id}/read  - Mark messages as read
GET    /threads/{id}/participants - Get thread participants
```

### Tag Management
```
GET    /tags               - List all tags
POST   /tags               - Create new tag
DELETE /tags/{id}          - Delete tag
POST   /events/{id}/tags   - Add tags to event
DELETE /events/{id}/tags/{tag_id} - Remove tag from event
```

### Development Tools
```
POST   /dev/seed           - Seed database with test data
POST   /dev/seed-tags      - Create sample tags
POST   /dev/create-tables  - Manually create database tables
GET    /dev/check-db       - Check database connection
```

### WebSocket
```
WS     /ws/{user_id}       - Real-time chat connection
```

## 🗄️ Database Schema

### Core Tables
- **users** - User profiles and information
- **events** - Event details and metadata
- **requests** - Join requests with approval status
- **bookings** - Confirmed event participations
- **threads** - Chat thread management
- **messages** - Chat messages and system notifications
- **thread_participants** - Chat thread membership
- **message_reads** - Message read status tracking

### Tagging Tables
- **tags** - Tag definitions with colors and descriptions
- **event_tags** - Many-to-many relationship between events and tags

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone <repository-url>
cd shua_backend
pip install -r requirements.txt
```

### 2. Start Development Server
```bash
python main.py
```

### 3. Access Interface
Open `http://localhost:8000` in your browser

### 4. Initialize Sample Data
1. Go to the "User Creation" panel (rightmost)
2. Click "Seed Backend" to create sample tags
3. Create some users and events
4. Start using the system!

## 🎮 How to Use the Interface

### Panel 1: Event Participants
- Select a user and event
- Request to join events
- Chat with approved participants
- Real-time messaging

### Panel 2: Event Creators
- Create new events with full details
- Add tags to events
- Generate random event data
- Manage event information

### Panel 3: Requests Overview
- View all pending requests
- Approve/decline requests
- See request status and details
- Organized by event and host

### Panel 4: Events Overview
- Browse all events
- Filter events by tags
- See event participants
- View event details

### Panel 5: User Creation
- Create new users
- Manage existing users
- Create and manage tags
- Seed sample data

## 🏷️ Tag System Usage

### Creating Tags
1. Go to "User Creation" panel
2. In "Tag Management" section:
   - Enter tag name and select color
   - Click "Create" for custom tags
   - Click "Add Samples" for predefined tags
   - Click "Seed Backend" for server-side creation

### Using Tags
1. When creating events, click "Show Tags"
2. Select multiple tags for your event
3. Use tag filter in Events Overview
4. Tags appear as colored badges on events

## 🔌 Frontend Integration

### For Other AI/Developers

This backend provides a complete REST API and WebSocket interface that can be easily integrated with any frontend framework:

#### API Base URL
```
Production: https://sweet-creativity-production.up.railway.app/
Development: http://localhost:8000
```

#### Key Integration Points
1. **Authentication**: Use `X-User-Id` header for user identification
2. **Real-time Chat**: Connect to WebSocket endpoint `/ws/{user_id}`
3. **Event Data**: All endpoints return JSON with consistent structure
4. **Tag System**: Full CRUD operations for tags and event-tag relationships

#### Sample API Calls
```javascript
// Get events with tags
fetch('/events?tag_filter=Beginner')

// Create event with tags
fetch('/events', {
  method: 'POST',
  headers: { 'X-User-Id': 'user123', 'Content-Type': 'application/json' },
  body: JSON.stringify({
    title: 'Morning Tennis',
    tag_ids: ['tag1', 'tag2']
  })
})

// WebSocket connection
const ws = new WebSocket('/ws/user123')
```

## 🛠️ Technical Stack

- **Backend**: FastAPI, SQLAlchemy 2.0, Pydantic
- **Database**: SQLite (dev) / PostgreSQL (production)
- **Real-time**: WebSockets, asyncio
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Deployment**: Railway with automatic deployments
- **API**: RESTful with WebSocket support

## 📊 Current Data

The system includes comprehensive sample data:
- **15 Sample Tags**: Beginner, Advanced, Outdoor, Indoor, etc.
- **User Management**: Full CRUD operations
- **Event System**: Complete event lifecycle
- **Chat System**: Real-time messaging with access control
- **Approval Workflow**: Full request/approval system

## 🚀 Deployment Status

- ✅ **Backend**: Deployed and running on Railway
- ✅ **Database**: PostgreSQL with full schema
- ✅ **WebSocket**: Real-time chat functionality
- ✅ **Tag System**: Complete tagging infrastructure
- ✅ **API**: All endpoints functional and documented

## 🎯 Ready for Integration!

This backend is production-ready and provides all the necessary APIs for a complete meetup service. The tagging system, real-time chat, and approval workflows make it suitable for any event management application.

**For frontend developers**: All API endpoints are documented, CORS is enabled, and the WebSocket interface is ready for real-time features.