# Shua Chat API Documentation

## Overview

The Shua backend provides a comprehensive chat system that creates **one group chat per event** where all accepted participants can communicate together. This follows a group coordination model where:

1. **One group chat per event**: When the first participant's request gets accepted by the host, a group chat is created for that event.
2. **All accepted participants join the same group**: Additional accepted participants are added to the existing group chat.
3. **Event details in chat**: Chat interface can display event details on the right side.
4. **Real-time messaging**: WebSocket support for live message updates.
5. **Read receipts**: Track unread message counts.

## Authentication

All chat endpoints require authentication using JWT Bearer tokens:

```
Authorization: Bearer <your_jwt_token>
```

Get tokens via `/auth/login` or `/auth/register` endpoints.

## Chat Flow

### 1. Request Creation
When a user requests to join an event:

```http
POST /requests
Content-Type: application/json
Authorization: Bearer <token>

{
  "event_id": "event-uuid",
  "host_id": "host-user-uuid",
  "auto_accept": false
}
```

This creates:
- A `Request` record
- No thread is created yet (thread created when first request is accepted)

### 2. First Request Acceptance
When the host accepts the **first** request for an event:

```http
POST /requests/{request_id}/act
Content-Type: application/json
Authorization: Bearer <token>

{
  "action": "accept"
}
```

This:
- Changes request status to "ACCEPTED"
- Creates a `Booking` record
- **Creates the event's group chat thread**
- Adds host and accepted participant to the group
- Sends welcome system message

### 3. Additional Request Acceptance
When the host accepts **additional** requests for the same event:

This:
- Changes request status to "ACCEPTED"
- Creates a `Booking` record
- **Adds the new participant to the existing group chat**
- Sends "X has joined the event!" system message

### 4. Group Chat Access
All accepted participants can communicate in the same group chat thread.

## Chat API Endpoints

### Get User's Threads

Get all chat threads for the authenticated user with event details and last messages.

```http
GET /threads
Authorization: Bearer <token>
```

**Response:**
```json
{
  "threads": [
    {
      "id": "thread-uuid",
      "scope": "booking",
      "request_id": "request-uuid",
      "booking_id": "booking-uuid",
      "event_id": "event-uuid",
      "is_locked": false,
      "unread_count": 3,
      "event": {
        "id": "event-uuid",
        "title": "Morning Tennis Match",
        "description": "Friendly tennis game at Central Park",
        "starts_at": "2024-01-15T09:00:00Z",
        "location": "Central Park Tennis Courts",
        "activity_type": "tennis"
      },
      "last_message": {
        "id": "message-uuid",
        "sender_id": "user-uuid",
        "sender_name": "John Doe",
        "body": "See you tomorrow!",
        "created_at": "2024-01-14T18:30:00Z",
        "kind": "user"
      }
    }
  ],
  "participants": {
    "thread-uuid": [
      {
        "thread_id": "thread-uuid",
        "user_id": "user-uuid",
        "user_name": "John Doe",
        "role": "guest"
      },
      {
        "thread_id": "thread-uuid",
        "user_id": "host-uuid",
        "user_name": "Jane Smith",
        "role": "host"
      }
    ]
  }
}
```

### Get Thread Details

Get comprehensive details about a specific thread including full event information.

```http
GET /threads/{thread_id}/details
Authorization: Bearer <token>
```

**Response:**
```json
{
  "thread": {
    "id": "thread-uuid",
    "scope": "booking",
    "is_locked": false,
    "event_id": "event-uuid"
  },
  "event": {
    "id": "event-uuid",
    "title": "Morning Tennis Match",
    "description": "Friendly tennis game at Central Park",
    "starts_at": "2024-01-15T09:00:00Z",
    "location": "Central Park Tennis Courts",
    "address": "Central Park, New York, NY",
    "activity_type": "tennis",
    "capacity": 4,
    "created_by": "host-uuid",
    "creator_name": "Jane Smith",
    "tags": [
      {
        "id": "tag-uuid",
        "name": "Beginner",
        "color": "#10b981",
        "description": "Suitable for beginners"
      }
    ]
  },
  "request": {
    "id": "request-uuid",
    "status": "ACCEPTED",
    "guest_id": "user-uuid",
    "guest_name": "John Doe",
    "host_id": "host-uuid",
    "host_name": "Jane Smith",
    "created_at": "2024-01-14T10:00:00Z"
  },
  "booking": {
    "id": "booking-uuid",
    "status": "CONFIRMED"
  }
}
```

### Get Thread Messages

Retrieve messages from a specific thread.

```http
GET /threads/{thread_id}/messages?limit=50&offset=0
Authorization: Bearer <token>
```

**Parameters:**
- `limit` (optional): Number of messages to retrieve (default: 50)
- `offset` (optional): Number of messages to skip (default: 0)

**Response:**
```json
[
  {
    "id": "message-uuid",
    "thread_id": "thread-uuid",
    "sender_id": "user-uuid",
    "kind": "user",
    "body": "Hi! Looking forward to the tennis match!",
    "created_at": "2024-01-14T10:05:00Z",
    "seq": 1
  },
  {
    "id": "system-message-uuid",
    "thread_id": "thread-uuid",
    "sender_id": null,
    "kind": "system",
    "body": "Request accepted, booking confirmed.",
    "created_at": "2024-01-14T10:01:00Z",
    "seq": 2
  }
]
```

### Send Message

Send a message to a thread.

```http
POST /threads/{thread_id}/messages
Content-Type: application/json
Authorization: Bearer <token>

{
  "client_msg_id": "unique-client-id",
  "body": "Great! What should I bring?"
}
```

**Response:**
```json
{
  "id": "message-uuid",
  "thread_id": "thread-uuid",
  "sender_id": "user-uuid",
  "kind": "user",
  "body": "Great! What should I bring?",
  "created_at": "2024-01-14T18:35:00Z",
  "seq": 3
}
```

### Mark Messages as Read

Update read status for messages in a thread.

```http
POST /threads/{thread_id}/read
Content-Type: application/json
Authorization: Bearer <token>

{
  "last_read_seq": 3
}
```

**Response:**
```json
{
  "status": "success"
}
```

### Get Thread Participants

Get all participants in a thread with their details.

```http
GET /threads/{thread_id}/participants
Authorization: Bearer <token>
```

**Response:**
```json
[
  {
    "thread_id": "thread-uuid",
    "user_id": "user-uuid",
    "user_name": "John Doe",
    "role": "guest"
  },
  {
    "thread_id": "thread-uuid",
    "user_id": "host-uuid",
    "user_name": "Jane Smith",
    "role": "host"
  }
]
```

## WebSocket Real-time Updates

Connect to WebSocket for real-time message updates:

```javascript
const ws = new WebSocket(`ws://localhost:8002/ws/${userId}`);

// Join a thread to receive messages
ws.send(JSON.stringify({
  type: "join_thread",
  thread_id: "thread-uuid"
}));

// Listen for new messages
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === "new_message") {
    console.log("New message:", data.message);
  }
};

// Leave a thread
ws.send(JSON.stringify({
  type: "leave_thread",
  thread_id: "thread-uuid"
}));
```

## Message Types

### User Messages
- `kind`: "user"
- `sender_id`: User UUID
- `body`: Message content

### System Messages
- `kind`: "system"
- `sender_id`: null
- `body`: System-generated message (e.g., "Request accepted, booking confirmed.")

## Thread Scopes

### Request Scope
- Initial state when request is created
- Both host and guest can see the thread
- Only host can send messages until request is accepted

### Booking Scope
- Activated when request is accepted
- Both participants can send messages
- Thread remains active for coordination

## Error Handling

### Common Error Responses

**403 Forbidden - Access Denied:**
```json
{
  "detail": "Access denied - you are not a participant in this thread"
}
```

**404 Not Found:**
```json
{
  "detail": "Thread not found"
}
```

**400 Bad Request - Thread Locked:**
```json
{
  "detail": "Thread is locked"
}
```

**400 Bad Request - Duplicate Message:**
```json
{
  "detail": "Duplicate client_msg_id"
}
```

## Frontend Integration Tips

### Chat List (Inbox)
1. Call `GET /threads` to get all user's chats
2. Display threads sorted by last message time
3. Show unread counts as badges
4. Display event title and last message preview

### Chat Interface
1. Call `GET /threads/{thread_id}/details` to get event info for sidebar
2. Call `GET /threads/{thread_id}/messages` to load message history
3. Connect to WebSocket for real-time updates
4. Call `POST /threads/{thread_id}/read` when user views messages

### Event Details Sidebar
Use the event data from thread details to show:
- Event title, description, time, location
- Host information
- Event tags
- Booking status

This creates an Airbnb-style experience where users can see event details alongside their conversation.
