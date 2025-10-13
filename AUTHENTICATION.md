# Authentication System

This application now uses JWT (JSON Web Token) based authentication. All endpoints except `/events` (list events) require authentication.

## Authentication Flow

### 1. Register a New User
**POST** `/auth/register`

Request body:
```json
{
  "email": "user@example.com",
  "password": "yourpassword123",
  "display_name": "John Doe"
}
```

Response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user_id": "uuid-here",
  "display_name": "John Doe",
  "email": "user@example.com"
}
```

### 2. Login (Sign In)
**POST** `/auth/login`

Request body:
```json
{
  "email": "user@example.com",
  "password": "yourpassword123"
}
```

Response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user_id": "uuid-here",
  "display_name": "John Doe",
  "email": "user@example.com"
}
```

### 3. Get Current User Info
**GET** `/auth/me`

Headers:
```
Authorization: Bearer <access_token>
```

Response:
```json
{
  "id": "uuid-here",
  "email": "user@example.com",
  "display_name": "John Doe"
}
```

### 4. Logout (Sign Off)
**POST** `/auth/logout`

Headers:
```
Authorization: Bearer <access_token>
```

Response:
```json
{
  "message": "Successfully logged out"
}
```

**Note:** With JWT tokens, logout is handled client-side by discarding the token. The server simply confirms the request.

## Using Authentication in Your Frontend

### 1. Save the Token After Login/Register
```javascript
// After successful login or registration
const response = await fetch('/auth/login', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ email, password })
});

const data = await response.json();
// Save token to localStorage or sessionStorage
localStorage.setItem('access_token', data.access_token);
localStorage.setItem('user_id', data.user_id);
localStorage.setItem('display_name', data.display_name);
```

### 2. Include Token in All Authenticated Requests
```javascript
const token = localStorage.getItem('access_token');

const response = await fetch('/threads', {
  method: 'GET',
  headers: {
    'Authorization': `Bearer ${token}`
  }
});
```

### 3. Handle Authentication Errors
```javascript
const response = await fetch('/threads', {
  method: 'GET',
  headers: {
    'Authorization': `Bearer ${token}`
  }
});

if (response.status === 401) {
  // Token is invalid or expired, redirect to login
  localStorage.removeItem('access_token');
  window.location.href = '/login';
}
```

### 4. Logout
```javascript
const token = localStorage.getItem('access_token');

await fetch('/auth/logout', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`
  }
});

// Clear local storage
localStorage.removeItem('access_token');
localStorage.removeItem('user_id');
localStorage.removeItem('display_name');

// Redirect to login page
window.location.href = '/login';
```

## Protected vs Public Endpoints

### Public Endpoints (No Authentication Required)
- `GET /events` - List all events
- `POST /auth/register` - Register a new user
- `POST /auth/login` - Login
- `GET /` - Serve index.html
- `GET /health` - Health check

### Protected Endpoints (Authentication Required)
All other endpoints require the `Authorization: Bearer <token>` header:
- User management: `/users`, `/auth/me`
- Event creation: `POST /events`
- Requests: `/requests`, `/requests/{id}/act`
- Threads & Messages: `/threads`, `/threads/{id}/messages`
- Tags: `/tags`, `/events/{id}/tags`
- RSVPs: `/rsvps`

## Security Considerations

1. **Token Expiration**: Tokens expire after 7 days by default
2. **HTTPS**: Always use HTTPS in production to protect tokens
3. **Secret Key**: Set a strong `SECRET_KEY` environment variable in production
4. **Password Requirements**: Minimum 6 characters (enforced by validation)

## Environment Variables

Set these in production:
```bash
SECRET_KEY=your-very-secure-random-secret-key-here
DATABASE_URL=postgresql://user:pass@host/db
```

## Migration Notes

The User model now requires:
- `email` (unique, required)
- `hashed_password` (required)

If you have existing users in the database, you'll need to:
1. Drop the existing database (or migrate data)
2. Recreate tables with the new schema
3. Re-register all users with email and password

