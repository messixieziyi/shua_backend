# 🎉 Meetup Service API 文档

完整的 API 端点文档，包含请求格式、响应格式和前端使用示例。

## 📋 目录

- [基础信息](#基础信息)
- [认证系统](#认证系统)
- [用户管理](#用户管理)
- [事件管理](#事件管理)
- [请求管理](#请求管理)
- [线程和消息](#线程和消息)
- [标签管理](#标签管理)
- [开发工具](#开发工具)

---

## 基础信息

### API Base URL
```
本地开发: http://localhost:9000
```

### 认证方式
除了公开端点外，所有端点都需要 JWT Token 认证。

**认证 Header 格式：**
```javascript
headers: {
    'Authorization': 'Bearer <your-jwt-token>',
    'Content-Type': 'application/json'
}
```

### 公开端点（无需认证）
- `GET /` - 主页
- `GET /health` - 健康检查
- `GET /events` - 查看所有事件
- `POST /auth/register` - 用户注册
- `POST /auth/login` - 用户登录

---

## 认证系统

### 1. 用户注册

**端点:** `POST /auth/register`

**请求体:**
```json
{
  "email": "user@example.com",
  "password": "your-password",
  "display_name": "Your Name"
}
```

**响应 (200 OK):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user_id": "uuid-string",
  "display_name": "Your Name",
  "email": "user@example.com"
}
```

**前端使用示例:**
```javascript
async function register(displayName, email, password) {
    const response = await fetch('http://localhost:9000/auth/register', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            display_name: displayName,
            email: email,
            password: password
        })
    });
    
    if (response.ok) {
        const data = await response.json();
        // 保存 token 和用户信息
        localStorage.setItem('auth_token', data.access_token);
        localStorage.setItem('current_user', JSON.stringify({
            id: data.user_id,
            display_name: data.display_name,
            email: data.email
        }));
        return data;
    } else {
        const error = await response.json();
        throw new Error(error.detail);
    }
}
```

**验证规则:**
- Email: 必须是有效的邮箱格式
- Password: 最少 6 个字符
- Display Name: 必填

---

### 2. 用户登录

**端点:** `POST /auth/login`

**请求体:**
```json
{
  "email": "user@example.com",
  "password": "your-password"
}
```

**响应 (200 OK):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user_id": "uuid-string",
  "display_name": "Your Name",
  "email": "user@example.com"
}
```

**前端使用示例:**
```javascript
async function login(email, password) {
    const response = await fetch('http://localhost:9000/auth/login', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ email, password })
    });
    
    if (response.ok) {
        const data = await response.json();
        localStorage.setItem('auth_token', data.access_token);
        localStorage.setItem('current_user', JSON.stringify({
            id: data.user_id,
            display_name: data.display_name,
            email: data.email
        }));
        return data;
    } else {
        throw new Error('Login failed');
    }
}
```

---

### 3. 用户登出

**端点:** `POST /auth/logout`

**说明:** 这是一个客户端操作，后端不做处理。前端需要删除存储的 token。

**前端使用示例:**
```javascript
function logout() {
    localStorage.removeItem('auth_token');
    localStorage.removeItem('current_user');
    // 重定向到登录页或刷新页面
}
```

---

### 4. 获取当前用户信息

**端点:** `GET /auth/me`

**需要认证:** ✅

**响应 (200 OK):**
```json
{
  "id": "uuid-string",
  "email": "user@example.com",
  "display_name": "Your Name"
}
```

**前端使用示例:**
```javascript
async function getCurrentUser(token) {
    const response = await fetch('http://localhost:9000/auth/me', {
        headers: {
            'Authorization': `Bearer ${token}`
        }
    });
    
    if (response.ok) {
        return await response.json();
    }
}
```

---

## 用户管理

### 5. 获取当前用户信息

**端点:** `GET /users`

**需要认证:** ✅

**说明:** 返回当前已登录用户的信息（不是所有用户）

**响应 (200 OK):**
```json
{
  "id": "uuid-string",
  "display_name": "Your Name",
  "email": "user@example.com"
}
```

**前端使用示例:**
```javascript
async function getUserInfo(token) {
    const response = await fetch('http://localhost:9000/users', {
        headers: {
            'Authorization': `Bearer ${token}`
        }
    });
    
    return await response.json();
}
```

---

## 事件管理

### 6. 查看所有事件

**端点:** `GET /events`

**需要认证:** ❌ (公开端点)

**查询参数:**
- `tag_filter` (可选): 按标签名称过滤事件

**示例:**
```
GET /events
GET /events?tag_filter=Beginner
```

**响应 (200 OK):**
```json
[
  {
    "id": "uuid-string",
    "title": "Morning Tennis",
    "description": "Friendly tennis match",
    "starts_at": "2025-10-15T10:00:00",
    "capacity": 10,
    "activity_type": "tennis",
    "location": "Central Park",
    "address": "123 Park Ave",
    "created_by": "user-uuid",
    "tags": [
      {
        "id": "tag-uuid",
        "name": "Beginner",
        "color": "#10b981",
        "description": "Suitable for beginners"
      }
    ]
  }
]
```

**前端使用示例:**
```javascript
// 获取所有事件
async function getAllEvents() {
    const response = await fetch('http://localhost:9000/events');
    return await response.json();
}

// 按标签过滤
async function getEventsByTag(tagName) {
    const response = await fetch(
        `http://localhost:9000/events?tag_filter=${encodeURIComponent(tagName)}`
    );
    return await response.json();
}
```

---

### 7. 创建事件

**端点:** `POST /events`

**需要认证:** ✅

**请求体:**
```json
{
  "title": "Morning Tennis",
  "description": "Friendly tennis match",
  "starts_at": "2025-10-15T10:00:00Z",
  "capacity": 10,
  "activity_type": "tennis",
  "location": "Central Park",
  "address": "123 Park Ave",
  "tag_ids": ["tag-uuid-1", "tag-uuid-2"]
}
```

**字段说明:**
- `title` (必填): 事件标题
- `description` (可选): 事件描述
- `starts_at` (必填): ISO 8601 格式的日期时间
- `capacity` (必填): 参与人数上限
- `activity_type` (必填): 活动类型 (tennis, basketball, yoga, etc.)
- `location` (可选): 地点名称
- `address` (可选): 详细地址
- `tag_ids` (可选): 标签 ID 数组

**注意:** `created_by` 字段会被后端自动设置为当前登录用户，不需要也不应该由前端提供。

**响应 (200 OK):**
```json
{
  "id": "uuid-string",
  "title": "Morning Tennis",
  "description": "Friendly tennis match",
  "capacity": 10,
  "starts_at": "2025-10-15T10:00:00",
  "activity_type": "tennis",
  "location": "Central Park",
  "address": "123 Park Ave",
  "created_by": "current-user-uuid",
  "tags": []
}
```

**前端使用示例:**
```javascript
async function createEvent(eventData, token) {
    // 构建日期时间
    const startsAt = new Date(
        eventData.date + 'T' + eventData.time
    ).toISOString();
    
    const response = await fetch('http://localhost:9000/events', {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            title: eventData.title,
            description: eventData.description,
            starts_at: startsAt,
            capacity: parseInt(eventData.capacity),
            activity_type: eventData.activity_type,
            location: eventData.location,
            address: eventData.address,
            tag_ids: eventData.selectedTags || []
        })
    });
    
    if (response.ok) {
        return await response.json();
    } else {
        const error = await response.text();
        throw new Error(error);
    }
}
```

---

## 请求管理

### 8. 获取我的请求

**端点:** `GET /requests`

**需要认证:** ✅

**说明:** 返回当前用户作为 guest 或 host 的所有请求

**响应 (200 OK):**
```json
[
  {
    "id": "request-uuid",
    "event_id": "event-uuid",
    "user_id": "guest-uuid",
    "host_id": "host-uuid",
    "status": "SUBMITTED",
    "user_name": "Guest Name",
    "host_name": "Host Name",
    "event_title": "Morning Tennis",
    "created_at": "2025-10-13T10:00:00"
  }
]
```

**状态值:**
- `SUBMITTED` - 已提交，等待批准
- `ACCEPTED` - 已接受
- `DECLINED` - 已拒绝

**前端使用示例:**
```javascript
async function getMyRequests(token) {
    const response = await fetch('http://localhost:9000/requests', {
        headers: {
            'Authorization': `Bearer ${token}`
        }
    });
    return await response.json();
}
```

---

### 9. 获取所有相关请求

**端点:** `GET /requests/all`

**需要认证:** ✅

**说明:** 与 `/requests` 相同，返回当前用户相关的所有请求

**响应:** 同上

---

### 10. 创建加入请求

**端点:** `POST /requests`

**需要认证:** ✅

**请求体:**
```json
{
  "event_id": "event-uuid",
  "auto_accept": false
}
```

**字段说明:**
- `event_id` (必填): 要加入的事件 ID
- `auto_accept` (可选): 是否自动接受，默认 false

**注意:** 
- `guest_id` 自动设置为当前登录用户
- `host_id` 自动设置为事件创建者
- 每个用户对同一事件只能创建一个请求

**响应 (200 OK):**
```json
{
  "request_id": "request-uuid",
  "thread_id": "thread-uuid",
  "status": "SUBMITTED"
}
```

**前端使用示例:**
```javascript
async function requestToJoinEvent(eventId, token) {
    const response = await fetch('http://localhost:9000/requests', {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            event_id: eventId,
            auto_accept: false
        })
    });
    
    if (response.ok) {
        return await response.json();
    } else {
        const error = await response.json();
        throw new Error(error.detail);
    }
}
```

---

### 11. 批准/拒绝请求

**端点:** `POST /requests/{request_id}/act`

**需要认证:** ✅

**权限:** 只有事件主办者（host）可以批准或拒绝请求

**请求体:**
```json
{
  "action": "accept"
}
```

**字段说明:**
- `action` (必填): "accept" 或 "decline"

**响应 (200 OK):**
```json
{
  "status": "ACCEPTED",
  "thread_id": "thread-uuid"
}
```

**前端使用示例:**
```javascript
async function approveRequest(requestId, token) {
    const response = await fetch(
        `http://localhost:9000/requests/${requestId}/act`,
        {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ action: 'accept' })
        }
    );
    return await response.json();
}

async function declineRequest(requestId, token) {
    const response = await fetch(
        `http://localhost:9000/requests/${requestId}/act`,
        {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ action: 'decline' })
        }
    );
    return await response.json();
}
```

---

## 线程和消息

### 12. 获取我的线程列表

**端点:** `GET /threads`

**需要认证:** ✅

**说明:** 返回当前用户参与的所有聊天线程

**响应 (200 OK):**
```json
{
  "threads": [
    {
      "id": "thread-uuid",
      "scope": "REQUEST",
      "request_id": "request-uuid",
      "booking_id": null,
      "event_id": "event-uuid",
      "is_locked": false
    }
  ],
  "participants": {
    "thread-uuid": [
      {
        "user_id": "user-uuid",
        "display_name": "User Name",
        "role": "guest"
      }
    ]
  }
}
```

**线程范围 (scope):**
- `REQUEST` - 请求相关的聊天
- `BOOKING` - 预订相关的聊天

**前端使用示例:**
```javascript
async function getMyThreads(token) {
    const response = await fetch('http://localhost:9000/threads', {
        headers: {
            'Authorization': `Bearer ${token}`
        }
    });
    return await response.json();
}
```

---

### 13. 获取线程消息

**端点:** `GET /threads/{thread_id}/messages`

**需要认证:** ✅

**权限:** 只能查看自己参与的线程消息

**查询参数:**
- `limit` (可选): 返回消息数量，默认 50
- `offset` (可选): 偏移量，默认 0

**示例:**
```
GET /threads/thread-uuid/messages?limit=20&offset=0
```

**响应 (200 OK):**
```json
{
  "messages": [
    {
      "id": "message-uuid",
      "thread_id": "thread-uuid",
      "sender_id": "user-uuid",
      "content": "Hello!",
      "created_at": "2025-10-13T10:00:00",
      "sender_name": "User Name"
    }
  ],
  "total": 10
}
```

**前端使用示例:**
```javascript
async function getThreadMessages(threadId, token, limit = 50, offset = 0) {
    const response = await fetch(
        `http://localhost:9000/threads/${threadId}/messages?limit=${limit}&offset=${offset}`,
        {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        }
    );
    return await response.json();
}
```

---

### 14. 发送消息

**端点:** `POST /threads/{thread_id}/messages`

**需要认证:** ✅

**权限:** 只能在自己参与的线程中发送消息

**请求体:**
```json
{
  "content": "Hello, how are you?"
}
```

**响应 (200 OK):**
```json
{
  "id": "message-uuid",
  "thread_id": "thread-uuid",
  "sender_id": "user-uuid",
  "content": "Hello, how are you?",
  "created_at": "2025-10-13T10:00:00"
}
```

**前端使用示例:**
```javascript
async function sendMessage(threadId, content, token) {
    const response = await fetch(
        `http://localhost:9000/threads/${threadId}/messages`,
        {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ content })
        }
    );
    return await response.json();
}
```

---

### 15. 标记线程为已读

**端点:** `POST /threads/{thread_id}/read`

**需要认证:** ✅

**说明:** 标记当前用户在该线程中的所有消息为已读

**响应 (200 OK):**
```json
{
  "message": "Thread marked as read"
}
```

**前端使用示例:**
```javascript
async function markThreadAsRead(threadId, token) {
    const response = await fetch(
        `http://localhost:9000/threads/${threadId}/read`,
        {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`
            }
        }
    );
    return await response.json();
}
```

---

### 16. 获取线程参与者

**端点:** `GET /threads/{thread_id}/participants`

**需要认证:** ✅

**响应 (200 OK):**
```json
[
  {
    "user_id": "user-uuid",
    "display_name": "User Name",
    "role": "guest",
    "joined_at": "2025-10-13T10:00:00"
  }
]
```

**前端使用示例:**
```javascript
async function getThreadParticipants(threadId, token) {
    const response = await fetch(
        `http://localhost:9000/threads/${threadId}/participants`,
        {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        }
    );
    return await response.json();
}
```

---

## 标签管理

### 17. 获取所有标签

**端点:** `GET /tags`

**需要认证:** ✅

**响应 (200 OK):**
```json
[
  {
    "id": "tag-uuid",
    "name": "Beginner",
    "color": "#10b981",
    "description": "Suitable for beginners",
    "created_at": "2025-10-13T10:00:00"
  }
]
```

**前端使用示例:**
```javascript
async function getAllTags(token) {
    const response = await fetch('http://localhost:9000/tags', {
        headers: {
            'Authorization': `Bearer ${token}`
        }
    });
    return await response.json();
}
```

---

### 18. 创建标签

**端点:** `POST /tags`

**需要认证:** ✅

**请求体:**
```json
{
  "name": "Beginner",
  "color": "#10b981",
  "description": "Suitable for beginners"
}
```

**字段说明:**
- `name` (必填): 标签名称，必须唯一
- `color` (可选): 十六进制颜色代码，默认 "#e5e7eb"
- `description` (可选): 标签描述

**响应 (200 OK):**
```json
{
  "id": "tag-uuid",
  "name": "Beginner",
  "color": "#10b981",
  "description": "Suitable for beginners",
  "created_at": "2025-10-13T10:00:00"
}
```

**前端使用示例:**
```javascript
async function createTag(name, color, description, token) {
    const response = await fetch('http://localhost:9000/tags', {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            name,
            color: color || '#e5e7eb',
            description: description || ''
        })
    });
    return await response.json();
}
```

---

### 19. 删除标签

**端点:** `DELETE /tags/{tag_id}`

**需要认证:** ✅

**说明:** 删除标签会同时删除所有事件与该标签的关联

**响应 (200 OK):**
```json
{
  "message": "Tag deleted successfully"
}
```

**前端使用示例:**
```javascript
async function deleteTag(tagId, token) {
    const response = await fetch(`http://localhost:9000/tags/${tagId}`, {
        method: 'DELETE',
        headers: {
            'Authorization': `Bearer ${token}`
        }
    });
    return await response.json();
}
```

---

### 20. 为事件添加标签

**端点:** `POST /events/{event_id}/tags`

**需要认证:** ✅

**权限:** 只有事件创建者可以管理事件标签

**请求体:**
```json
{
  "tag_ids": ["tag-uuid-1", "tag-uuid-2"]
}
```

**响应 (200 OK):**
```json
{
  "message": "Tags added to event successfully",
  "added": ["tag-uuid-1", "tag-uuid-2"]
}
```

**前端使用示例:**
```javascript
async function addTagsToEvent(eventId, tagIds, token) {
    const response = await fetch(
        `http://localhost:9000/events/${eventId}/tags`,
        {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ tag_ids: tagIds })
        }
    );
    return await response.json();
}
```

---

### 21. 从事件移除标签

**端点:** `DELETE /events/{event_id}/tags/{tag_id}`

**需要认证:** ✅

**权限:** 只有事件创建者可以管理事件标签

**响应 (200 OK):**
```json
{
  "message": "Tag removed from event successfully"
}
```

**前端使用示例:**
```javascript
async function removeTagFromEvent(eventId, tagId, token) {
    const response = await fetch(
        `http://localhost:9000/events/${eventId}/tags/${tagId}`,
        {
            method: 'DELETE',
            headers: {
                'Authorization': `Bearer ${token}`
            }
        }
    );
    return await response.json();
}
```

---

## 开发工具

### 22. 健康检查

**端点:** `GET /health`

**需要认证:** ❌

**响应 (200 OK):**
```json
{
  "status": "healthy",
  "message": "App is running"
}
```

---

### 23. 创建示例标签

**端点:** `POST /dev/seed-tags`

**需要认证:** ✅

**说明:** 创建一组预定义的示例标签用于开发和测试

**响应 (200 OK):**
```json
{
  "message": "Sample tags created successfully",
  "created_tags": ["Beginner", "Advanced", "Outdoor", ...]
}
```

---

## 错误响应格式

所有错误响应遵循以下格式：

```json
{
  "detail": "Error message describing what went wrong"
}
```

### 常见 HTTP 状态码

- `200 OK` - 请求成功
- `400 Bad Request` - 请求参数错误
- `401 Unauthorized` - 未认证或 token 无效
- `403 Forbidden` - 没有权限执行此操作
- `404 Not Found` - 资源不存在
- `500 Internal Server Error` - 服务器内部错误

---

## 前端认证工具函数

### 完整的认证辅助函数

```javascript
// 全局认证状态
let authToken = null;
let currentUser = null;

// 初始化认证状态
function initAuth() {
    authToken = localStorage.getItem('auth_token');
    const userStr = localStorage.getItem('current_user');
    if (userStr) {
        try {
            currentUser = JSON.parse(userStr);
        } catch (e) {
            console.error('Failed to parse user data:', e);
            logout();
        }
    }
}

// 获取认证 Headers
function getAuthHeaders() {
    if (!authToken) {
        throw new Error('Not authenticated');
    }
    return {
        'Authorization': `Bearer ${authToken}`,
        'Content-Type': 'application/json'
    };
}

// 注册
async function register(displayName, email, password) {
    const response = await fetch('http://localhost:9000/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ display_name: displayName, email, password })
    });
    
    if (response.ok) {
        const data = await response.json();
        authToken = data.access_token;
        currentUser = {
            id: data.user_id,
            display_name: data.display_name,
            email: data.email
        };
        localStorage.setItem('auth_token', authToken);
        localStorage.setItem('current_user', JSON.stringify(currentUser));
        return data;
    } else {
        const error = await response.json();
        throw new Error(error.detail);
    }
}

// 登录
async function login(email, password) {
    const response = await fetch('http://localhost:9000/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
    });
    
    if (response.ok) {
        const data = await response.json();
        authToken = data.access_token;
        currentUser = {
            id: data.user_id,
            display_name: data.display_name,
            email: data.email
        };
        localStorage.setItem('auth_token', authToken);
        localStorage.setItem('current_user', JSON.stringify(currentUser));
        return data;
    } else {
        throw new Error('Login failed');
    }
}

// 登出
function logout() {
    authToken = null;
    currentUser = null;
    localStorage.removeItem('auth_token');
    localStorage.removeItem('current_user');
}

// 处理 401 错误
async function fetchWithAuth(url, options = {}) {
    const response = await fetch(url, {
        ...options,
        headers: {
            ...getAuthHeaders(),
            ...options.headers
        }
    });
    
    if (response.status === 401) {
        logout();
        window.location.href = '/'; // 重定向到登录页
        throw new Error('Authentication expired');
    }
    
    return response;
}

// 页面加载时初始化
initAuth();
```

---

## 使用示例：创建完整的事件流程

```javascript
// 1. 用户注册
await register('John Doe', 'john@example.com', 'password123');

// 2. 获取所有标签
const tags = await getAllTags(authToken);

// 3. 创建事件
const event = await createEvent({
    title: 'Morning Tennis',
    description: 'Fun tennis match',
    date: '2025-10-15',
    time: '10:00',
    capacity: 10,
    activity_type: 'tennis',
    location: 'Central Park',
    selectedTags: [tags[0].id]
}, authToken);

// 4. 另一个用户请求加入
await requestToJoinEvent(event.id, anotherUserToken);

// 5. 事件创建者批准请求
const requests = await getMyRequests(authToken);
await approveRequest(requests[0].id, authToken);

// 6. 在聊天线程中发送消息
const threads = await getMyThreads(authToken);
await sendMessage(threads.threads[0].id, 'Welcome!', authToken);
```

---

## 注意事项

1. **Token 过期**: JWT Token 在 7 天后过期，需要重新登录
2. **安全性**: 永远不要在客户端代码中暴露 `SECRET_KEY`
3. **HTTPS**: 生产环境中务必使用 HTTPS
4. **CORS**: 如果前端和后端在不同域名，需要配置 CORS
5. **错误处理**: 始终检查响应状态码并适当处理错误

---

## 支持的活动类型

- `tennis` - 网球
- `basketball` - 篮球
- `yoga` - 瑜伽
- `golf` - 高尔夫
- `running` - 跑步
- `swimming` - 游泳
- `hiking` - 徒步
- `cycling` - 骑行

可以在 `main.py` 中添加更多活动类型。

---

## 开发环境设置

```bash
# 安装依赖
pip install -r requirements.txt

# 启动开发服务器
uvicorn main:app --host 0.0.0.0 --port 9000 --reload

# 访问
http://localhost:9000
```

---

**文档版本**: 1.0  
**最后更新**: 2025-10-13  
**API 版本**: v1

