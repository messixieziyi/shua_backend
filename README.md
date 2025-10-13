# 🎉 Meetup Service - Complete Event Management Platform

一个完整的聚会管理服务，包含用户认证、实时聊天、审批流程和高级标签系统。使用 FastAPI、SQLAlchemy 和现代 Web 技术构建。

> **📚 完整 API 文档**: 查看 [API_DOCUMENTATION.md](./API_DOCUMENTATION.md) 获取所有端点的详细说明和使用示例。

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

## 🔐 认证系统

### **JWT Token 认证**
- ✅ 用户注册 (`POST /auth/register`)
- ✅ 用户登录 (`POST /auth/login`)
- ✅ 用户登出 (`POST /auth/logout`)
- ✅ 获取当前用户信息 (`GET /auth/me`)
- ✅ Token 过期时间：7 天
- ✅ 密码加密：Argon2 算法

### **权限控制**
- ✅ 除查看事件外，所有操作需要认证
- ✅ 用户只能操作自己的数据
- ✅ 事件创建者可以管理自己的事件
- ✅ 只有主办方可以批准/拒绝加入请求

## 🎯 Current Features

### 🏷️ **Advanced Tagging System**
- **Tag Management**: Create, edit, delete tags with custom colors
- **Event Tagging**: Assign multiple tags to events
- **Tag Filtering**: Filter events by tags in real-time
- **Visual Tags**: Color-coded badges with automatic contrast
- **Sample Tags**: 15 pre-built categories (Beginner, Advanced, Outdoor, etc.)

### 👥 **用户管理**
- **用户注册**: 使用邮箱和密码注册
- **用户登录**: JWT Token 认证
- **用户信息**: 获取当前用户信息
- **安全性**: 密码 Argon2 加密，Token 过期保护

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

> **📖 详细文档**: 查看 [API_DOCUMENTATION.md](./API_DOCUMENTATION.md) 获取完整的 API 文档，包括：
> - 所有端点的详细说明
> - 请求/响应格式
> - 前端使用示例
> - 错误处理
> - 认证方式

### 认证端点
```
POST   /auth/register      - 用户注册 (公开)
POST   /auth/login         - 用户登录 (公开)
POST   /auth/logout        - 用户登出
GET    /auth/me            - 获取当前用户信息
```

### 核心端点
```
GET    /                   - 主页界面
GET    /health             - 健康检查 (公开)
GET    /users              - 获取当前用户信息
GET    /events             - 查看所有事件 (公开)
POST   /events             - 创建新事件
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

### 1. 克隆和设置
```bash
git clone <repository-url>
cd Shua
pip install -r requirements.txt
```

### 2. 启动开发服务器
```bash
# 方式 1: 使用 uvicorn
uvicorn main:app --host 0.0.0.0 --port 9000

# 方式 2: 直接运行 main.py
python main.py
```

### 3. 访问界面
打开浏览器访问 `http://localhost:9000`

### 4. 首次使用
1. **注册账户**: 点击右上角 "Login" 按钮，切换到 "Register" 标签
2. **登录**: 使用注册的邮箱和密码登录
3. **创建标签**: 在右侧 "Tag Management" 面板创建标签
4. **创建事件**: 在中间面板创建你的第一个事件
5. **开始使用**: 浏览事件、发送请求、聊天交流！

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

## 🔌 前端集成指南

### 给前端开发者

本后端提供完整的 REST API 和 WebSocket 接口，可以轻松集成到任何前端框架。

#### API Base URL
```
Production: https://sweet-creativity-production.up.railway.app/
Development: http://localhost:9000
```

#### 认证集成
```javascript
// 1. 注册用户
const response = await fetch('http://localhost:9000/auth/register', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        display_name: 'John Doe',
        email: 'john@example.com',
        password: 'password123'
    })
});
const { access_token } = await response.json();

// 2. 保存 Token
localStorage.setItem('auth_token', access_token);

// 3. 使用 Token 调用 API
const events = await fetch('http://localhost:9000/events', {
    headers: {
        'Authorization': `Bearer ${access_token}`,
        'Content-Type': 'application/json'
    }
});
```

#### 完整示例代码
查看 [API_DOCUMENTATION.md](./API_DOCUMENTATION.md) 获取：
- 🔐 认证辅助函数
- 📝 所有端点的使用示例
- ⚠️ 错误处理最佳实践
- 🎯 完整的业务流程示例

#### WebSocket 连接
```javascript
// 注意：WebSocket 目前正在升级以支持 JWT 认证
const ws = new WebSocket('ws://localhost:9000/ws/user123');
ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    console.log('New message:', message);
};
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