# 🎉 聚会服务平台 - 完整的活动管理系统

一个功能完整的聚会管理服务，包含用户认证、实时聊天、审批流程和标签系统。

## 📚 文档导航

- **[API_DOCUMENTATION.md](./API_DOCUMENTATION.md)** - 完整的 API 文档（包含所有端点详细说明和代码示例）
- **[README.md](./README.md)** - 英文版 README
- **[AUTHENTICATION.md](./AUTHENTICATION.md)** - 认证系统详细说明

## 🚀 快速开始

### 安装和运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动服务器
uvicorn main:app --host 0.0.0.0 --port 9000

# 3. 访问界面
# 打开浏览器访问: http://localhost:9000
```

### 首次使用

1. **注册账户** - 点击右上角 "Login"，切换到 "Register"
2. **登录系统** - 使用邮箱和密码登录
3. **创建事件** - 在中间面板填写事件信息
4. **管理请求** - 批准或拒绝加入请求
5. **实时聊天** - 与参与者交流

## 🔐 认证系统

### 支持的认证方式

- ✅ 用户注册（邮箱 + 密码）
- ✅ 用户登录（JWT Token）
- ✅ 自动认证（Token 过期保护）
- ✅ 密码加密（Argon2 算法）

### 前端集成示例

```javascript
// 注册
const response = await fetch('http://localhost:9000/auth/register', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        display_name: '张三',
        email: 'zhangsan@example.com',
        password: 'password123'
    })
});
const { access_token } = await response.json();

// 使用 Token 调用 API
const events = await fetch('http://localhost:9000/events', {
    headers: {
        'Authorization': `Bearer ${access_token}`
    }
});
```

## 📋 核心功能

### 🔐 用户认证
- JWT Token 认证
- 密码 Argon2 加密
- Token 过期时间：7 天
- 自动刷新机制

### 📅 事件管理
- 创建活动（标题、描述、时间、地点等）
- 添加标签分类
- 设置参与人数上限
- 查看所有公开活动

### ✅ 审批流程
- 发送加入请求
- 主办方批准/拒绝
- 状态跟踪（待审批、已接受、已拒绝）
- 自动创建聊天线程

### 💬 实时聊天
- WebSocket 实时通信
- 线程式聊天管理
- 消息已读状态
- 系统通知

### 🏷️ 标签系统
- 自定义标签（名称、颜色、描述）
- 事件标签分类
- 按标签筛选事件
- 预置标签模板

## 🔧 API 端点总览

### 认证相关
```
POST   /auth/register     - 用户注册 ✅ 公开
POST   /auth/login        - 用户登录 ✅ 公开
POST   /auth/logout       - 用户登出
GET    /auth/me           - 获取当前用户信息
```

### 事件管理
```
GET    /events            - 查看所有事件 ✅ 公开
POST   /events            - 创建事件 🔒 需要认证
GET    /events?tag_filter={tag} - 按标签筛选
```

### 请求管理
```
GET    /requests          - 我的请求列表 🔒
POST   /requests          - 发送加入请求 🔒
POST   /requests/{id}/act - 批准/拒绝请求 🔒
```

### 聊天系统
```
GET    /threads                     - 我的聊天列表 🔒
GET    /threads/{id}/messages       - 获取消息 🔒
POST   /threads/{id}/messages       - 发送消息 🔒
POST   /threads/{id}/read          - 标记已读 🔒
GET    /threads/{id}/participants  - 参与者列表 🔒
```

### 标签管理
```
GET    /tags              - 所有标签 🔒
POST   /tags              - 创建标签 🔒
DELETE /tags/{id}         - 删除标签 🔒
POST   /events/{id}/tags  - 添加标签到事件 🔒
DELETE /events/{id}/tags/{tag_id} - 移除标签 🔒
```

> 🔒 表示需要 JWT Token 认证  
> ✅ 表示公开端点，无需认证

## 📖 完整 API 文档

查看 **[API_DOCUMENTATION.md](./API_DOCUMENTATION.md)** 获取：

- ✅ 每个端点的详细说明
- ✅ 请求/响应格式
- ✅ 前端代码示例
- ✅ 错误处理指南
- ✅ 认证集成方法
- ✅ 完整业务流程示例

## 🎯 权限控制说明

### 公开端点（无需登录）
- 查看所有事件
- 用户注册
- 用户登录

### 需要认证的端点
- 创建事件
- 发送加入请求
- 查看消息和聊天
- 管理标签

### 特殊权限
- **事件创建者**：可以管理自己的事件和标签
- **主办方**：可以批准/拒绝加入请求
- **参与者**：只能在自己参与的聊天中发送消息

## 💡 使用技巧

### 1. 如何创建带标签的事件

```javascript
// 先获取标签列表
const tags = await fetch('http://localhost:9000/tags', {
    headers: { 'Authorization': `Bearer ${token}` }
}).then(r => r.json());

// 创建事件时选择标签
const event = await fetch('http://localhost:9000/events', {
    method: 'POST',
    headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        title: '周末篮球赛',
        starts_at: '2025-10-20T14:00:00Z',
        capacity: 10,
        activity_type: 'basketball',
        tag_ids: [tags[0].id, tags[1].id]  // 选择标签
    })
});
```

### 2. 如何处理加入请求

```javascript
// 查看我收到的请求
const requests = await fetch('http://localhost:9000/requests', {
    headers: { 'Authorization': `Bearer ${token}` }
}).then(r => r.json());

// 批准请求
await fetch(`http://localhost:9000/requests/${requests[0].id}/act`, {
    method: 'POST',
    headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({ action: 'accept' })
});
```

### 3. 如何发送消息

```javascript
// 获取聊天线程
const threads = await fetch('http://localhost:9000/threads', {
    headers: { 'Authorization': `Bearer ${token}` }
}).then(r => r.json());

// 发送消息
await fetch(`http://localhost:9000/threads/${threads.threads[0].id}/messages`, {
    method: 'POST',
    headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({ content: '大家好！' })
});
```

## 🛠️ 技术栈

- **后端**: FastAPI + SQLAlchemy 2.0
- **数据库**: SQLite (开发) / PostgreSQL (生产)
- **认证**: JWT + Argon2 密码加密
- **实时通信**: WebSocket
- **前端**: HTML5 + JavaScript (ES6+)

## 📊 数据库表结构

### 核心表
- `users` - 用户信息（邮箱、密码、昵称）
- `events` - 事件信息
- `requests` - 加入请求
- `bookings` - 确认的预订
- `threads` - 聊天线程
- `messages` - 聊天消息
- `tags` - 标签定义
- `event_tags` - 事件-标签关联

## 🔥 常见问题

### Q: Token 过期了怎么办？
A: Token 有效期为 7 天，过期后需要重新登录获取新 Token。

### Q: 如何修改密码？
A: 当前版本暂不支持修改密码，可以重新注册新账户。

### Q: 可以删除事件吗？
A: 当前版本暂不支持删除事件，后续版本会添加此功能。

### Q: 如何获取其他用户的信息？
A: 出于隐私保护，只能获取自己的用户信息。其他用户信息会通过事件和请求间接获取。

### Q: WebSocket 如何认证？
A: WebSocket 认证正在升级中，当前版本使用用户 ID 连接。

## 🚀 部署

### 本地开发
```bash
uvicorn main:app --host 0.0.0.0 --port 9000 --reload
```

### 生产部署
查看 [DEPLOYMENT.md](./DEPLOYMENT.md) 了解详细的部署指南。

## 📞 支持和反馈

如有问题或建议，请：
1. 查看 [API_DOCUMENTATION.md](./API_DOCUMENTATION.md)
2. 检查 [AUTHENTICATION.md](./AUTHENTICATION.md)
3. 提交 Issue 或 Pull Request

---

**版本**: 1.0  
**最后更新**: 2025-10-13

