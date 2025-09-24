# Railway Deployment Guide

This guide will help you deploy your FastAPI Meetup Chat & Booking API to Railway.

## Prerequisites

1. A Railway account (sign up at [railway.app](https://railway.app))
2. Git repository with your code
3. Railway CLI (optional but recommended)

## Step 1: Prepare Your Repository

Your project should have these files:
- `requirements.txt` - Python dependencies
- `railway.toml` - Railway configuration
- `Procfile` - Process definition
- `app_production.py` - Production-ready FastAPI app

## Step 2: Deploy to Railway

### Option A: Using Railway Dashboard (Recommended)

1. Go to [railway.app](https://railway.app) and sign in
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Connect your GitHub account and select this repository
5. Railway will automatically detect it's a Python project
6. Add a PostgreSQL database:
   - Click "New" → "Database" → "PostgreSQL"
   - Railway will automatically set the `DATABASE_URL` environment variable

### Option B: Using Railway CLI

1. Install Railway CLI:
   ```bash
   npm install -g @railway/cli
   ```

2. Login to Railway:
   ```bash
   railway login
   ```

3. Initialize project:
   ```bash
   railway init
   ```

4. Add PostgreSQL database:
   ```bash
   railway add postgresql
   ```

5. Deploy:
   ```bash
   railway up
   ```

## Step 3: Configure Environment Variables

Railway will automatically set:
- `DATABASE_URL` - PostgreSQL connection string
- `PORT` - Port number for the application

## Step 4: Verify Deployment

1. Check the deployment logs in Railway dashboard
2. Visit your app URL (provided by Railway)
3. Test the health endpoint: `https://your-app.railway.app/health`
4. Test the API: `https://your-app.railway.app/`

## Step 5: Seed the Database (Optional)

You can seed the database with test data by calling:
```bash
curl -X POST https://your-app.railway.app/dev/seed
```

## Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DATABASE_URL` | PostgreSQL connection string | - | Yes (auto-set by Railway) |
| `PORT` | Port for the application | 8000 | No (auto-set by Railway) |

## Troubleshooting

### Common Issues

1. **Database Connection Error**
   - Ensure PostgreSQL service is added to your project
   - Check that `DATABASE_URL` is set correctly

2. **Port Binding Error**
   - Make sure your app binds to `0.0.0.0` and uses `$PORT` environment variable

3. **Dependencies Error**
   - Check that all dependencies are listed in `requirements.txt`
   - Ensure Python version compatibility

### Logs

View logs in Railway dashboard or using CLI:
```bash
railway logs
```

## Production Considerations

1. **Database**: Railway provides managed PostgreSQL with automatic backups
2. **Scaling**: Railway can auto-scale based on traffic
3. **Monitoring**: Use Railway's built-in monitoring and metrics
4. **Custom Domain**: Add custom domain in Railway dashboard
5. **Environment**: Use Railway's environment variables for configuration

## API Endpoints

Once deployed, your API will be available at:
- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /requests` - Create request
- `POST /requests/{id}/act` - Act on request
- `POST /dev/seed` - Seed database (development only)

## Support

- Railway Documentation: [docs.railway.app](https://docs.railway.app)
- Railway Discord: [discord.gg/railway](https://discord.gg/railway)
