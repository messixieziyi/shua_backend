"""
Simple Meetup Service - Working Version
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="Simple Meetup Service")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
async def root():
    return {"message": "Meetup Service is running!"}

@app.get("/users")
async def get_users():
    return [
        {"id": 1, "full_name": "Sarah Johnson", "handle": "sarah_j"},
        {"id": 2, "full_name": "Mike Chen", "handle": "mike_c"},
        {"id": 3, "full_name": "Emma Davis", "handle": "emma_d"},
        {"id": 4, "full_name": "Alex Rodriguez", "handle": "alex_r"},
        {"id": 5, "full_name": "Lisa Wang", "handle": "lisa_w"}
    ]

@app.get("/events")
async def get_events():
    return [
        {"id": 1, "title": "Morning Tennis Doubles", "starts_at": "2024-01-15T09:00:00Z", "capacity": 8},
        {"id": 2, "title": "Basketball Pickup Game", "starts_at": "2024-01-16T18:00:00Z", "capacity": 20},
        {"id": 3, "title": "Sunset Yoga Session", "starts_at": "2024-01-17T18:00:00Z", "capacity": 25},
        {"id": 4, "title": "Golf Tournament", "starts_at": "2024-01-19T08:00:00Z", "capacity": 32},
        {"id": 5, "title": "Morning Run Group", "starts_at": "2024-01-20T07:00:00Z", "capacity": 30}
    ]

@app.post("/requests")
async def create_request(request_data: dict):
    return {
        "request_id": "req_123",
        "thread_id": "thread_456", 
        "status": "SUBMITTED",
        "message": "Request created successfully!"
    }

@app.post("/rsvps")
async def create_rsvp(rsvp_data: dict):
    return {
        "rsvp_id": "rsvp_789",
        "status": rsvp_data.get("status", "going"),
        "message": "RSVP created successfully!"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
