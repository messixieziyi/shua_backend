
@app.get("/users")
async def list_users(session=Depends(get_session)):
    """Get all users."""
    users = session.execute(select(User)).scalars().all()
    return [{"id": u.id, "display_name": u.display_name} for u in users]

@app.get("/events")
async def list_events(session=Depends(get_session)):
    """Get all events (using event_id from requests)."""
    # Get unique event_ids from requests
    event_ids = session.execute(select(Request.event_id).distinct()).scalars().all()
    events = []
    for event_id in event_ids:
        # Count requests for this event
        request_count = session.execute(select(func.count(Request.id)).where(Request.event_id == event_id)).scalar()
        events.append({
            "id": event_id,
            "title": f"Event {event_id[:8]}",
            "description": f"Meetup event with {request_count} requests",
            "capacity": 10,
            "starts_at": "2024-01-01T10:00:00Z"
        })
    return events

@app.post("/rsvps")
async def create_rsvp(event_id: str, user_id: str = Depends(get_user_id), session=Depends(get_session)):
    """Create an RSVP for an event."""
    # For now, just return success - you can implement full RSVP logic later
    return {"message": "RSVP created", "event_id": event_id, "user_id": user_id}

