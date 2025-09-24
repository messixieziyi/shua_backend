// Replace the existing addParticipantMessage function
function addParticipantMessage(content, type = 'user', userName = null) {
    const messagesDiv = document.getElementById('participantChat');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    
    if (type === 'user' && userName) {
        const nameDiv = document.createElement('div');
        nameDiv.style.fontSize = '12px';
        nameDiv.style.fontWeight = 'bold';
        nameDiv.style.marginBottom = '4px';
        nameDiv.style.opacity = '0.8';
        nameDiv.textContent = userName;
        messageDiv.appendChild(nameDiv);
    }
    
    const contentDiv = document.createElement('div');
    contentDiv.textContent = content;
    messageDiv.appendChild(contentDiv);
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

// Replace the existing sendParticipantMessage function
async function sendParticipantMessage(customMessage = null) {
    if (!currentUserId || !currentEventId) {
        return;
    }

    if (!currentRequest || currentRequest.status !== 'ACCEPTED') {
        return;
    }

    const input = document.getElementById('participantInput');
    const message = customMessage || input.value.trim();
    
    if (!message) return;
    
    const userName = users.find(u => u.id == currentUserId)?.full_name || 'Unknown User';
    addParticipantMessage(message, 'user', userName);
    input.value = '';
    
    // Simulate other participants responding
    setTimeout(() => {
        const otherUsers = users.filter(u => u.id != currentUserId);
        const randomUser = otherUsers[Math.floor(Math.random() * otherUsers.length)];
        const responses = [
            "Thanks for your interest! I'd be happy to have you join us.",
            "Great question! The event starts at the time shown and should be really fun.",
            "It's completely free! We just ask that you RSVP so we know how many people to expect.",
            "We'll have refreshments and some amazing activities lined up!",
            "Perfect! I'll add you to the guest list. See you there!",
            "The venue is at the location shown. Let me know if you need directions.",
            "Looking forward to meeting everyone!",
            "This is going to be such a great event!",
            "Anyone else excited about this?",
            "I've been to similar events before - they're always fun!"
        ];
        const randomResponse = responses[Math.floor(Math.random() * responses.length)];
        addParticipantMessage(randomResponse, 'user', randomUser.full_name);
    }, 1000 + Math.random() * 2000);
}

// Update selectParticipantEvent to clear chat and show welcome
function selectParticipantEvent() {
    const eventSelect = document.getElementById('eventSelect');
    const eventId = eventSelect.value;
    
    if (!eventId) {
        currentEventId = null;
        return;
    }
    
    const event = events.find(e => e.id == eventId);
    if (!event) return;
    
    console.log('Selecting participant event:', event);
    currentEventId = event.id;
    
    const userRequest = requests.find(r => r.user_id == currentUserId && r.event_id == currentEventId);
    
    if (userRequest) {
        currentRequest = userRequest;
        updateParticipantUI();
    } else {
        currentRequest = null;
        document.getElementById('participantActions').style.display = 'flex';
        document.getElementById('requestBtn').disabled = false;
        document.getElementById('rsvpGoingBtn').disabled = true;
        document.getElementById('rsvpMaybeBtn').disabled = true;
        document.getElementById('participantChatInput').style.display = 'none';
        
        // Clear chat and show welcome message
        const messagesDiv = document.getElementById('participantChat');
        messagesDiv.innerHTML = '';
        addParticipantMessage(`Welcome to the ${event.title} chat!`, 'system');
    }
}

// Update updateParticipantUI to show group chat welcome
function updateParticipantUI() {
    if (!currentRequest) return;
    
    document.getElementById('participantActions').style.display = 'flex';
    
    if (currentRequest.status === 'ACCEPTED') {
        document.getElementById('requestBtn').disabled = true;
        document.getElementById('rsvpGoingBtn').disabled = false;
        document.getElementById('rsvpMaybeBtn').disabled = false;
        document.getElementById('participantChatInput').style.display = 'flex';
        
        // Clear chat and show welcome to group chat
        const messagesDiv = document.getElementById('participantChat');
        messagesDiv.innerHTML = '';
        addParticipantMessage(`Welcome to the group chat for ${events.find(e => e.id == currentEventId)?.title}!`, 'system');
    } else if (currentRequest.status === 'SUBMITTED') {
        document.getElementById('requestBtn').disabled = true;
        document.getElementById('rsvpGoingBtn').disabled = true;
        document.getElementById('rsvpMaybeBtn').disabled = true;
        document.getElementById('participantChatInput').style.display = 'none';
    } else if (currentRequest.status === 'DECLINED') {
        document.getElementById('requestBtn').disabled = true;
        document.getElementById('rsvpGoingBtn').disabled = true;
        document.getElementById('rsvpMaybeBtn').disabled = true;
        document.getElementById('participantChatInput').style.display = 'none';
    }
}
