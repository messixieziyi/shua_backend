import re

# Read the current file
with open('split_interface_four_panels_fixed.html', 'r') as f:
    content = f.read()

# Replace the addParticipantMessage function
old_add_message = r'function addParticipantMessage\(content, type = "user"\) \{[^}]*\}'
new_add_message = '''function addParticipantMessage(content, type = 'user', userName = null) {
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
}'''

content = re.sub(old_add_message, new_add_message, content, flags=re.DOTALL)

# Replace sendParticipantMessage to include user names
old_send_message = r'async function sendParticipantMessage\(customMessage = null\) \{[^}]*\}'
new_send_message = '''async function sendParticipantMessage(customMessage = null) {
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
}'''

content = re.sub(old_send_message, new_send_message, content, flags=re.DOTALL)

# Write the updated content
with open('split_interface_chat_improved.html', 'w') as f:
    f.write(content)

print("Chat improvements applied!")
