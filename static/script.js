async function askQuestion() {
    const input = document.getElementById('user-input');
    const chatBox = document.getElementById('chat-box');
    const query = input.value.trim();

    if (!query) return;

    // 1. Add User Message to UI
    chatBox.innerHTML += `
        <div class="message user-msg">
            <b>You:</b> ${query}
        </div>
    `;
    
    // Clear input and scroll to bottom
    input.value = "";
    chatBox.scrollTop = chatBox.scrollHeight;

    // 2. Add a "Thinking..." placeholder
    const loadingId = "loading-" + Date.now();
    chatBox.innerHTML += `
        <div class="message ai-msg" id="${loadingId}">
            <i>AI is thinking...</i>
        </div>
    `;
    chatBox.scrollTop = chatBox.scrollHeight;

    try {
        // 3. Call Flask Backend
        const response = await fetch('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query })
        });

        const data = await response.json();

        // 4. Replace "Thinking..." with actual AI response
        const loadingElement = document.getElementById(loadingId);
        if (data.answer) {
            loadingElement.innerHTML = `<b>AI:</b> ${data.answer.replace(/\n/g, '<br>')}`;
        } else {
            loadingElement.innerHTML = `<b>AI:</b> Sorry, I encountered an error.`;
        }

    } catch (error) {
        console.error("Error:", error);
        document.getElementById(loadingId).innerHTML = `<b>AI:</b> Connection failed. Is Flask running?`;
    }

    // Final scroll to bottom
    chatBox.scrollTop = chatBox.scrollHeight;
}

// 5. Allow "Enter" key to send message
document.getElementById('user-input').addEventListener('keypress', function (e) {
    if (e.key === 'Enter') {
        askQuestion();
    }
});