const apiUrl = "https://aquib8112--carchatbot-rag-call-dev.modal.run"; 
const messagesDiv = document.getElementById('messages');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');

function appendMessage(sender, text) {
  const msgDiv = document.createElement('div');
  msgDiv.className = 'msg ' + sender;
  msgDiv.innerHTML = text; // No label, just the message
  messagesDiv.appendChild(msgDiv);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

let chatHistory = [];

async function sendMessage() {
  const text = userInput.value.trim();
  if (!text) return;
  appendMessage('user', text);
  userInput.value = '';
  sendBtn.disabled = true;
  appendMessage('assistant', '<em>Fetching Documents....</em>');

  // Add user message to chat history
  // chatHistory.push({ role: "user", content: text });

  try {
    const response = await fetch(apiUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: text,
        chat_history: chatHistory
      })
    });
    const data = await response.json();
    messagesDiv.removeChild(messagesDiv.lastChild);
    appendMessage('assistant', data.answer || 'No answer received.');

    if (data.answer) {
        // Only add to chat history if we got a valid answer
        chatHistory.push({ role: "user", content: text });
        chatHistory.push({ role: "assistant", content: data.answer });
    }

    // Keep only the last 4 messages (2 exchanges)
    if (chatHistory.length > 4) chatHistory = chatHistory.slice(-4);

  } catch (e) {
    messagesDiv.removeChild(messagesDiv.lastChild);
    appendMessage('assistant', 'Error: Could not reach the API.');
  }
  sendBtn.disabled = false;
  userInput.focus();
}

sendBtn.onclick = sendMessage;
userInput.addEventListener("keydown", function(e) {
  if (e.key === "Enter") sendMessage();
});
