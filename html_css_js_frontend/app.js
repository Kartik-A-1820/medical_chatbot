// ===== Configuration =====
const API_BASE = document.querySelector('meta[name="api-base-url"]')?.content || 'http://localhost:8000';
const SESSION_KEY = 'medassist-session-id';

// ===== State Management =====
let sessionId = localStorage.getItem(SESSION_KEY) || generateSessionId();
let messageCount = 0;
let isProcessing = false;

// ===== DOM Elements =====
const elements = {
  chatMessages: document.getElementById('chatMessages'),
  chatInput: document.getElementById('chatInput'),
  btnSend: document.getElementById('btnSend'),
  btnClearChat: document.getElementById('btnClearChat'),
  btnGeneratePDF: document.getElementById('btnGeneratePDF'),
  prescriptionModal: document.getElementById('prescriptionModal'),
  btnCloseModal: document.getElementById('btnCloseModal'),
  btnCancelPDF: document.getElementById('btnCancelPDF'),
  btnCreatePDF: document.getElementById('btnCreatePDF'),
  statusText: document.getElementById('statusText'),
  messageCountEl: document.getElementById('messageCount'),
  suggestedPrompts: document.getElementById('suggestedPrompts'),
  chatContainer: document.getElementById('chatContainer')
};

// ===== Utility Functions =====
function generateSessionId() {
  const id = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  localStorage.setItem(SESSION_KEY, id);
  return id;
}

function formatTime(date = new Date()) {
  return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
}

function updateStatus(text, isConnected = true) {
  if (elements.statusText) {
    elements.statusText.textContent = text;
    const statusDot = document.querySelector('.status-dot');
    if (statusDot) {
      statusDot.style.background = isConnected ? 'var(--success)' : 'var(--danger)';
    }
  }
}

function updateMessageCount() {
  messageCount++;
  if (elements.messageCountEl) {
    elements.messageCountEl.textContent = messageCount;
  }
}

function showToast(message, type = 'info') {
  const container = document.getElementById('toastContainer');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  
  const icons = {
    success: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
    error: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
    info: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>'
  };

  toast.innerHTML = `
    <div class="toast-icon">${icons[type] || icons.info}</div>
    <div class="toast-message">${message}</div>
  `;

  container.appendChild(toast);

  setTimeout(() => {
    toast.style.animation = 'toastSlideIn 0.3s ease reverse';
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

// ===== Chat Functions =====
function appendMessage(role, content, sources = null, isTyping = false) {
  const messageDiv = document.createElement('div');
  messageDiv.className = `message ${role}`;
  
  const avatar = document.createElement('div');
  avatar.className = 'message-avatar';
  avatar.textContent = role === 'assistant' ? 'AI' : 'U';
  
  const contentDiv = document.createElement('div');
  contentDiv.className = 'message-content';
  
  const bubble = document.createElement('div');
  bubble.className = 'message-bubble';
  
  if (isTyping) {
    bubble.innerHTML = `
      <div class="typing-indicator">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
      </div>
    `;
  } else {
    bubble.innerHTML = formatMessageContent(content);
    
    const timeDiv = document.createElement('div');
    timeDiv.className = 'message-time';
    timeDiv.textContent = formatTime();
    contentDiv.appendChild(bubble);
    contentDiv.appendChild(timeDiv);
    
    if (sources && sources.length > 0) {
      const referencesDiv = createReferencesSection(sources);
      contentDiv.appendChild(referencesDiv);
    }
  }
  
  if (isTyping) {
    contentDiv.appendChild(bubble);
  }
  
  messageDiv.appendChild(avatar);
  messageDiv.appendChild(contentDiv);
  
  elements.chatMessages.appendChild(messageDiv);
  scrollToBottom();
  
  if (!isTyping) {
    updateMessageCount();
  }
  
  return messageDiv;
}

function formatMessageContent(content) {
  content = content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  content = content.replace(/\n/g, '<br>');
  return content;
}

function createReferencesSection(sources) {
  const referencesDiv = document.createElement('div');
  referencesDiv.className = 'message-references';
  
  const toggle = document.createElement('button');
  toggle.className = 'references-toggle';
  toggle.innerHTML = `
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
      <polyline points="14 2 14 8 20 8"/>
    </svg>
    View ${sources.length} source${sources.length > 1 ? 's' : ''}
  `;
  
  const content = document.createElement('div');
  content.className = 'references-content hidden';
  
  sources.forEach((source, idx) => {
    const item = document.createElement('div');
    item.className = 'reference-item';
    item.textContent = `[${idx + 1}] ${source}`;
    content.appendChild(item);
  });
  
  toggle.addEventListener('click', () => {
    content.classList.toggle('hidden');
  });
  
  referencesDiv.appendChild(toggle);
  referencesDiv.appendChild(content);
  
  return referencesDiv;
}

function scrollToBottom() {
  if (elements.chatContainer) {
    elements.chatContainer.scrollTop = elements.chatContainer.scrollHeight;
  }
}

async function sendMessage(message) {
  if (!message.trim() || isProcessing) return;
  
  isProcessing = true;
  elements.btnSend.disabled = true;
  elements.chatInput.disabled = true;
  
  if (elements.suggestedPrompts && !elements.suggestedPrompts.classList.contains('hidden')) {
    elements.suggestedPrompts.classList.add('hidden');
  }
  
  appendMessage('user', message);
  elements.chatInput.value = '';
  autoResizeTextarea();
  
  const typingMessage = appendMessage('assistant', '', null, true);
  updateStatus('AI is thinking...', true);
  
  try {
    const response = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: message,
        session_id: sessionId
      })
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const data = await response.json();
    
    typingMessage.remove();
    
    if (data.answer) {
      const sources = Array.isArray(data.references) && data.references.length > 0
        ? data.references
        : null;
      appendMessage('assistant', data.answer, sources);
      
      if (data.triage_level) {
        const triageColors = {
          emergency: 'error',
          urgent: 'error',
          routine: 'info'
        };
        showToast(`Triage: ${data.triage_level.toUpperCase()}`, triageColors[data.triage_level] || 'info');
      }
      
      updateStatus('Connected', true);
    } else {
      throw new Error('No answer received from server');
    }
    
  } catch (error) {
    console.error('Chat error:', error);
    typingMessage.remove();
    appendMessage('assistant', `⚠️ Error: ${error.message}. Please try again or check your connection.`);
    updateStatus('Connection error', false);
    showToast('Failed to send message', 'error');
  } finally {
    isProcessing = false;
    elements.btnSend.disabled = false;
    elements.chatInput.disabled = false;
    elements.chatInput.focus();
  }
}

function clearChat() {
  if (confirm('Clear all messages? This cannot be undone.')) {
    elements.chatMessages.innerHTML = '';
    sessionId = generateSessionId();
    messageCount = 0;
    if (elements.messageCountEl) {
      elements.messageCountEl.textContent = '0';
    }
    if (elements.suggestedPrompts) {
      elements.suggestedPrompts.classList.remove('hidden');
    }
    showToast('Chat cleared', 'success');
  }
}

// ===== PDF Generation =====
function openPDFModal() {
  if (messageCount === 0) {
    showToast('Please have a conversation first before generating a prescription', 'info');
    return;
  }
  elements.prescriptionModal.classList.add('active');
}

function closePDFModal() {
  elements.prescriptionModal.classList.remove('active');
}

async function generatePDF() {
  const patientName = document.getElementById('patientName').value.trim();
  const patientAge = document.getElementById('patientAge').value.trim();
  const patientGender = document.getElementById('patientGender').value;
  const doctorName = document.getElementById('doctorName').value.trim();
  const clinicName = document.getElementById('clinicName').value.trim();
  
  if (!patientName) {
    showToast('Please enter patient name', 'error');
    return;
  }
  
  elements.btnCreatePDF.disabled = true;
  elements.btnCreatePDF.innerHTML = `
    <div class="typing-indicator">
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
    </div>
    Generating...
  `;
  
  try {
    const response = await fetch(`${API_BASE}/generate_prescription_pdf_from_chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        patient_name: patientName,
        patient_age: patientAge || 'N/A',
        patient_gender: patientGender || 'Not specified',
        doctor_name: doctorName || 'Dr. On-call',
        clinic_name: clinicName || 'Virtual Health'
      })
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}`);
    }
    
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `prescription_${patientName.replace(/\s+/g, '_')}_${Date.now()}.pdf`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
    
    showToast('Prescription PDF generated successfully!', 'success');
    closePDFModal();
    
  } catch (error) {
    console.error('PDF generation error:', error);
    showToast(`Failed to generate PDF: ${error.message}`, 'error');
  } finally {
    elements.btnCreatePDF.disabled = false;
    elements.btnCreatePDF.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
        <polyline points="14 2 14 8 20 8"/>
      </svg>
      Generate PDF
    `;
  }
}

// ===== Input Handling =====
function autoResizeTextarea() {
  const textarea = elements.chatInput;
  textarea.style.height = 'auto';
  textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
}

function handleKeyPress(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage(elements.chatInput.value);
  }
}

// ===== Event Listeners =====
elements.btnSend.addEventListener('click', () => sendMessage(elements.chatInput.value));
elements.chatInput.addEventListener('keydown', handleKeyPress);
elements.chatInput.addEventListener('input', autoResizeTextarea);
elements.btnClearChat.addEventListener('click', clearChat);
elements.btnGeneratePDF.addEventListener('click', openPDFModal);
elements.btnCloseModal.addEventListener('click', closePDFModal);
elements.btnCancelPDF.addEventListener('click', closePDFModal);
elements.btnCreatePDF.addEventListener('click', generatePDF);

document.querySelectorAll('.chip').forEach(chip => {
  chip.addEventListener('click', () => {
    const prompt = chip.dataset.prompt;
    elements.chatInput.value = prompt;
    autoResizeTextarea();
    elements.chatInput.focus();
  });
});

elements.prescriptionModal.querySelector('.modal-overlay').addEventListener('click', closePDFModal);

// ===== Initialization =====
document.addEventListener('DOMContentLoaded', () => {
  console.log('MedAssist AI initialized');
  console.log('Session ID:', sessionId);
  console.log('API Base:', API_BASE);
  
  elements.chatInput.focus();
  updateStatus('Connected', true);
  
  fetch(`${API_BASE}/`)
    .then(response => response.json())
    .then(data => {
      console.log('Backend status:', data);
      if (data.status === 'ok') {
        updateStatus('Connected', true);
        showToast('Connected to MedAssist AI', 'success');
      }
    })
    .catch(error => {
      console.error('Backend connection error:', error);
      updateStatus('Backend offline', false);
      showToast('Backend connection failed', 'error');
    });
});
