const API_BASE = window.location.origin;

// DOM Elements
const taskSelect = document.getElementById('task-select');
const btnReset = document.getElementById('btn-reset');
const systemTime = document.getElementById('system-time');
const rewardScore = document.getElementById('reward-score');
const emailList = document.getElementById('email-list');
const taskListContainer = document.getElementById('task-list');
const emailCount = document.getElementById('email-count');
const taskCount = document.getElementById('task-count');
const actionLog = document.getElementById('action-log');

// IMAP Elements
const imapForm = document.getElementById('imap-form');
const imapStatus = document.getElementById('imap-status');
const btnIngest = document.getElementById('btn-ingest');

let currentState = null;

// Helper: Log message
function logMsg(msg, isError = false) {
    const div = document.createElement('div');
    div.textContent = `> ${msg}`;
    if (isError) div.classList.add('text-red-400');
    actionLog.appendChild(div);
    actionLog.scrollTop = actionLog.scrollHeight;
}

// Render Emails
function renderEmails(emails) {
    emailList.innerHTML = '';
    emailCount.textContent = emails.length;

    if (emails.length === 0) {
        emailList.innerHTML = '<div class="text-center text-gray-400 py-10 text-sm">No emails</div>';
        return;
    }

    emails.forEach(e => {
        let statusBadge = '';
        if (e.is_archived) statusBadge = '<span class="px-2 py-0.5 rounded text-[10px] bg-gray-200 text-gray-600 font-bold ml-2">ARCHIVED</span>';
        else if (e.is_read) statusBadge = '<span class="px-2 py-0.5 rounded text-[10px] bg-blue-100 text-blue-800 font-bold ml-2">READ</span>';
        else statusBadge = '<span class="px-2 py-0.5 rounded text-[10px] bg-yellow-100 text-yellow-800 font-bold ml-2">UNREAD</span>';

        const div = document.createElement('div');
        div.className = `p-3 rounded-lg border text-sm ${e.is_read ? 'bg-white border-gray-200 text-gray-600' : 'bg-blue-50 border-blue-200 font-medium text-gray-900'} ${e.is_archived ? 'opacity-50' : ''}`;

        div.innerHTML = `
            <div class="flex justify-between items-start mb-1">
                <div class="truncate pr-2 font-semibold">${e.sender}</div>
                <div class="text-xs text-gray-400 whitespace-nowrap">ID: ${e.id.substring(0,6)}</div>
            </div>
            <div class="mb-1 truncate text-gray-800">${e.subject} ${statusBadge}</div>
            <div class="text-xs text-gray-500 line-clamp-2">${e.body}</div>
        `;
        emailList.appendChild(div);
    });
}

// Render Tasks
function renderTasks(tasks) {
    taskListContainer.innerHTML = '';
    taskCount.textContent = tasks.length;

    if (tasks.length === 0) {
        taskListContainer.innerHTML = '<div class="text-center text-gray-400 py-10 text-sm">No tasks pending</div>';
        return;
    }

    tasks.forEach(t => {
        let prioColor = 'bg-gray-200 text-gray-800';
        if (t.priority === 1) prioColor = 'bg-red-100 text-red-800';
        else if (t.priority === 2) prioColor = 'bg-yellow-100 text-yellow-800';
        else if (t.priority === 3) prioColor = 'bg-green-100 text-green-800';

        const prioBadge = t.priority ? `<span class="px-2 py-0.5 rounded text-[10px] font-bold ${prioColor}">P${t.priority}</span>` : '';

        const div = document.createElement('div');
        div.className = 'p-3 rounded-lg border bg-white border-gray-200 text-sm flex flex-col shadow-sm';
        div.innerHTML = `
            <div class="flex justify-between items-center mb-1">
                <div class="font-semibold text-gray-800 flex items-center space-x-2">
                    <i class="fa-regular fa-square text-gray-400"></i>
                    <span>${t.title}</span>
                </div>
                ${prioBadge}
            </div>
            <div class="text-xs text-gray-500 pl-5">${t.description}</div>
            ${t.deadline ? `<div class="text-xs text-red-500 mt-2 pl-5"><i class="fa-regular fa-clock"></i> Due: ${t.deadline}</div>` : ''}
        `;
        taskListContainer.appendChild(div);
    });
}

// Update Dashboard
function updateDashboard(data, score = 0) {
    currentState = data;
    systemTime.innerHTML = `<i class="fa-regular fa-clock mr-1"></i> ${data.current_time}`;
    rewardScore.textContent = `Reward: ${score.toFixed(2)}`;

    renderEmails(data.emails);
    renderTasks(data.tasks);

    if (data.last_action_result) {
        logMsg(`[Env] ${data.last_action_result}`);
    }
}

// Fetch Initial State
async function fetchState() {
    try {
        const res = await fetch(`${API_BASE}/state`);
        const data = await res.json();
        updateDashboard(data);
    } catch (e) {
        logMsg("Failed to connect to backend API.", true);
    }
}

// Reset Environment
btnReset.addEventListener('click', async () => {
    btnReset.disabled = true;
    btnReset.innerHTML = '<i class="fa-solid fa-spinner fa-spin mr-2"></i>Resetting...';

    const taskName = taskSelect.value;
    try {
        const res = await fetch(`${API_BASE}/reset`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({task_name: taskName})
        });
        const data = await res.json();
        actionLog.innerHTML = '';
        logMsg(`Environment reset for task: ${taskName}`);
        updateDashboard(data);
    } catch (e) {
        logMsg("Failed to reset environment.", true);
    } finally {
        btnReset.disabled = false;
        btnReset.innerHTML = '<i class="fa-solid fa-rotate-right mr-2"></i>Reset Environment';
    }
});

// IMAP Ingestion
imapForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const user = document.getElementById('imap-user').value;
    const pass = document.getElementById('imap-pass').value;
    const server = document.getElementById('imap-server').value;

    btnIngest.disabled = true;
    btnIngest.innerHTML = '<i class="fa-solid fa-spinner fa-spin mr-2"></i>Connecting...';
    imapStatus.className = "text-xs mt-2 text-blue-600 block";
    imapStatus.textContent = "Connecting to mail server...";

    try {
        const res = await fetch(`${API_BASE}/ingest-real-emails`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ username: user, password: pass, server: server, count: 5 })
        });

        const data = await res.json();

        if (res.ok) {
            imapStatus.className = "text-xs mt-2 text-green-600 block";
            imapStatus.textContent = `Successfully ingested ${data.ingested_count} emails.`;
            logMsg(`Ingested real emails into the environment.`);
            // Fetch updated state
            fetchState();
        } else {
            imapStatus.className = "text-xs mt-2 text-red-600 block";
            imapStatus.textContent = `Error: ${data.detail}`;
            logMsg(`IMAP Error: ${data.detail}`, true);
        }
    } catch (err) {
        imapStatus.className = "text-xs mt-2 text-red-600 block";
        imapStatus.textContent = "Connection failed.";
        logMsg("Failed to reach API for IMAP ingestion.", true);
    } finally {
        btnIngest.disabled = false;
        btnIngest.innerHTML = '<i class="fa-solid fa-cloud-arrow-down mr-2"></i>Ingest Latest Mails';
    }
});

// Init
fetchState();
