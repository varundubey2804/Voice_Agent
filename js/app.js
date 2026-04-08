// Dummy Data
const currentUser = {
  id: 1,
  name: "Jane Doe",
  headline: "Senior Frontend Engineer | UI/UX Designer",
  company: "TechCorp",
  avatar: "https://i.pravatar.cc/150?u=jane",
  connections: 500,
  views: 120
};

const postsData = [
  {
    id: 1,
    author: {
      name: "John Smith",
      headline: "Product Manager at StartupInc",
      avatar: "https://i.pravatar.cc/150?u=john",
    },
    time: "2h",
    content: "Just launched our new product feature! Extremely proud of the team's hard work over the past few months. 🚀",
    image: "https://images.unsplash.com/photo-1460925895917-afdab827c52f?auto=format&fit=crop&w=800&q=80",
    likes: 124,
    comments: 12,
    connectionLevel: 1 // 1st degree
  },
  {
    id: 2,
    author: {
      name: "Alice Johnson",
      headline: "UX Researcher",
      avatar: "https://i.pravatar.cc/150?u=alice",
    },
    time: "5h",
    content: "Interesting read on the future of design systems and how they are evolving with AI. What are your thoughts?",
    image: null,
    likes: 45,
    comments: 8,
    connectionLevel: 2 // 2nd degree
  },
  {
    id: 3,
    author: {
      name: "Bob Williams",
      headline: "Software Engineer at BigTech",
      avatar: "https://i.pravatar.cc/150?u=bob",
    },
    time: "1d",
    content: "Happy to announce that I've started a new position as Software Engineer! 🎉",
    image: null,
    likes: 342,
    comments: 56,
    connectionLevel: 1
  }
];

const suggestionsData = [
  { id: 4, name: "Charlie Davis", headline: "Data Scientist", avatar: "https://i.pravatar.cc/150?u=charlie" },
  { id: 5, name: "Diana Prince", headline: "Marketing Director", avatar: "https://i.pravatar.cc/150?u=diana" },
  { id: 6, name: "Evan Wright", headline: "Founder & CEO", avatar: "https://i.pravatar.cc/150?u=evan" }
];

const notificationsData = [
  { id: 1, user: { name: "Alice Johnson", avatar: "https://i.pravatar.cc/150?u=alice" }, action: "viewed your profile", time: "2h", unread: true },
  { id: 2, user: { name: "Bob Williams", avatar: "https://i.pravatar.cc/150?u=bob" }, action: "liked your post", time: "5h", unread: true },
  { id: 3, user: { name: "Charlie Davis", avatar: "https://i.pravatar.cc/150?u=charlie" }, action: "sent you a connection request", time: "1d", unread: false }
];

const connectionsData = [
  { id: 7, name: "Grace Lee", headline: "Product Designer", avatar: "https://i.pravatar.cc/150?u=grace" },
  { id: 8, name: "Henry Ford", headline: "Backend Developer", avatar: "https://i.pravatar.cc/150?u=henry" },
  { id: 9, name: "Ivy Chen", headline: "HR Manager", avatar: "https://i.pravatar.cc/150?u=ivy" },
  { id: 10, name: "Jack Wilson", headline: "Sales Lead", avatar: "https://i.pravatar.cc/150?u=jack" }
];

// App State
let state = {
  theme: localStorage.getItem('theme') || 'light',
  posts: [...postsData]
};

// --- DOM Elements ---
const themeToggleBtn = document.getElementById('themeToggle');
const postsContainer = document.getElementById('postsContainer');
const suggestionsContainer = document.getElementById('suggestionsContainer');
const notificationsContainer = document.getElementById('notificationsContainer');
const connectionsContainer = document.getElementById('connectionsContainer');

const createPostBtn = document.getElementById('createPostBtn');
const postModalOverlay = document.getElementById('postModalOverlay');
const closeModalBtn = document.getElementById('closeModalBtn');
const submitPostBtn = document.getElementById('submitPostBtn');
const postTextarea = document.getElementById('postTextarea');

// --- Initialization ---
function init() {
  applyTheme(state.theme);

  if (themeToggleBtn) {
    themeToggleBtn.addEventListener('click', toggleTheme);
  }

  // Render content based on which container is present (which page we are on)
  if (postsContainer) renderPosts(state.posts);
  if (suggestionsContainer) renderSuggestions();
  if (notificationsContainer) renderNotifications();
  if (connectionsContainer) renderConnections();

  // Modal logic
  if (createPostBtn && postModalOverlay) {
    createPostBtn.addEventListener('click', () => {
      postModalOverlay.classList.add('active');
      postTextarea.focus();
    });

    closeModalBtn.addEventListener('click', () => {
      postModalOverlay.classList.remove('active');
    });

    postModalOverlay.addEventListener('click', (e) => {
      if (e.target === postModalOverlay) {
        postModalOverlay.classList.remove('active');
      }
    });

    if (submitPostBtn) {
      submitPostBtn.addEventListener('click', handleCreatePost);
    }
  }

  // Setup toast container if it doesn't exist
  if (!document.getElementById('toastContainer')) {
    const tc = document.createElement('div');
    tc.id = 'toastContainer';
    tc.className = 'toast-container';
    document.body.appendChild(tc);
  }
}

// --- Theme Logic ---
function toggleTheme() {
  state.theme = state.theme === 'light' ? 'dark' : 'light';
  localStorage.setItem('theme', state.theme);
  applyTheme(state.theme);
}

function applyTheme(theme) {
  document.body.setAttribute('data-theme', theme);
  if (themeToggleBtn) {
    themeToggleBtn.innerHTML = theme === 'dark' ? '<i class="fas fa-sun"></i>' : '<i class="fas fa-moon"></i>';
  }
}

// --- Rendering Functions ---
function renderPosts(posts) {
  if (!postsContainer) return;

  // Simulate loading
  postsContainer.innerHTML = Array(2).fill(getPostSkeleton()).join('');

  setTimeout(() => {
    postsContainer.innerHTML = '';
    posts.forEach(post => {
      const postEl = document.createElement('div');
      postEl.className = 'card';
      postEl.innerHTML = `
        <div class="card-body">
          <div class="post-header">
            <img src="${post.author.avatar}" alt="${post.author.name}" class="post-avatar">
            <div class="post-meta">
              <h4>${post.author.name} <span class="text-secondary text-xs">• ${post.connectionLevel === 1 ? '1st' : '2nd'}</span></h4>
              <p>${post.author.headline}</p>
              <p class="text-xs">${post.time}</p>
            </div>
          </div>
          <div class="post-content">
            <p>${post.content}</p>
          </div>
          ${post.image ? `<img src="${post.image}" alt="Post image" class="post-image">` : ''}
          <div class="post-stats">
            <span><i class="fas fa-thumbs-up"></i> ${post.likes}</span>
            <span>${post.comments} comments</span>
          </div>
          <div class="post-actions mt-2">
            <button class="action-btn" onclick="toggleLike(this, ${post.id})"><i class="far fa-thumbs-up"></i> Like</button>
            <button class="action-btn"><i class="far fa-comment"></i> Comment</button>
            <button class="action-btn"><i class="fas fa-share"></i> Share</button>
            <button class="action-btn"><i class="fas fa-paper-plane"></i> Send</button>
          </div>
        </div>
      `;
      postsContainer.appendChild(postEl);
    });
  }, 500); // Simulate network delay
}

function renderSuggestions() {
  if (!suggestionsContainer) return;

  suggestionsContainer.innerHTML = '';
  suggestionsData.forEach(user => {
    const userEl = document.createElement('div');
    userEl.className = 'suggested-user';
    userEl.innerHTML = `
      <img src="${user.avatar}" alt="${user.name}" class="suggested-avatar">
      <div class="suggested-info flex-col justify-center">
        <h5>${user.name}</h5>
        <p>${user.headline}</p>
        <button class="btn btn-outline text-xs" onclick="toggleConnect(this)">+ Connect</button>
      </div>
    `;
    suggestionsContainer.appendChild(userEl);
  });
}

function renderNotifications() {
  if (!notificationsContainer) return;

  notificationsContainer.innerHTML = '';
  notificationsData.forEach(notif => {
    const notifEl = document.createElement('div');
    notifEl.className = `notification-item ${notif.unread ? 'notification-unread' : ''}`;
    notifEl.innerHTML = `
      <img src="${notif.user.avatar}" alt="${notif.user.name}" class="notification-avatar">
      <div class="notification-content">
        <p><strong>${notif.user.name}</strong> ${notif.action}</p>
      </div>
      <div class="notification-time">${notif.time}</div>
    `;
    notificationsContainer.appendChild(notifEl);
  });
}

function renderConnections() {
  if (!connectionsContainer) return;

  connectionsContainer.innerHTML = '';
  connectionsData.forEach(conn => {
    const connEl = document.createElement('div');
    connEl.className = 'card connection-card';
    connEl.innerHTML = `
      <img src="${conn.avatar}" alt="${conn.name}" class="connection-avatar">
      <h5 class="mb-2">${conn.name}</h5>
      <p class="text-xs text-secondary mb-4">${conn.headline}</p>
      <button class="btn btn-outline w-full" onclick="showToast('Message sent to ${conn.name}')">Message</button>
    `;
    connectionsContainer.appendChild(connEl);
  });
}

// --- Interaction Logic ---

function toggleConnect(btn) {
  if (btn.innerText === '+ Connect') {
    btn.innerText = 'Pending';
    btn.className = 'btn btn-ghost text-xs';
    showToast('Connection request sent!');
  } else if (btn.innerText === 'Pending') {
    btn.innerText = '+ Connect';
    btn.className = 'btn btn-outline text-xs';
    showToast('Connection request withdrawn.');
  }
}

function toggleLike(btn, postId) {
  const icon = btn.querySelector('i');
  if (icon.classList.contains('far')) {
    icon.classList.remove('far');
    icon.classList.add('fas');
    btn.style.color = 'var(--primary-color)';
    showToast('You liked a post');
  } else {
    icon.classList.remove('fas');
    icon.classList.add('far');
    btn.style.color = 'var(--text-secondary)';
  }
}

function handleCreatePost() {
  const content = postTextarea.value.trim();
  if (!content) return;

  const newPost = {
    id: Date.now(),
    author: {
      name: currentUser.name,
      headline: currentUser.headline,
      avatar: currentUser.avatar
    },
    time: "Just now",
    content: content,
    image: null,
    likes: 0,
    comments: 0,
    connectionLevel: 0
  };

  state.posts.unshift(newPost);
  postModalOverlay.classList.remove('active');
  postTextarea.value = '';
  renderPosts(state.posts);
  showToast('Post published successfully!');
}

// --- Utility Functions ---

function showToast(message) {
  const container = document.getElementById('toastContainer');
  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.innerText = message;

  container.appendChild(toast);

  // Trigger reflow
  toast.offsetHeight;
  toast.classList.add('show');

  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => {
      toast.remove();
    }, 300); // Wait for transition
  }, 3000);
}

function getPostSkeleton() {
  return `
    <div class="card card-body mb-4">
      <div class="flex gap-3 mb-4">
        <div class="skeleton skeleton-avatar"></div>
        <div class="flex-col justify-center" style="flex:1;">
          <div class="skeleton skeleton-text" style="width: 40%"></div>
          <div class="skeleton skeleton-text" style="width: 25%"></div>
        </div>
      </div>
      <div class="skeleton skeleton-text"></div>
      <div class="skeleton skeleton-text"></div>
      <div class="skeleton skeleton-text" style="width: 60%"></div>
    </div>
  `;
}

// Simulate Form Submissions
function handleAuth(event, type) {
  event.preventDefault();
  const btn = event.submitter;
  const originalText = btn.innerText;

  btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading...';
  btn.disabled = true;

  setTimeout(() => {
    showToast(`${type} successful! Redirecting...`);
    setTimeout(() => {
      window.location.href = 'index.html';
    }, 1000);
  }, 1500);
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', init);
