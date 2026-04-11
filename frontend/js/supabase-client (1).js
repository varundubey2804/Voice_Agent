// =============================================
// SUPABASE CLIENT - SIMPLIFIED
// =============================================

// Replace with your Supabase credentials
const SUPABASE_URL = 'https://genrfjbmyitomplekptg.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdlbnJmamJteWl0b21wbGVrcHRnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Njg3NDU5NzksImV4cCI6MjA4NDMyMTk3OX0.K7imwFUcO3hQnEX4uTuV4Clij7-ZVEwPTicj8JawQFQ';

// Initialize Supabase safely
let supabase;
try {
    if (!window.supabase || typeof window.supabase.createClient !== 'function') {
        console.error('Supabase CDN not loaded. Make sure <script src="https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2"></script> is included BEFORE js/supabase-client.js');
    } else {
        supabase = window.supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
            auth: { persistSession: true }
        });
        // keep CDN global intact; also expose client if needed
        window.supabaseClient = supabase;
        console.log('✅ Supabase client initialized');
    }
} catch (initErr) {
    console.error('Error initializing Supabase client:', initErr);
}

// Helper to ensure supabase available
function ensureClient() {
    if (!supabase) throw new Error('Supabase client not initialized. Check console for earlier errors.');
}

// =============================================
// AUTHENTICATION
// =============================================
const SupabaseAuth = {
    
    // Sign up - works for both customer and agent
    async signUp(email, password, fullName, userType) {
        ensureClient();
        const { data, error } = await supabase.auth.signUp({
            email: email,
            password: password,
            options: {
                data: {
                    full_name: fullName,
                    user_type: userType  // 'customer' or 'agent'
                }
            }
        });
        
        if (error) throw error;
        return data;
    },
    
    // Sign in
    async signIn(email, password) {
        ensureClient();
        const { data, error } = await supabase.auth.signInWithPassword({
            email: email,
            password: password
        });
        
        if (error) throw error;
        return data;
    },
    
    // Sign out
    async signOut() {
        ensureClient();
        const { error } = await supabase.auth.signOut();
        if (error) throw error;
        localStorage.removeItem('veena_user');
        window.location.href = 'login.html';
    },
    
    // Get current user
    async getCurrentUser() {
        ensureClient();
        const { data: { user } } = await supabase.auth.getUser();
        return user;
    },
    
    // Get profile
    async getProfile(userId) {
        ensureClient();
        const { data, error } = await supabase
            .from('profiles')
            .select('*')
            .eq('id', userId)
            .single();
        
        if (error) throw error;
        return data;
    },
    
    // Check if authenticated
    async isAuthenticated() {
        ensureClient();
        const { data: { session } } = await supabase.auth.getSession();
        return !!session;
    }
};

// =============================================
// KNOWLEDGE BASE - SIMPLIFIED
// =============================================
const KnowledgeBase = {
    
    // Get all entries
    async getAll() {
        const { data, error } = await supabase
            .from('knowledge_base')
            .select(`
                *,
                creator:profiles!knowledge_base_created_by_fkey(full_name),
                updater:profiles!knowledge_base_updated_by_fkey(full_name)
            `)
            .order('updated_at', { ascending: false });
        
        if (error) throw error;
        return data;
    },
    
    // Get by category
    async getByCategory(category) {
        const { data, error } = await supabase
            .from('knowledge_base')
            .select('*')
            .eq('category', category)
            .order('updated_at', { ascending: false });
        
        if (error) throw error;
        return data;
    },
    
    // Search
    async search(query) {
        const { data, error } = await supabase
            .from('knowledge_base')
            .select('*')
            .or(`title.ilike.%${query}%,content.ilike.%${query}%`)
            .order('updated_at', { ascending: false });
        
        if (error) throw error;
        return data;
    },
    
    // Create new entry
    async create(title, category, content) {
        const user = await SupabaseAuth.getCurrentUser();
        
        const { data, error } = await supabase
            .from('knowledge_base')
            .insert({
                title: title,
                category: category,
                content: content,
                created_by: user.id,
                updated_by: user.id
            })
            .select()
            .single();
        
        if (error) throw error;
        return data;
    },
    
    // Update entry
    async update(id, title, category, content) {
        const user = await SupabaseAuth.getCurrentUser();
        
        const { data, error } = await supabase
            .from('knowledge_base')
            .update({
                title: title,
                category: category,
                content: content,
                updated_by: user.id
            })
            .eq('id', id)
            .select()
            .single();
        
        if (error) throw error;
        return data;
    },
    
    // Delete entry
    async delete(id) {
        const { error } = await supabase
            .from('knowledge_base')
            .delete()
            .eq('id', id);
        
        if (error) throw error;
    },
    
    // Get stats
    async getStats() {
        const { data } = await supabase
            .from('knowledge_base')
            .select('category');
        
        const total = data?.length || 0;
        const categories = [...new Set(data?.map(d => d.category) || [])];
        
        return { total, categoryCount: categories.length };
    }
};

// =============================================
// UTILITY
// =============================================
const Utils = {
    formatDate(dateString) {
        return new Date(dateString).toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    },
    
    formatRelativeTime(dateString) {
        const diff = Date.now() - new Date(dateString).getTime();
        const mins = Math.floor(diff / 60000);
        const hours = Math.floor(diff / 3600000);
        const days = Math.floor(diff / 86400000);
        
        if (mins < 1) return 'Just now';
        if (mins < 60) return `${mins} min ago`;
        if (hours < 24) return `${hours} hours ago`;
        if (days < 7) return `${days} days ago`;
        return this.formatDate(dateString);
    }
};

// =============================================
// EXPORT GLOBALLY (CRITICAL)
// =============================================
window.SupabaseAuth = SupabaseAuth;
window.KnowledgeBase = (typeof KnowledgeBase !== 'undefined') ? KnowledgeBase : {};
window.Utils = (typeof Utils !== 'undefined') ? Utils : {};
if (supabase) window.supabase = window.supabase || supabase;