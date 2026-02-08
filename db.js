const DB_NAME = 'NeuralChatDB';
const DB_VERSION = 1;

const dbPromise = new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onupgradeneeded = (event) => {
        const db = event.target.result;
        // Chats Store
        if (!db.objectStoreNames.contains('chats')) {
            db.createObjectStore('chats', { keyPath: 'id' });
        }
        // Messages Store
        if (!db.objectStoreNames.contains('messages')) {
            const msgStore = db.createObjectStore('messages', { keyPath: 'id', autoIncrement: true });
            msgStore.createIndex('chat_id', 'chat_id', { unique: false });
        }
    };

    request.onsuccess = (event) => resolve(event.target.result);
    request.onerror = (event) => reject(event.target.error);
});

const DB = {
    async createChat(title) {
        const db = await dbPromise;
        const chat = { 
            id: Date.now().toString(), 
            title: title || 'Novo Chat', 
            created: new Date(), 
            updated: new Date() 
        };
        const tx = db.transaction('chats', 'readwrite');
        tx.objectStore('chats').add(chat);
        return chat;
    },

    async getChats() {
        const db = await dbPromise;
        return new Promise((resolve) => {
            const tx = db.transaction('chats', 'readonly');
            const req = tx.objectStore('chats').getAll();
            req.onsuccess = () => resolve(req.result.sort((a,b) => b.updated - a.updated));
        });
    },

    async addMessage(chatId, role, content, type = 'text', meta = {}) {
        const db = await dbPromise;
        const msg = {
            chat_id: chatId,
            role, // 'user' or 'ai'
            content,
            type, // 'text', 'image', 'search'
            meta,
            timestamp: new Date()
        };
        const tx = db.transaction(['messages', 'chats'], 'readwrite');
        tx.objectStore('messages').add(msg);
        
        // Atualiza timestamp do chat
        const chatStore = tx.objectStore('chats');
        const chatReq = chatStore.get(chatId);
        chatReq.onsuccess = () => {
            const chat = chatReq.result;
            if(chat) {
                chat.updated = new Date();
                chatStore.put(chat);
            }
        };
    },

    async getMessages(chatId) {
        const db = await dbPromise;
        return new Promise((resolve) => {
            const tx = db.transaction('messages', 'readonly');
            const index = tx.objectStore('messages').index('chat_id');
            const req = index.getAll(IDBKeyRange.only(chatId));
            req.onsuccess = () => resolve(req.result);
        });
    }
};