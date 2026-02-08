// Estado Global
let currentChatId = null;
let currentMode = 'text'; // text, image, search

// Inicialização
document.addEventListener('DOMContentLoaded', async () => {
    await textEngine.init();
    await imageEngine.init();
    loadSidebar();
    
    // Event Listeners
    document.getElementById('btn-send').addEventListener('click', handleSend);
    document.getElementById('btn-new-chat').addEventListener('click', createNewChat);
    document.getElementById('mode-selector').addEventListener('change', (e) => {
        currentMode = e.target.value;
        const ar = document.getElementById('aspect-ratio');
        ar.style.display = currentMode === 'image' ? 'block' : 'none';
        
        const txt = document.getElementById('user-input');
        if(currentMode === 'image') txt.placeholder = "Descreva a imagem...";
        else if(currentMode === 'search') txt.placeholder = "O que deseja pesquisar?";
        else txt.placeholder = "Digite sua mensagem...";
    });
    
    // Configurações
    const modal = document.getElementById('settings-modal');
    document.getElementById('btn-settings').addEventListener('click', () => modal.classList.remove('hidden'));
    document.getElementById('btn-close-settings').addEventListener('click', () => modal.classList.add('hidden'));
    
    // Slider Inputs
    document.getElementById('temp-slider').addEventListener('input', (e) => document.getElementById('temp-val').innerText = e.target.value);
    document.getElementById('steps-slider').addEventListener('input', (e) => document.getElementById('steps-val').innerText = e.target.value);
});

async function createNewChat() {
    const chat = await DB.createChat();
    currentChatId = chat.id;
    loadSidebar();
    document.getElementById('messages-container').innerHTML = '';
    document.getElementById('current-chat-title').innerText = 'Novo Chat';
}

async function loadSidebar() {
    const list = document.getElementById('history-list');
    list.innerHTML = '';
    const chats = await DB.getChats();
    
    chats.forEach(chat => {
        const div = document.createElement('div');
        div.className = 'history-item';
        div.innerText = chat.title;
        div.onclick = () => loadChat(chat.id);
        list.appendChild(div);
    });

    // Check Storage
    if(navigator.storage && navigator.storage.estimate) {
        const est = await navigator.storage.estimate();
        const percent = ((est.usage / est.quota) * 100).toFixed(1);
        document.getElementById('storage-indicator').innerText = `Storage: ${percent}%`;
    }
}

async function loadChat(id) {
    currentChatId = id;
    const msgs = await DB.getMessages(id);
    const container = document.getElementById('messages-container');
    container.innerHTML = '';
    
    msgs.forEach(msg => {
        if (msg.type === 'image') renderImageMessage(msg.role, msg.content, msg.meta);
        else renderMessage(msg.role, msg.content, msg.type);
    });
}

async function handleSend() {
    const input = document.getElementById('user-input');
    const text = input.value.trim();
    if (!text) return;

    if (!currentChatId) await createNewChat();

    // 1. Render User Message
    renderMessage('user', text);
    await DB.addMessage(currentChatId, 'user', text);
    input.value = '';

    const loadingId = renderLoading();

    try {
        if (currentMode === 'text') {
            const temp = parseFloat(document.getElementById('temp-slider').value);
            const response = await textEngine.generateResponse(text, temp);
            removeLoading(loadingId);
            renderMessage('ai', response);
            await DB.addMessage(currentChatId, 'ai', response);

        } else if (currentMode === 'search') {
            const results = await searchEngine.search(text);
            removeLoading(loadingId);
            
            // Formatando resposta da busca
            let content = `🔍 **Resultados para: "${text}"**

`;
            results.forEach((r, i) => {
                content += `**${i+1}. [${r.source}] ${r.title}**
${r.snippet}
[Link](${r.url})

`;
            });
            
            renderMessage('ai', content, 'search');
            await DB.addMessage(currentChatId, 'ai', content, 'search');

        } else if (currentMode === 'image') {
            const ratio = document.getElementById('aspect-ratio').value;
            let width = 512, height = 512;
            if(ratio === '16:9') { width = 640; height = 360; }
            if(ratio === '9:16') { width = 360; height = 640; }
            
            const steps = parseInt(document.getElementById('steps-slider').value);
            
            // Generate
            const tensor = await imageEngine.generateImage(text, steps, width, height, (progress) => {
                document.getElementById(loadingId).innerText = `Gerando Imagem (Rectified Flow): ${Math.round(progress * 100)}%`;
            });

            removeLoading(loadingId);
            
            // Render Canvas & Save
            const canvas = document.createElement('canvas');
            canvas.width = width;
            canvas.height = height;
            await tf.browser.toPixels(tensor, canvas);
            
            const dataUrl = canvas.toDataURL('image/png');
            renderImageMessage('ai', dataUrl, { prompt: text, steps });
            await DB.addMessage(currentChatId, 'ai', dataUrl, 'image', { prompt: text });
            
            tensor.dispose();
        }
    } catch (e) {
        removeLoading(loadingId);
        renderMessage('ai', 'Erro no processamento: ' + e.message);
    }
}

function renderMessage(role, text, type = 'text') {
    const container = document.getElementById('messages-container');
    const div = document.createElement('div');
    div.className = `message ${role}`;
    
    // Simple Markdown Parser (Bold and Link)
    let formatted = text
        .replace(/**(.*?)**/g, '<b>$1</b>')
        .replace(/[Link]((.*?))/g, '<a href="$1" target="_blank" style="color: lightblue">Fonte 🔗</a>');

    div.innerHTML = formatted;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

function renderImageMessage(role, dataUrl, meta) {
    const container = document.getElementById('messages-container');
    const div = document.createElement('div');
    div.className = `message ${role}`;
    
    const img = document.createElement('img');
    img.src = dataUrl;
    
    const btn = document.createElement('button');
    btn.innerText = "💾 Baixar PNG";
    btn.style.marginTop = "10px";
    btn.style.padding = "5px 10px";
    btn.style.cursor = "pointer";
    btn.onclick = () => saveAs(dataUrl, `neural_img_${Date.now()}.png`);

    const info = document.createElement('div');
    info.style.fontSize = "0.8em";
    info.style.color = "#aaa";
    info.innerText = `Prompt: ${meta.prompt} | ${new Date().toLocaleTimeString()}`;

    div.appendChild(img);
    div.appendChild(document.createElement('br'));
    div.appendChild(info);
    div.appendChild(document.createElement('br'));
    div.appendChild(btn);
    
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

function renderLoading() {
    const id = 'loading-' + Date.now();
    const container = document.getElementById('messages-container');
    const div = document.createElement('div');
    div.className = 'message ai';
    div.id = id;
    div.innerText = 'Pensando...';
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
    return id;
}

function removeLoading(id) {
    const el = document.getElementById(id);
    if(el) el.remove();
}

// No DOMContentLoaded, adicione:
document.getElementById('mode-selector').innerHTML += 
    '<option value="video">🎥 Vídeo 3D (Text-to-Video)</option>';

// Na função handleSend(), adicione:
} else if (currentMode === 'video') {
    try {
        // 1. Gera código TSX
        const tsxCode = await videoEngine.generateTSXCode(text, (msg) => {
            document.getElementById(loadingId).innerText = msg;
        });

        // 2. Renderiza vídeo
        const videoData = await videoEngine.renderTSXToVideo(tsxCode, (msg) => {
            document.getElementById(loadingId).innerText = msg;
        });

        removeLoading(loadingId);

        // 3. Exibe vídeo original
        const videoMsg = renderVideoMessage('ai', videoData.url, videoData.metadata, false);
        await DB.addMessage(currentChatId, 'ai', videoData.url, 'video', { 
            tsx: tsxCode,
            ...videoData.metadata
        });

        // 4. Botão de Upscale
        const upscaleBtn = document.createElement('button');
        upscaleBtn.innerText = '✨ Increase Quality (Neural Upscale)';
        upscaleBtn.style.marginTop = '10px';
        upscaleBtn.onclick = async () => {
            upscaleBtn.disabled = true;
            upscaleBtn.innerText = 'Processando...';

            const hdLoadingId = renderLoading();
            
            const hdVideo = await videoEngine.upscaleVideo(
                videoData.blob, 
                videoData.metadata,
                (msg) => document.getElementById(hdLoadingId).innerText = msg
            );

            removeLoading(hdLoadingId);
            renderVideoMessage('ai', hdVideo.url, hdVideo.metadata, true);
        };

        videoMsg.appendChild(upscaleBtn);

    } catch (e) {
        removeLoading(loadingId);
        renderMessage('ai', `❌ Erro na geração: ${e.message}`);
    }
}

// Nova função para renderizar vídeos
function renderVideoMessage(role, videoUrl, metadata, isHD) {
    const container = document.getElementById('messages-container');
    const div = document.createElement('div');
    div.className = `message ${role}`;

    const video = document.createElement('video');
    video.src = videoUrl;
    video.controls = true;
    video.style.maxWidth = '100%';
    video.style.borderRadius = '8px';

    const info = document.createElement('div');
    info.style.fontSize = '0.8em';
    info.style.color = '#aaa';
    info.innerText = `${isHD ? '🎬 HD' : '📹 Original'} | ${metadata.width}x${metadata.height} | ${metadata.fps}fps`;

    const downloadBtn = document.createElement('button');
    downloadBtn.innerText = '💾 Download MP4';
    downloadBtn.style.marginTop = '10px';
    downloadBtn.onclick = () => {
        const a = document.createElement('a');
        a.href = videoUrl;
        a.download = `video_${isHD ? 'hd' : 'standard'}_${Date.now()}.mp4`;
        a.click();
    };

    div.appendChild(video);
    div.appendChild(document.createElement('br'));
    div.appendChild(info);
    div.appendChild(document.createElement('br'));
    div.appendChild(downloadBtn);

    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
    
    return div;
}