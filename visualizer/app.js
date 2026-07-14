const canvas = document.getElementById('sim-canvas');
const ctx = canvas.getContext('2d');
const container = document.getElementById('canvas-container');

// Nodes positions (normalized 0 to 1)
const nodes = {
    PT: { x: 0.1, y: 0.1, label: 'Primary Transmitter', color: 'var(--pu-color)' },
    PR: { x: 0.9, y: 0.1, label: 'Primary Receiver', color: 'var(--pu-color)' },
    SU: { x: 0.1, y: 0.8, label: 'SU Source', color: 'var(--su-color)' },
    Relay: { x: 0.5, y: 0.6, label: 'Shared Relay', color: 'var(--su-color)' },
    SUD: { x: 0.9, y: 0.8, label: 'SU Destination', color: 'var(--su-color)' }
};

let traceData = [];
let currentIndex = 0;
let isPlaying = false;
let animationSpeed = 800; // ms per step (two slots)
let lastTimestamp = 0;
let currentSlot = 1; // 1 or 2

// Resize canvas
function resizeCanvas() {
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
    drawStatic();
}
window.addEventListener('resize', resizeCanvas);

// File loading
document.getElementById('btn-load').addEventListener('click', () => {
    document.getElementById('file-input').click();
});

document.getElementById('file-input').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (event) => {
        try {
            const lines = event.target.result.trim().split('\n');
            traceData = lines.map(line => JSON.parse(line));
            document.getElementById('status-text').innerText = `Loaded ${traceData.length} steps.`;
            currentIndex = 0;
            updateDashboard(traceData[0]);
            resizeCanvas();
        } catch (err) {
            document.getElementById('status-text').innerText = 'Error loading JSONL trace.';
        }
    };
    reader.readAsText(file);
});

// Controls
const btnPlay = document.getElementById('btn-play');
btnPlay.addEventListener('click', () => {
    if (!traceData.length) return;
    isPlaying = !isPlaying;
    btnPlay.innerText = isPlaying ? 'Pause' : 'Play';
    if (isPlaying) requestAnimationFrame(animationLoop);
});

document.getElementById('btn-prev').addEventListener('click', () => {
    if (currentIndex > 0) {
        currentIndex--;
        updateDashboard(traceData[currentIndex]);
        currentSlot = 1;
        drawStatic();
    }
});

document.getElementById('btn-next').addEventListener('click', () => {
    if (currentIndex < traceData.length - 1) {
        currentIndex++;
        updateDashboard(traceData[currentIndex]);
        currentSlot = 1;
        drawStatic();
    }
});

document.getElementById('speed').addEventListener('input', (e) => {
    animationSpeed = parseInt(e.target.value);
});

function updateDashboard(data) {
    document.getElementById('metric-ep').innerText = data.episode || 0;
    document.getElementById('metric-step').innerText = data.step || 0;
    document.getElementById('metric-reward').innerText = (data.reward || 0).toFixed(2);
    document.getElementById('metric-su-rate').innerText = (data.throughput || 0).toFixed(3);
    document.getElementById('metric-pu-outage').innerText = ((data.outage || 0) * 100).toFixed(2);
    document.getElementById('metric-alpha').innerText = (data.alpha || 0).toFixed(3);
}

function getNodePos(nodeKey) {
    return {
        x: nodes[nodeKey].x * canvas.width,
        y: nodes[nodeKey].y * canvas.height
    };
}

function drawStatic() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw links
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.setLineDash([5, 5]);
    ctx.lineWidth = 1;
    
    const links = [
        ['PT', 'PR'], ['PT', 'Relay'], ['SU', 'Relay'],
        ['Relay', 'PR'], ['Relay', 'SUD'], ['PT', 'SUD']
    ];
    
    links.forEach(([n1, n2]) => {
        const p1 = getNodePos(n1);
        const p2 = getNodePos(n2);
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();
    });
    
    // Draw nodes
    ctx.setLineDash([]);
    for (const key in nodes) {
        const pos = getNodePos(key);
        const color = getComputedStyle(document.documentElement).getPropertyValue(nodes[key].color).trim() || '#fff';
        
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, 15, 0, 2 * Math.PI);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.shadowBlur = 15;
        ctx.shadowColor = color;
        ctx.fill();
        ctx.shadowBlur = 0;
        
        ctx.fillStyle = '#fff';
        ctx.font = '12px Inter';
        ctx.textAlign = 'center';
        ctx.fillText(nodes[key].label, pos.x, pos.y + 30);
    }
}

function drawPacket(fromKey, toKey, colorVar, progress) {
    const p1 = getNodePos(fromKey);
    const p2 = getNodePos(toKey);
    
    const x = p1.x + (p2.x - p1.x) * progress;
    const y = p1.y + (p2.y - p1.y) * progress;
    
    const color = getComputedStyle(document.documentElement).getPropertyValue(colorVar).trim();
    
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.shadowBlur = 10;
    ctx.shadowColor = color;
    ctx.fill();
    ctx.shadowBlur = 0;
}

function animationLoop(timestamp) {
    if (!isPlaying) return;
    if (!lastTimestamp) lastTimestamp = timestamp;
    
    const elapsed = timestamp - lastTimestamp;
    const slotDuration = animationSpeed / 2;
    
    drawStatic();
    
    const indicator = document.getElementById('timeslot-indicator');
    
    if (elapsed < slotDuration) {
        currentSlot = 1;
        const progress = elapsed / slotDuration;
        indicator.innerText = "Slot 1: SU → Relay, PT → PR & Relay";
        
        // Slot 1 transmissions
        drawPacket('SU', 'Relay', '--su-color', progress);
        drawPacket('PT', 'PR', '--pu-color', progress);
        drawPacket('PT', 'Relay', '--pu-color', progress);
        
    } else if (elapsed < animationSpeed) {
        currentSlot = 2;
        const progress = (elapsed - slotDuration) / slotDuration;
        indicator.innerText = "Slot 2: Relay → SUD & PR (Cooperative), PT → PR";
        
        // Slot 2 transmissions
        // Relay splits power, sends both PU and SU
        drawPacket('Relay', 'PR', '--pu-color', progress); // Forwarding PU
        drawPacket('Relay', 'SUD', '--su-color', progress); // Forwarding SU
        drawPacket('PT', 'PR', '--pu-color', progress); // PT new transmission
        
    } else {
        // Step complete, move to next step
        lastTimestamp = timestamp;
        currentIndex++;
        
        if (currentIndex >= traceData.length) {
            isPlaying = false;
            btnPlay.innerText = 'Play';
            currentIndex = traceData.length - 1;
        } else {
            updateDashboard(traceData[currentIndex]);
        }
    }
    
    if (isPlaying) {
        requestAnimationFrame(animationLoop);
    }
}

// Initial draw
setTimeout(resizeCanvas, 100);
