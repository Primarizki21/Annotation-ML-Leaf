// State
var state = {
    currentPatch: null,
    history: [],
    annotatorName: '',
    done: false,
    loading: true
};

// DOM elements
var mainContent = document.getElementById('mainContent');
var setupModal = document.getElementById('setupModal');
var setupError = document.getElementById('setupError');
var nameInput = document.getElementById('nameInput');
var setupBtn = document.getElementById('setupBtn');
var flash = document.getElementById('flash');
var annotatorNameEl = document.getElementById('annotatorName');
var progressTextEl = document.getElementById('progressText');

// Initialize
function init() {
    fetch('/api/status')
        .then(function(res) { return res.json(); })
        .then(function(data) {
            if (data.setup) {
                state.annotatorName = data.name;
                annotatorNameEl.textContent = data.name;
                setupModal.classList.add('hidden');
                loadCurrentPatch();
            } else {
                setupModal.classList.remove('hidden');
            }
        })
        .catch(function(err) {
            console.error('Init error:', err);
            setupModal.classList.remove('hidden');
        });
}

// Setup
setupBtn.addEventListener('click', doSetup);
nameInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter') doSetup();
});

function doSetup() {
    var name = nameInput.value.trim();
    if (!name) {
        setupError.textContent = 'Please enter your name';
        setupError.style.display = 'block';
        return;
    }
    fetch('/api/setup', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({name: name})
    })
    .then(function(res) {
        if (!res.ok) {
            return res.json().then(function(err) {
                throw new Error(err.detail || 'Setup failed');
            });
        }
        return res.json();
    })
    .then(function(data) {
        state.annotatorName = data.name;
        annotatorNameEl.textContent = data.name;
        setupModal.classList.add('hidden');
        loadCurrentPatch();
    })
    .catch(function(err) {
        setupError.textContent = err.message || 'Connection error';
        setupError.style.display = 'block';
    });
}

// Load current patch
function loadCurrentPatch() {
    fetch('/api/patch/current')
        .then(function(res) { return res.json(); })
        .then(function(data) {
            if (data.done) {
                showDoneScreen();
                return;
            }
            state.currentPatch = data;
            state.loading = false;
            renderPatch();
        })
        .catch(function(err) {
            console.error('Load error:', err);
        });
}

// Render current patch
function renderPatch() {
    if (!state.currentPatch) return;
    var p = state.currentPatch;
    var imgSrc = '/image/' + p.patch_path;
    var pct = ((p.index + 1) / p.total * 100).toFixed(1);

    progressTextEl.textContent = p.annotated_count + ' annotated | ' + (p.index + 1) + ' / ' + p.total;

    var disabledAttr = state.history.length === 0 ? 'disabled' : '';

    mainContent.innerHTML =
        '<div class="class-info">' +
            '<div class="class-name">' + escapeHtml(p.class_name) + '</div>' +
            '<div>' + escapeHtml(p.split) + '</div>' +
        '</div>' +
        '<div class="image-container">' +
            '<img class="patch-image" src="' + imgSrc + '" alt="Patch" onerror="handleImageError(this)">' +
            '<div class="image-counter">' + (p.index + 1) + ' / ' + p.total + '</div>' +
        '</div>' +
        '<div class="progress-container">' +
            '<div class="progress-bar-bg">' +
                '<div class="progress-bar-fill" style="width: ' + pct + '%"></div>' +
            '</div>' +
            '<div class="progress-stats">' +
                '<span>' + p.annotated_count + ' annotated</span>' +
                '<span>' + pct + '%</span>' +
            '</div>' +
        '</div>' +
        '<div class="button-row">' +
            '<button class="btn btn-healthy" onclick="annotate(\'healthy\')">' +
                '&#10003; Healthy <span class="shortcut">H</span>' +
            '</button>' +
            '<button class="btn btn-unhealthy" onclick="annotate(\'unhealthy\')">' +
                '&#10007; Unhealthy <span class="shortcut">U</span>' +
            '</button>' +
            '<button class="btn btn-skip" onclick="skipPatch()">' +
                'Skip <span class="shortcut">S</span>' +
            '</button>' +
        '</div>' +
        '<div class="history-row">' +
            '<button class="btn-undo" onclick="undo()" ' + disabledAttr + '>' +
                '&#8592; Undo <span class="shortcut">ArrowLeft</span>' +
            '</button>' +
            '<div class="thumbnails" id="thumbnails"></div>' +
        '</div>';

    renderThumbnails();
}

// Handle missing image
function handleImageError(img) {
    img.src = 'data:image/svg+xml,' + encodeURIComponent(
        '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64">' +
        '<rect width="64" height="64" fill="#333"/>' +
        '<text x="32" y="30" text-anchor="middle" fill="#888" font-size="9">Missing</text>' +
        '<text x="32" y="42" text-anchor="middle" fill="#888" font-size="9">Image</text>' +
        '</svg>'
    );
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    var div = document.createElement('div');
    div.appendChild(document.createTextNode(text));
    return div.innerHTML;
}

// Render thumbnail strip
function renderThumbnails() {
    var container = document.getElementById('thumbnails');
    if (!container) return;
    container.innerHTML = '';
    var recent = state.history.slice(-5);
    recent.forEach(function(h) {
        var img = document.createElement('img');
        img.className = 'thumbnail ' + h.label;
        img.src = '/image/' + h.patch_path;
        img.title = h.class_name + ' - ' + h.label;
        container.appendChild(img);
    });
}

// Annotate
function annotate(label) {
    if (!state.currentPatch || state.loading) return;
    state.loading = true;

    fetch('/api/annotate', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            patch_path: state.currentPatch.patch_path,
            label: label
        })
    })
    .then(function(res) {
        if (!res.ok) throw new Error('Annotate failed');
        return res.json();
    })
    .then(function(data) {
        state.history.push({
            patch_path: state.currentPatch.patch_path,
            class_name: state.currentPatch.class_name,
            label: label
        });

        showFlash(label);

        if (data.done) {
            showDoneScreen();
        } else {
            state.currentPatch = data;
            state.loading = false;
            renderPatch();
        }
    })
    .catch(function(err) {
        state.loading = false;
        console.error('Annotate error:', err);
    });
}

// Skip
function skipPatch() {
    if (!state.currentPatch || state.loading) return;
    state.loading = true;

    fetch('/api/skip', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({patch_path: state.currentPatch.patch_path})
    })
    .then(function(res) {
        if (!res.ok) throw new Error('Skip failed');
        return res.json();
    })
    .then(function(data) {
        showFlash('skip');

        if (data.done) {
            showDoneScreen();
        } else {
            state.currentPatch = data;
            state.loading = false;
            renderPatch();
        }
    })
    .catch(function(err) {
        state.loading = false;
        console.error('Skip error:', err);
    });
}

// Undo
function undo() {
    if (state.history.length === 0 || state.loading) return;
    state.loading = true;

    fetch('/api/undo', {method: 'POST'})
        .then(function(res) {
            if (!res.ok) throw new Error('Undo failed');
            return res.json();
        })
        .then(function(data) {
            state.history.pop();
            state.currentPatch = data;
            state.loading = false;
            renderPatch();
        })
        .catch(function(err) {
            state.loading = false;
            console.error('Undo error:', err);
        });
}

// Show flash notification
function showFlash(type) {
    var labels = {healthy: 'Healthy', unhealthy: 'Unhealthy', skip: 'Skipped'};
    flash.textContent = labels[type] || type;
    flash.className = 'flash ' + type + ' show';
    setTimeout(function() {
        flash.classList.remove('show');
    }, 800);
}

// Show done screen
function showDoneScreen() {
    state.done = true;
    mainContent.innerHTML =
        '<div class="done-screen">' +
            '<h2>All Done!</h2>' +
            '<p>You have annotated all assigned patches.</p>' +
            '<p style="margin-top: 16px;">' +
                '<a href="/dashboard" class="btn-link" style="font-size: 16px;">View Dashboard</a>' +
            '</p>' +
        '</div>';
}

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (!setupModal.classList.contains('hidden')) return;

    switch(e.key.toLowerCase()) {
        case 'h':
            e.preventDefault();
            annotate('healthy');
            break;
        case 'u':
            e.preventDefault();
            annotate('unhealthy');
            break;
        case 's':
            e.preventDefault();
            skipPatch();
            break;
        case 'arrowleft':
            e.preventDefault();
            undo();
            break;
    }
});

// Start
init();
