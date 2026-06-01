// State
var state = {
    currentPatch: null,
    history: [],
    annotatorName: '',
    done: false,
    loading: true,
    leafContext: null,
    currentLeafStem: null
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

// Derive leaf stem from patch path
function getLeafStem(patchPath) {
    var filename = patchPath.split('/').pop();
    return filename.replace(/__r\d+_c\d+\.\w+$/, '');
}

// Fetch leaf context and update state
function fetchLeafContext(patch) {
    var stem = getLeafStem(patch.patch_path);
    var url = '/api/leaf-context/' + patch.split + '/' +
              encodeURIComponent(patch.class_name) + '/' +
              encodeURIComponent(stem);
    return fetch(url)
        .then(function(res) {
            if (!res.ok) throw new Error('Leaf context fetch failed');
            return res.json();
        })
        .then(function(data) {
            state.leafContext = data;
            return data;
        });
}

// Build center panel HTML
function buildCenterPanelHTML(p) {
    var imgSrc = '/image/' + p.patch_path;
    var pct = ((p.index + 1) / p.total * 100).toFixed(1);
    var disabledAttr = state.history.length === 0 ? 'disabled' : '';

    return '<div class="class-info">' +
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
    '</div>' +
    '<div class="patch-strip" id="patchStrip"></div>';
}

// Render current patch — three-panel layout
function renderPatch() {
    if (!state.currentPatch) return;
    var p = state.currentPatch;

    progressTextEl.textContent = p.annotated_count + ' annotated | ' + (p.index + 1) + ' / ' + p.total;

    var newStem = getLeafStem(p.patch_path);

    if (newStem === state.currentLeafStem && state.leafContext) {
        // Same leaf — only update center panel + grid overlay
        var center = document.getElementById('panelCenter');
        if (center) {
            center.innerHTML = buildCenterPanelHTML(p);
        }
        var ctx = state.leafContext;
        ctx.patches.forEach(function(pp) {
            pp.is_current = (pp.patch_path === p.patch_path);
        });
        renderRightPanel(ctx);
        renderPatchStrip(ctx);
    } else {
        // Different leaf or first load — rebuild everything
        state.currentLeafStem = newStem;

        mainContent.innerHTML =
            '<div class="panel panel-left" id="panelLeft">' +
                '<div class="panel-header">Full Leaf</div>' +
                '<div id="leafImageWrap"><div class="loading">Loading leaf...</div></div>' +
            '</div>' +
            '<div class="panel panel-center" id="panelCenter">' +
                buildCenterPanelHTML(p) +
            '</div>' +
            '<div class="panel panel-right" id="panelRight">' +
                '<div class="panel-header">Leaf Overview</div>' +
                '<div id="gridWrap"><div class="loading">Loading grid...</div></div>' +
                '<div class="leaf-stats" id="leafStats"></div>' +
            '</div>';

        fetchLeafContext(p).then(function(ctx) {
            renderLeftPanel(ctx);
            renderRightPanel(ctx);
            renderPatchStrip(ctx);
        }).catch(function(err) {
            console.error('Leaf context error:', err);
            document.getElementById('leafImageWrap').innerHTML =
                '<div class="loading">Could not load leaf context</div>';
            document.getElementById('gridWrap').innerHTML =
                '<div class="loading">Could not load grid</div>';
        });
    }
}

// Render left panel — full leaf image
function renderLeftPanel(ctx) {
    var wrap = document.getElementById('leafImageWrap');
    if (!wrap) return;
    wrap.innerHTML = '<img src="' + escapeHtml(ctx.source_image_url) + '" alt="Full leaf" onerror="handleImageError(this)">';
}

// Render right panel — leaf with canvas grid overlay
function renderRightPanel(ctx) {
    var wrap = document.getElementById('gridWrap');
    var statsEl = document.getElementById('leafStats');
    if (!wrap) return;

    wrap.innerHTML =
        '<div class="grid-wrapper">' +
            '<img id="gridLeafImg" src="' + escapeHtml(ctx.source_image_url) + '" alt="Leaf grid" onerror="handleImageError(this)">' +
            '<canvas id="gridCanvas"></canvas>' +
        '</div>';

    if (statsEl) {
        statsEl.textContent = 'Leaf: ' + ctx.annotated_count + ' / ' + ctx.total_patches + ' patches annotated';
    }

    var img = document.getElementById('gridLeafImg');
    var canvas = document.getElementById('gridCanvas');

    function drawGrid() {
        var w = img.clientWidth;
        var h = img.clientHeight;
        if (w === 0 || h === 0) return;

        canvas.width = w;
        canvas.height = h;
        var c = canvas.getContext('2d');
        c.clearRect(0, 0, w, h);

        // Scale from actual image pixels to displayed size
        var scaleX = w / ctx.img_width;
        var scaleY = h / ctx.img_height;
        var cellW = 64 * scaleX;
        var cellH = 64 * scaleY;

        ctx.patches.forEach(function(patch) {
            var x = patch.col * cellW;
            var y = patch.row * cellH;

            // Fill annotated patches
            if (patch.label === 'healthy') {
                c.fillStyle = 'rgba(46, 204, 113, 0.35)';
                c.fillRect(x, y, cellW, cellH);
            } else if (patch.label === 'unhealthy') {
                c.fillStyle = 'rgba(231, 76, 60, 0.35)';
                c.fillRect(x, y, cellW, cellH);
            } else if (patch.label === 'skipped') {
                c.fillStyle = 'rgba(149, 165, 166, 0.25)';
                c.fillRect(x, y, cellW, cellH);
            }

            // Highlight current patch
            if (patch.is_current) {
                c.strokeStyle = '#00d4ff';
                c.lineWidth = 3;
                c.strokeRect(x + 1, y + 1, cellW - 2, cellH - 2);
            }
        });

        // Draw grid lines
        c.strokeStyle = 'rgba(255, 255, 255, 0.2)';
        c.lineWidth = 1;
        for (var r = 0; r <= ctx.grid_rows; r++) {
            c.beginPath();
            c.moveTo(0, r * cellH);
            c.lineTo(w, r * cellH);
            c.stroke();
        }
        for (var col = 0; col <= ctx.grid_cols; col++) {
            c.beginPath();
            c.moveTo(col * cellW, 0);
            c.lineTo(col * cellW, h);
            c.stroke();
        }
    }

    // Draw once image loads
    if (img.complete && img.naturalWidth > 0) {
        drawGrid();
    } else {
        img.addEventListener('load', drawGrid);
    }

    // Redraw on resize
    var observer = new ResizeObserver(function() { drawGrid(); });
    observer.observe(wrap);

    // Click handler — navigate to clicked patch
    canvas.addEventListener('click', function(e) {
        var rect = canvas.getBoundingClientRect();
        var canvasScaleX = canvas.width / rect.width;
        var canvasScaleY = canvas.height / rect.height;
        var clickX = (e.clientX - rect.left) * canvasScaleX;
        var clickY = (e.clientY - rect.top) * canvasScaleY;

        var cellW = 64 * (canvas.width / ctx.img_width);
        var cellH = 64 * (canvas.height / ctx.img_height);
        var clickedCol = Math.floor(clickX / cellW);
        var clickedRow = Math.floor(clickY / cellH);

        // Find the patch at this position
        var target = null;
        for (var i = 0; i < ctx.patches.length; i++) {
            if (ctx.patches[i].row === clickedRow && ctx.patches[i].col === clickedCol) {
                target = ctx.patches[i];
                break;
            }
        }
        if (target) {
            jumpToPatch(target.patch_path);
        }
    });
}

// Render patch strip — all patches from current leaf
function renderPatchStrip(ctx) {
    var strip = document.getElementById('patchStrip');
    if (!strip) return;
    strip.innerHTML = '';

    ctx.patches.forEach(function(patch) {
        var img = document.createElement('img');
        img.className = 'strip-thumb';
        if (patch.is_current) img.classList.add('current');
        if (patch.label) img.setAttribute('data-label', patch.label);
        img.src = '/image/' + patch.patch_path;
        img.title = 'r' + patch.row + ' c' + patch.col +
                    (patch.label ? ' (' + patch.label + ')' : '');
        img.addEventListener('click', function() {
            jumpToPatch(patch.patch_path);
        });
        strip.appendChild(img);
    });

    // Auto-scroll to current patch
    var currentThumb = strip.querySelector('.strip-thumb.current');
    if (currentThumb) {
        currentThumb.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
    }
}

// Jump to a specific patch via grid/strip click
function jumpToPatch(patchPath) {
    if (state.loading) return;
    state.loading = true;

    fetch('/api/jump-to-patch?annotator=' + encodeURIComponent(state.annotatorName) +
          '&patch_path=' + encodeURIComponent(patchPath))
        .then(function(res) {
            if (!res.ok) throw new Error('Jump failed');
            return res.json();
        })
        .then(function(data) {
            state.currentPatch = data;
            state.loading = false;
            renderPatch();
        })
        .catch(function(err) {
            state.loading = false;
            console.error('Jump error:', err);
        });
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

// Update cached leaf context after an annotation action
function updateCachedLeafLabel(patchPath, label) {
    if (!state.leafContext) return;
    var ctx = state.leafContext;
    for (var i = 0; i < ctx.patches.length; i++) {
        if (ctx.patches[i].patch_path === patchPath) {
            if (!ctx.patches[i].label) ctx.annotated_count++;
            ctx.patches[i].label = label;
            break;
        }
    }
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
        var annotatedPath = state.currentPatch.patch_path;
        state.history.push({
            patch_path: annotatedPath,
            class_name: state.currentPatch.class_name,
            label: label
        });
        updateCachedLeafLabel(annotatedPath, label);

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
        updateCachedLeafLabel(state.currentPatch.patch_path, 'skipped');
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
            // Clear cached leaf context so labels get re-fetched
            state.leafContext = null;
            state.currentLeafStem = null;
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
    mainContent.className = 'main single-panel';
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
