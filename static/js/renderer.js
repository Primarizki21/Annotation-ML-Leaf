/**
 * Renderer — All DOM manipulation for the annotation page.
 *
 * Reads from State, never mutates it directly.
 * No fetch() calls — receives data, produces HTML/canvas output.
 * User actions (clicks) are forwarded to App via the callbacks object.
 *
 * Methods:
 *   updateProgress(patch)           → header progress text
 *   updateCenterPanel(patch)        → refreshes center panel HTML + listeners
 *   buildFullLayout(patch)          → rebuilds three-panel layout
 *   renderLeftPanel(ctx)            → full leaf image
 *   renderRightPanel(ctx)           → canvas grid + stats + click nav
 *   renderPatchStrip(ctx)           → thumbnail strip at bottom
 *   showDoneScreen()                → "All Done!" view
 *   showFlash(type)                 → brief label notification
 *   handleLeafContextError()        → error state for leaf context
 *
 * Constants:
 *   PATCH_SIZE = 32
 *
 * @imports { escapeHtml, handleImageError, getLeafStem } from './utils.js'
 * @exports { Renderer }
 */

import { escapeHtml, handleImageError, getLeafStem } from './utils.js';

const PATCH_SIZE = 32;
const FLASH_LABELS = {healthy: 'Healthy', unhealthy: 'Unhealthy', skip: 'Skipped'};

export class Renderer {
    constructor(state, callbacks) {
        this.state = state;
        this.callbacks = callbacks;
        this._gridObserver = null;

        this.mainContent = document.getElementById('mainContent');
        this.flash = document.getElementById('flash');
        this.progressTextEl = document.getElementById('progressText');
        this.toastContainer = document.getElementById('toastContainer');
        this._toastQueue = [];

        // Modal refs (Phase 4)
        this.alIntroModal = document.getElementById('alIntroModal');
        this.helpModal = document.getElementById('helpModal');
        this.historyModal = document.getElementById('historyModal');
        this.historyList = document.getElementById('historyList');
        this.alIntroBtn = document.getElementById('alIntroBtn');
        this.helpCloseBtn = document.getElementById('helpCloseBtn');
        this.historyCloseBtn = document.getElementById('historyCloseBtn');
    }

    updateProgress(patch) {
        this.progressTextEl.textContent =
            patch.annotated_count + ' annotated | ' + (patch.index + 1) + ' / ' + patch.total;
    }

    // ── Center panel ──

    buildCenterPanelHTML(p) {
        var imgSrc = '/image/' + p.patch_path;
        var pct = ((p.index + 1) / p.total * 100).toFixed(1);
        var disabledAttr = this.state.history.length === 0 ? 'disabled' : '';

        return '<div class="class-info">' +
            '<div class="class-name">' + escapeHtml(p.class_name) + '</div>' +
            '<div>' + escapeHtml(p.split) + '</div>' +
        '</div>' +
        '<div class="image-container">' +
            '<img class="patch-image" src="' + imgSrc + '" alt="Patch">' +
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
            '<button class="btn btn-healthy" data-action="healthy">' +
                '&#10003; Healthy <span class="shortcut">H</span>' +
            '</button>' +
            '<button class="btn btn-unhealthy" data-action="unhealthy">' +
                '&#10007; Unhealthy <span class="shortcut">U</span>' +
            '</button>' +
            '<button class="btn btn-skip" data-action="skip">' +
                'Skip <span class="shortcut">S</span>' +
            '</button>' +
        '</div>' +
        '<div class="history-row">' +
            '<button class="btn-undo" data-action="undo" ' + disabledAttr + '>' +
                '&#8592; Undo <span class="shortcut">ArrowLeft</span>' +
            '</button>' +
        '</div>' +
        '<div class="patch-strip" id="patchStrip"></div>';
    }

    attachCenterPanelListeners() {
        var center = document.getElementById('panelCenter');
        if (!center) return;

        center.querySelector('[data-action="healthy"]')
            ?.addEventListener('click', () => this.callbacks.onAnnotate('healthy'));
        center.querySelector('[data-action="unhealthy"]')
            ?.addEventListener('click', () => this.callbacks.onAnnotate('unhealthy'));
        center.querySelector('[data-action="skip"]')
            ?.addEventListener('click', () => this.callbacks.onSkip());
        center.querySelector('[data-action="undo"]')
            ?.addEventListener('click', () => this.callbacks.onUndo());

        var patchImg = center.querySelector('.patch-image');
        if (patchImg) {
            patchImg.addEventListener('error', function() { handleImageError(patchImg); });
        }
    }

    updateCenterPanel(p) {
        var center = document.getElementById('panelCenter');
        if (center) {
            center.innerHTML = this.buildCenterPanelHTML(p);
        }
        this.attachCenterPanelListeners();
    }

    // ── Full layout ──

    buildFullLayout(p) {
        this.mainContent.innerHTML =
            '<div class="panel panel-left" id="panelLeft">' +
                '<div class="panel-header">Full Leaf</div>' +
                '<div id="leafImageWrap"><div class="loading">Loading leaf...</div></div>' +
            '</div>' +
            '<div class="panel panel-center" id="panelCenter">' +
                this.buildCenterPanelHTML(p) +
            '</div>' +
            '<div class="panel panel-right" id="panelRight">' +
                '<div class="panel-header">Leaf Overview</div>' +
                '<div id="gridWrap"><div class="loading">Loading grid...</div></div>' +
                '<div class="leaf-stats" id="leafStats"></div>' +
            '</div>';
        this.attachCenterPanelListeners();
    }

    // ── Left panel ──

    renderLeftPanel(ctx) {
        var wrap = document.getElementById('leafImageWrap');
        if (!wrap) return;

        var img = document.createElement('img');
        img.src = escapeHtml(ctx.source_image_url);
        img.alt = 'Full leaf';
        img.addEventListener('error', function() { handleImageError(img); });
        wrap.innerHTML = '';
        wrap.appendChild(img);
    }

    // ── Right panel ──

    renderRightPanel(ctx) {
        var wrap = document.getElementById('gridWrap');
        var statsEl = document.getElementById('leafStats');
        if (!wrap) return;

        wrap.innerHTML =
            '<div class="grid-wrapper">' +
                '<img id="gridLeafImg" src="' + escapeHtml(ctx.source_image_url) + '" alt="Leaf grid">' +
                '<canvas id="gridCanvas"></canvas>' +
            '</div>';

        if (statsEl) {
            statsEl.textContent = 'Leaf: ' + ctx.annotated_count + ' / ' + ctx.total_patches + ' patches annotated';
        }

        var img = document.getElementById('gridLeafImg');
        var canvas = document.getElementById('gridCanvas');

        img.addEventListener('error', function() { handleImageError(img); });

        var self = this;

        function drawGrid() {
            var w = img.clientWidth;
            var h = img.clientHeight;
            if (w === 0 || h === 0) return;

            canvas.width = w;
            canvas.height = h;
            canvas.style.width = w + 'px';
            canvas.style.height = h + 'px';
            var c = canvas.getContext('2d');
            c.clearRect(0, 0, w, h);

            var scaleX = w / ctx.img_width;
            var scaleY = h / ctx.img_height;
            var cellW = PATCH_SIZE * scaleX;
            var cellH = PATCH_SIZE * scaleY;

            ctx.patches.forEach(function(patch) {
                var x = patch.col * cellW;
                var y = patch.row * cellH;

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

                if (patch.is_current) {
                    c.strokeStyle = '#00d4ff';
                    c.lineWidth = 1;
                    c.strokeRect(x, y, cellW, cellH);
                    // c.strokeRect(x + 1, y +1, cellW - 2, cellH - 2);
                }
            });

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

        if (img.complete && img.naturalWidth > 0) {
            drawGrid();
        } else {
            img.addEventListener('load', drawGrid);
        }

        if (self._gridObserver) self._gridObserver.disconnect();
        self._gridObserver = new ResizeObserver(function() { drawGrid(); });
        self._gridObserver.observe(wrap);

        canvas.addEventListener('click', function(e) {
            var rect = canvas.getBoundingClientRect();
            var canvasScaleX = canvas.width / rect.width;
            var canvasScaleY = canvas.height / rect.height;
            var clickX = (e.clientX - rect.left) * canvasScaleX;
            var clickY = (e.clientY - rect.top) * canvasScaleY;

            var cellW = PATCH_SIZE * (canvas.width / ctx.img_width);
            var cellH = PATCH_SIZE * (canvas.height / ctx.img_height);
            var clickedCol = Math.floor(clickX / cellW);
            var clickedRow = Math.floor(clickY / cellH);

            var target = null;
            for (var i = 0; i < ctx.patches.length; i++) {
                if (ctx.patches[i].row === clickedRow && ctx.patches[i].col === clickedCol) {
                    target = ctx.patches[i];
                    break;
                }
            }
            if (target) {
                self.callbacks.onJumpToPatch(target.patch_path);
            }
        });
    }

    // ── Patch strip ──

    renderPatchStrip(ctx) {
        var strip = document.getElementById('patchStrip');
        if (!strip) return;
        strip.innerHTML = '';

        ctx.patches.forEach((patch) => {
            var img = document.createElement('img');
            img.className = 'strip-thumb';
            if (patch.is_current) img.classList.add('current');
            if (patch.label) img.setAttribute('data-label', patch.label);
            img.src = '/image/' + patch.patch_path;
            img.title = 'r' + patch.row + ' c' + patch.col +
                        (patch.label ? ' (' + patch.label + ')' : '');
            img.addEventListener('click', () => {
                this.callbacks.onJumpToPatch(patch.patch_path);
            });
            strip.appendChild(img);
        });

        var currentThumb = strip.querySelector('.strip-thumb.current');
        if (currentThumb) {
            currentThumb.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
        }
    }

    // ── Done screen ──

    showDoneScreen(mode) {
        var heading = (mode === 'review') ? 'Review Complete!' : 'All Done!';
        var message = (mode === 'review')
            ? 'You have finished reviewing all disputed patches.'
            : 'You have annotated all assigned patches.';
        this.mainContent.className = 'main single-panel';
        this.mainContent.innerHTML =
            '<div class="done-screen">' +
                '<h2>' + heading + '</h2>' +
                '<p>' + message + '</p>' +
                '<p style="margin-top: 16px;">' +
                    '<a href="/dashboard" class="btn-link" style="font-size: 16px;">View Dashboard</a>' +
                '</p>' +
            '</div>';
    }

    // ── Error states ──

    handleLeafContextError() {
        var leafWrap = document.getElementById('leafImageWrap');
        var gridWrap = document.getElementById('gridWrap');
        if (leafWrap) leafWrap.innerHTML = '<div class="loading">Could not load leaf context</div>';
        if (gridWrap) gridWrap.innerHTML = '<div class="loading">Could not load grid</div>';
    }

    // ── Active Learning mode ──

    renderALPatch(p) {
        var imgSrc = '/image/' + p.patch_path;
        var pct = ((p.index + 1) / p.total * 100).toFixed(1);
        var disabledAttr = this.state.history.length === 0 ? 'disabled' : '';
        var taskLabel, taskColor, action1Label, action1Class, action2Label, action2Class, action1Value, action2Value;
        var modelBox = '';
        var clusterInfo = '';

        if (p.task_type === 'verify_pseudo') {
            taskLabel = 'VERIFIKASI PSEUDO-LABEL';
            taskColor = '#3498db';
            action1Label = '&#10003; Benar';
            action1Class = 'btn-correct';
            action1Value = 'correct';
            action2Label = '&#10007; Salah';
            action2Class = 'btn-wrong';
            action2Value = 'wrong';
            var conf = (p.model_confidence !== null && p.model_confidence !== undefined)
                ? p.model_confidence.toFixed(3) : '?';
            var margin = (p.model_margin !== null && p.model_margin !== undefined)
                ? p.model_margin.toFixed(3) : '?';
            modelBox =
                '<div class="model-prediction-box">' +
                    '<div class="model-prediction-label">Prediksi model:</div>' +
                    '<div class="model-prediction-value">' +
                        '<strong>' + escapeHtml(p.model_prediction || '?') + '</strong>' +
                    '</div>' +
                    '<div class="model-prediction-meta">' +
                        'keyakinan: ' + conf + ' &nbsp;|&nbsp; ' +
                        'margin: ' + margin +
                    '</div>' +
                '</div>';
        } else if (p.task_type === 'label_hitl') {
            taskLabel = 'LABEL HITL — Label Manual';
            taskColor = '#e67e22';
            action1Label = '&#10003; Sehat';
            action1Class = 'btn-healthy';
            action1Value = 'healthy';
            action2Label = '&#10007; Tidak Sehat';
            action2Class = 'btn-unhealthy';
            action2Value = 'unhealthy';
            if (p.cluster_id !== null && p.cluster_id !== undefined) {
                var cMargin = (p.cluster_margin !== null && p.cluster_margin !== undefined)
                    ? p.cluster_margin.toFixed(4) : '?';
                clusterInfo =
                    '<div class="cluster-info-box">' +
                        '<span class="cluster-info-label">Cluster #' + p.cluster_id + '</span>' +
                        '<span class="cluster-info-meta">margin: ' + cMargin + ' (paling tidak yakin di klaster)</span>' +
                    '</div>';
            }
        } else {
            taskLabel = 'UNKNOWN TASK: ' + escapeHtml(p.task_type || '?');
            taskColor = '#e74c3c';
            action1Label = 'OK';
            action1Class = 'btn-healthy';
            action1Value = 'healthy';
            action2Label = 'Bad';
            action2Class = 'btn-unhealthy';
            action2Value = 'unhealthy';
        }

        // Per-task-type progress (2.1)
        var verifyPct = (p.verify_total > 0)
            ? (p.verify_done / p.verify_total * 100).toFixed(0) : 0;
        var hitlPct = (p.hitl_total > 0)
            ? (p.hitl_done / p.hitl_total * 100).toFixed(0) : 0;
        var taskProgress =
            '<div class="al-task-progress">' +
                '<div class="al-task-row al-task-verify">' +
                    '<span class="al-task-name">Verify pseudo</span>' +
                    '<span class="al-task-counts">' + p.verify_done + ' / ' + p.verify_total + '</span>' +
                    '<span class="al-task-pct">' + verifyPct + '%</span>' +
                '</div>' +
                '<div class="al-task-row al-task-hitl">' +
                    '<span class="al-task-name">Label HITL</span>' +
                    '<span class="al-task-counts">' + p.hitl_done + ' / ' + p.hitl_total + '</span>' +
                    '<span class="al-task-pct">' + hitlPct + '%</span>' +
                '</div>' +
                '<div class="al-task-row al-task-total">' +
                    '<span class="al-task-name">Total</span>' +
                    '<span class="al-task-counts">' + p.annotated_count + ' / ' + (p.verify_total + p.hitl_total) + '</span>' +
                    '<span class="al-task-pct">' + pct + '%</span>' +
                '</div>' +
            '</div>';

        // Indonesian task instructions
        var instructions = this._buildALInstructions(p, action1Label, action2Label);

        // Per-class progress (3.1)
        var classPct = (p.class_total > 0)
            ? (p.class_done / p.class_total * 100).toFixed(0) : 0;
        var classProgress = (p.class_total > 0)
            ? '<span class="al-class-progress">' + p.class_done + ' / ' + p.class_total + ' di kelas ini (' + classPct + '%)</span>'
            : '';

        var html =
            '<div class="al-task-banner" style="background:' + taskColor + '">' +
                taskLabel +
            '</div>' +
            instructions +
            '<div class="al-class-info">' +
                '<span class="al-class-name">' + escapeHtml(p.class_name) + '</span>' +
                '<span class="al-split">' + escapeHtml(p.split) + '</span>' +
                classProgress +
            '</div>' +
            modelBox +
            clusterInfo +
            '<div class="al-image-container">' +
                '<img class="al-patch-image" src="' + imgSrc + '" alt="Patch">' +
                '<div class="al-image-counter">' + (p.index + 1) + ' / ' + p.total + '</div>' +
            '</div>' +
            taskProgress +
            '<div class="al-button-row">' +
                '<button class="btn ' + action1Class + '" data-al-action="' + action1Value + '">' +
                    action1Label +
                '</button>' +
                '<button class="btn ' + action2Class + '" data-al-action="' + action2Value + '">' +
                    action2Label +
                '</button>' +
                '<button class="btn btn-skip" data-al-action="skip">Lewati (Skip)</button>' +
            '</div>' +
            '<div class="al-history-row">' +
                '<button class="btn-undo" data-al-action="undo" ' + disabledAttr + '>' +
                    '&#8592; Undo (note: undo reloads previous patch)' +
                '</button>' +
            '</div>';

        this.mainContent.className = 'main al-mode';
        this.mainContent.innerHTML = html;
        this.attachALListeners();
    }

    _buildALInstructions(p, action1Label, action2Label) {
        if (p.task_type === 'verify_pseudo') {
            var pred = escapeHtml(p.model_prediction || '?');
            var confPct = (p.model_confidence !== null && p.model_confidence !== undefined)
                ? (p.model_confidence * 100).toFixed(1) + '%' : '?';
            return '<div class="al-instructions" style="border-left: 4px solid #3498db;">' +
                '<div class="al-instructions-title">Apa yang harus dilakukan?</div>' +
                '<div class="al-instructions-body">' +
                    'Model memprediksi patch ini sebagai <strong>' + pred + '</strong> ' +
                    'dengan keyakinan <strong>' + confPct + '</strong>.<br>' +
                    'Periksa dengan teliti, lalu konfirmasi:' +
                '</div>' +
                '<ul class="al-instructions-list">' +
                    '<li><span class="al-instr-correct">&#10003; Benar</span> &rarr; Anda setuju dengan prediksi model</li>' +
                    '<li><span class="al-instr-wrong">&#10007; Salah</span> &rarr; Model keliru, perlu perbaikan</li>' +
                    '<li><span class="al-instr-skip">Lewati</span> &rarr; Tidak yakin, akan diulang</li>' +
                '</ul>' +
            '</div>';
        }
        if (p.task_type === 'label_hitl') {
            return '<div class="al-instructions" style="border-left: 4px solid #e67e22;">' +
                '<div class="al-instructions-title">Apa yang harus dilakukan?</div>' +
                '<div class="al-instructions-body">' +
                    'Model <strong>sangat tidak yakin</strong> dengan patch ini. ' +
                    'Beri label manual dari awal berdasarkan kondisi daun yang terlihat.' +
                '</div>' +
                '<ul class="al-instructions-list">' +
                    '<li><span class="al-instr-healthy">&#10003; Sehat</span> &rarr; Daun terlihat sehat (hijau, tidak bercak)</li>' +
                    '<li><span class="al-instr-unhealthy">&#10007; Tidak Sehat</span> &rarr; Daun menunjukkan gejala penyakit</li>' +
                    '<li><span class="al-instr-skip">Lewati</span> &rarr; Tidak yakin, akan diulang</li>' +
                '</ul>' +
            '</div>';
        }
        return '<div class="al-instructions" style="border-left: 4px solid #e74c3c;">' +
            '<div class="al-instructions-title">Tugas tidak dikenal: ' + escapeHtml(p.task_type || '?') + '</div>' +
        '</div>';
    }

    attachALListeners() {
        var self = this;
        this.mainContent.querySelectorAll('[data-al-action]').forEach(function(btn) {
            btn.addEventListener('click', function() {
                var action = btn.getAttribute('data-al-action');
                if (action === 'undo') {
                    self.callbacks.onUndo();
                } else if (action === 'skip') {
                    self.callbacks.onSkip();
                } else {
                    // 'correct' / 'wrong' / 'healthy' / 'unhealthy'
                    self.callbacks.onAnnotate(action);
                }
            });
        });
        var patchImg = this.mainContent.querySelector('.al-patch-image');
        if (patchImg) {
            patchImg.addEventListener('error', function() { handleImageError(patchImg); });
        }
    }

    // AL layout helpers — used when 3-panel leaf context is shown for AL
    buildALCenterPanelHTML(p) {
        var imgSrc = '/image/' + p.patch_path;
        var pct = ((p.index + 1) / p.total * 100).toFixed(1);
        var disabledAttr = this.state.history.length === 0 ? 'disabled' : '';
        var taskLabel, taskColor, action1Label, action1Class, action2Label, action2Class, action1Value, action2Value;
        var modelBox = '';
        var clusterInfo = '';

        if (p.task_type === 'verify_pseudo') {
            taskLabel = 'VERIFIKASI PSEUDO-LABEL';
            taskColor = '#3498db';
            action1Label = '&#10003; Benar';
            action1Class = 'btn-correct';
            action1Value = 'correct';
            action2Label = '&#10007; Salah';
            action2Class = 'btn-wrong';
            action2Value = 'wrong';
            var conf = (p.model_confidence !== null && p.model_confidence !== undefined)
                ? p.model_confidence.toFixed(3) : '?';
            var margin = (p.model_margin !== null && p.model_margin !== undefined)
                ? p.model_margin.toFixed(3) : '?';
            modelBox =
                '<div class="model-prediction-box">' +
                    '<div class="model-prediction-label">Prediksi model:</div>' +
                    '<div class="model-prediction-value">' +
                        '<strong>' + escapeHtml(p.model_prediction || '?') + '</strong>' +
                    '</div>' +
                    '<div class="model-prediction-meta">' +
                        'keyakinan: ' + conf + ' &nbsp;|&nbsp; ' +
                        'margin: ' + margin +
                    '</div>' +
                '</div>';
        } else if (p.task_type === 'label_hitl') {
            taskLabel = 'LABEL HITL — Label Manual';
            taskColor = '#e67e22';
            action1Label = '&#10003; Sehat';
            action1Class = 'btn-healthy';
            action1Value = 'healthy';
            action2Label = '&#10007; Tidak Sehat';
            action2Class = 'btn-unhealthy';
            action2Value = 'unhealthy';
            if (p.cluster_id !== null && p.cluster_id !== undefined) {
                var cMargin = (p.cluster_margin !== null && p.cluster_margin !== undefined)
                    ? p.cluster_margin.toFixed(4) : '?';
                clusterInfo =
                    '<div class="cluster-info-box">' +
                        '<span class="cluster-info-label">Cluster #' + p.cluster_id + '</span>' +
                        '<span class="cluster-info-meta">margin: ' + cMargin + ' (paling tidak yakin di klaster)</span>' +
                    '</div>';
            }
        } else {
            taskLabel = 'UNKNOWN TASK: ' + escapeHtml(p.task_type || '?');
            taskColor = '#e74c3c';
            action1Label = 'OK';
            action1Class = 'btn-healthy';
            action1Value = 'healthy';
            action2Label = 'Bad';
            action2Class = 'btn-unhealthy';
            action2Value = 'unhealthy';
        }

        var verifyPct = (p.verify_total > 0)
            ? (p.verify_done / p.verify_total * 100).toFixed(0) : 0;
        var hitlPct = (p.hitl_total > 0)
            ? (p.hitl_done / p.hitl_total * 100).toFixed(0) : 0;
        var taskProgress =
            '<div class="al-task-progress">' +
                '<div class="al-task-row al-task-verify">' +
                    '<span class="al-task-name">Verify pseudo</span>' +
                    '<span class="al-task-counts">' + p.verify_done + ' / ' + p.verify_total + '</span>' +
                    '<span class="al-task-pct">' + verifyPct + '%</span>' +
                '</div>' +
                '<div class="al-task-row al-task-hitl">' +
                    '<span class="al-task-name">Label HITL</span>' +
                    '<span class="al-task-counts">' + p.hitl_done + ' / ' + p.hitl_total + '</span>' +
                    '<span class="al-task-pct">' + hitlPct + '%</span>' +
                '</div>' +
                '<div class="al-task-row al-task-total">' +
                    '<span class="al-task-name">Total</span>' +
                    '<span class="al-task-counts">' + p.annotated_count + ' / ' + (p.verify_total + p.hitl_total) + '</span>' +
                    '<span class="al-task-pct">' + pct + '%</span>' +
                '</div>' +
            '</div>';

        var instructions = this._buildALInstructions(p, action1Label, action2Label);

        // Per-class progress (3.1)
        var classPct = (p.class_total > 0)
            ? (p.class_done / p.class_total * 100).toFixed(0) : 0;
        var classProgress = (p.class_total > 0)
            ? '<span class="al-class-progress">' + p.class_done + ' / ' + p.class_total + ' di kelas ini (' + classPct + '%)</span>'
            : '';

        return '<div class="al-task-banner" style="background:' + taskColor + '">' +
                taskLabel +
            '</div>' +
            instructions +
            '<div class="al-class-info">' +
                '<span class="al-class-name">' + escapeHtml(p.class_name) + '</span>' +
                '<span class="al-split">' + escapeHtml(p.split) + '</span>' +
                classProgress +
            '</div>' +
            modelBox +
            clusterInfo +
            '<div class="al-image-container">' +
                '<img class="al-patch-image" src="' + imgSrc + '" alt="Patch">' +
                '<div class="al-image-counter">' + (p.index + 1) + ' / ' + p.total + '</div>' +
            '</div>' +
            taskProgress +
            '<div class="al-button-row">' +
                '<button class="btn ' + action1Class + '" data-al-action="' + action1Value + '">' +
                    action1Label +
                '</button>' +
                '<button class="btn ' + action2Class + '" data-al-action="' + action2Value + '">' +
                    action2Label +
                '</button>' +
                '<button class="btn btn-skip" data-al-action="skip">Lewati (Skip)</button>' +
            '</div>' +
            '<div class="al-history-row">' +
                '<button class="btn-undo" data-al-action="undo" ' + disabledAttr + '>' +
                    '&#8592; Undo (note: undo reloads previous patch)' +
                '</button>' +
            '</div>' +
            '<div class="patch-strip" id="patchStrip"></div>';
    }

    buildFullALLayout(p) {
        this.mainContent.className = 'main al-mode-three-panel';
        this.mainContent.innerHTML =
            '<div class="panel panel-left" id="panelLeft">' +
                '<div class="panel-header">Full Leaf</div>' +
                '<div id="leafImageWrap"><div class="loading">Loading leaf...</div></div>' +
            '</div>' +
            '<div class="panel panel-center" id="panelCenter">' +
                this.buildALCenterPanelHTML(p) +
            '</div>' +
            '<div class="panel panel-right" id="panelRight">' +
                '<div class="panel-header">Leaf Overview</div>' +
                '<div id="gridWrap"><div class="loading">Loading grid...</div></div>' +
                '<div class="leaf-stats" id="leafStats"></div>' +
            '</div>';
        this.attachALListeners();
    }

    updateALCenterPanel(p) {
        var center = document.getElementById('panelCenter');
        if (center) {
            center.innerHTML = this.buildALCenterPanelHTML(p);
        }
        this.attachALListeners();
    }

    showALDoneScreen() {
        var round = (this.state && this.state.round) || '?';
        this.mainContent.className = 'main single-panel';
        this.mainContent.innerHTML =
            '<div class="done-screen">' +
                '<h2>All Done for Round ' + round + '!</h2>' +
                '<p>You have processed all assigned active-learning patches.</p>' +
                '<p style="margin-top: 16px;">' +
                    'After all 5 annotators finish, run:<br>' +
                    '<code>python active_learning_round.py --phase 2 verify --round ' + round + '</code>' +
                '</p>' +
            '</div>';
    }

    // ── Flash notification ──

    showFlash(type) {
        var label = FLASH_LABELS[type];
        var duration = 800;
        if (label) {
            this.flash.textContent = label;
            this.flash.className = 'flash ' + type + ' show';
        } else {
            this.flash.textContent = type;
            this.flash.className = 'flash info show';
            duration = 2500;
        }
        setTimeout(() => {
            this.flash.classList.remove('show');
        }, duration);
    }

    // ── Toast notifications (non-blocking, stackable) ──

    showToast(message, type, duration) {
        if (!this.toastContainer) return;
        type = type || 'info';
        duration = duration || 4000;
        // Cap visible toasts at 3 — drop oldest if over
        var visible = this.toastContainer.querySelectorAll('.toast.show').length;
        if (visible >= 3) {
            var oldest = this.toastContainer.querySelector('.toast.show');
            if (oldest) {
                oldest.classList.remove('show');
                var self = this;
                setTimeout(function() { oldest.remove(); }, 250);
            }
        }
        var toast = document.createElement('div');
        toast.className = 'toast ' + type;
        toast.textContent = message;
        this.toastContainer.appendChild(toast);
        // Trigger reflow then add 'show' for transition
        toast.offsetHeight; // eslint-disable-line no-unused-expressions
        toast.classList.add('show');
        setTimeout(function() {
            toast.classList.remove('show');
            setTimeout(function() { toast.remove(); }, 250);
        }, duration);
    }

    // ── Modal helpers (Phase 4) ──

    hideAllModals() {
        if (this.alIntroModal) this.alIntroModal.classList.add('hidden');
        if (this.helpModal) this.helpModal.classList.add('hidden');
        if (this.historyModal) this.historyModal.classList.add('hidden');
    }

    isAnyModalOpen() {
        return (this.alIntroModal && !this.alIntroModal.classList.contains('hidden'))
            || (this.helpModal && !this.helpModal.classList.contains('hidden'))
            || (this.historyModal && !this.historyModal.classList.contains('hidden'));
    }

    showALIntroModal(total) {
        if (!this.alIntroModal) return;
        var totalEl = document.getElementById('alIntroTotal');
        if (totalEl && total) totalEl.textContent = total;
        this.alIntroModal.classList.remove('hidden');
    }

    hideALIntroModal() {
        if (this.alIntroModal) this.alIntroModal.classList.add('hidden');
    }

    showHelpModal() {
        if (!this.helpModal) return;
        this.helpModal.classList.remove('hidden');
    }

    hideHelpModal() {
        if (this.helpModal) this.helpModal.classList.add('hidden');
    }

    showHistoryModal(items) {
        if (!this.historyModal) return;
        this._renderHistoryList(items || []);
        this.historyModal.classList.remove('hidden');
    }

    hideHistoryModal() {
        if (this.historyModal) this.historyModal.classList.add('hidden');
    }

    _renderHistoryList(items) {
        if (!this.historyList) return;
        if (!items || items.length === 0) {
            this.historyList.innerHTML = '<div class="history-empty">Belum ada anotasi di sesi ini.</div>';
            return;
        }
        var html = '';
        items.forEach(function(item) {
            var isSkipped = item.is_skipped === 'True' || item.is_skipped === true;
            var displayLabel = isSkipped ? 'Skipped' : item.label;
            var labelClass = isSkipped ? 'skipped'
                : (item.label || '').toLowerCase()
                    .replace(' ', '');
            var shortPath = (item.patch_path || '').split('/').pop();
            html += '<div class="history-item">' +
                '<span class="history-task ' + escapeHtml(item.task_type || '') + '">' +
                    escapeHtml(item.task_type || '') +
                '</span>' +
                '<span class="history-path" title="' + escapeHtml(item.patch_path || '') + '">' +
                    escapeHtml(shortPath) +
                '</span>' +
                '<span class="history-label ' + escapeHtml(labelClass) + '">' +
                    escapeHtml(displayLabel) +
                '</span>' +
            '</div>';
        });
        this.historyList.innerHTML = html;
    }
}
