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
                    c.lineWidth = 3;
                    c.strokeRect(x + 1, y + 1, cellW - 2, cellH - 2);
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
}
