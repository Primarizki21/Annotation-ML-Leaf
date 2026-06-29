/**
 * App — Main orchestrator for the annotation page.
 *
 * Creates and wires: State (data), ApiClient (backend calls), Renderer (DOM).
 * Handles all user actions: annotate, skip, undo, jump-to-patch.
 * Sets up keyboard shortcuts (H/U/S/ArrowLeft).
 *
 * @imports { State } from './state.js'
 * @imports { ApiClient } from './api.js'
 * @imports { Renderer } from './renderer.js'
 * @imports { getLeafStem } from './utils.js'
 * @exports { App }
 */

import { State } from './state.js';
import { ApiClient } from './api.js';
import { Renderer } from './renderer.js';
import { getLeafStem } from './utils.js';

export class App {
    constructor() {
        this.state = new State();
        this.api = new ApiClient();

        this.renderer = new Renderer(this.state, {
            onAnnotate: (label) => this.annotate(label),
            onSkip: () => this.skipPatch(),
            onUndo: () => this.undo(),
            onJumpToPatch: (path) => this.jumpToPatch(path)
        });

        // DOM references for setup modal + header
        this.setupModal = document.getElementById('setupModal');
        this.setupError = document.getElementById('setupError');
        this.nameInput = document.getElementById('nameInput');
        this.setupBtn = document.getElementById('setupBtn');
        this.annotatorNameEl = document.getElementById('annotatorName');
        this.reviewBtn = document.getElementById('reviewBtn');
        this.alBtn = document.getElementById('alBtn');
        this.historyBtn = document.getElementById('historyBtn');
        this.helpBtn = document.getElementById('helpBtn');

        // Bind setup events
        this.setupBtn.addEventListener('click', () => this.doSetup());
        this.nameInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') this.doSetup();
        });

        // Review mode toggle
        this.reviewBtn.addEventListener('click', () => this.toggleReviewMode());

        // AL mode toggle
        if (this.alBtn) {
            this.alBtn.addEventListener('click', () => this.toggleALMode());
        }

        // History + Help buttons (Phase 4.2 + 4.3)
        if (this.historyBtn) {
            this.historyBtn.addEventListener('click', () => this.toggleHistoryModal());
        }
        if (this.helpBtn) {
            this.helpBtn.addEventListener('click', () => this.renderer.showHelpModal());
        }
        var alIntroBtn = document.getElementById('alIntroBtn');
        if (alIntroBtn) {
            alIntroBtn.addEventListener('click', () => this.dismissIntroModal());
        }
        var helpCloseBtn = document.getElementById('helpCloseBtn');
        if (helpCloseBtn) {
            helpCloseBtn.addEventListener('click', () => this.renderer.hideHelpModal());
        }
        var historyCloseBtn = document.getElementById('historyCloseBtn');
        if (historyCloseBtn) {
            historyCloseBtn.addEventListener('click', () => this.renderer.hideHistoryModal());
        }

        this.setupKeyboardShortcuts();
    }

    init() {
        var self = this;
        this.api.getStatus()
            .then((data) => {
                if (data.setup) {
                    self.state.annotatorName = data.name;
                    self.state.mode = data.mode || 'normal';
                    self.state.round = data.round || null;
                    self.annotatorNameEl.textContent = data.name;
                    var revBanner = document.getElementById('reviewBanner');
                    var alBanner = document.getElementById('alBanner');
                    if (data.mode === 'review') {
                        self.annotatorNameEl.textContent = '[Review] ' + data.name;
                        if (revBanner) revBanner.classList.remove('hidden');
                    } else {
                        if (revBanner) revBanner.classList.add('hidden');
                    }
                    if (data.mode === 'al') {
                        self.annotatorNameEl.textContent =
                            '[AL R' + (data.round || '?') + '] ' + data.name;
                        if (alBanner) {
                            alBanner.classList.remove('hidden');
                            var roundEl = document.getElementById('alRound');
                            if (roundEl) roundEl.textContent = data.round || '?';
                        }
                    } else {
                        if (alBanner) alBanner.classList.add('hidden');
                    }
                    self.updateReviewButton(data.mode, data.has_disputed);
                    self.updateALButton(data.mode, data.round);
                    self.updateHistoryButton(data.mode);
                    self.setupModal.classList.add('hidden');
                    self.loadCurrentPatch();
                    // Phase 5.4: session resume notification
                    if (data.annotated > 0) {
                        self.renderer.showToast(
                            'Melanjutkan sesi. ' + data.annotated + ' patch sudah dianotasi.',
                            'info', 4000);
                    }
                    if (data.mode === 'al' && !data.seen_intro) {
                        // First-time AL session — show intro modal
                        self.api.getCurrentALPatch().then(function(patch) {
                            var total = patch && patch.total ? patch.total : 320;
                            self.renderer.showALIntroModal(total);
                        }).catch(function() {
                            self.renderer.showALIntroModal(320);
                        });
                    }
                } else {
                    self.setupModal.classList.remove('hidden');
                }
            })
            .catch((err) => {
                console.error('Init error:', err);
                self.renderer.showToast('Gagal memuat status. Periksa koneksi.', 'error', 5000);
                self.setupModal.classList.remove('hidden');
            });
    }

    updateReviewButton(mode, hasDisputed) {
        if (!this.reviewBtn) return;
        this.reviewBtn.style.display = '';
        if (mode === 'review') {
            this.reviewBtn.textContent = 'Exit Review';
        } else if (hasDisputed) {
            this.reviewBtn.textContent = 'Review Disputed';
        } else {
            this.reviewBtn.style.display = 'none';
        }
    }

    updateALButton(mode, round) {
        if (!this.alBtn) return;
        if (mode === 'al') {
            this.alBtn.textContent = 'Exit AL';
        } else {
            this.alBtn.textContent = 'Active Learning' + (round ? ' R' + round : '');
            this.alBtn.style.display = '';
        }
    }

    updateHistoryButton(mode) {
        if (!this.historyBtn) return;
        // Show History button only in AL mode (data source is annotations_*_al_round*.csv)
        this.historyBtn.style.display = (mode === 'al') ? '' : 'none';
    }

    dismissIntroModal() {
        this.renderer.hideALIntroModal();
        this.api.markIntroSeen().catch(function(err) {
            console.warn('Could not mark intro seen:', err);
        });
    }

    toggleHistoryModal() {
        if (this.state.mode !== 'al') return;
        if (this.renderer.isAnyModalOpen()
            && !this.renderer.historyModal.classList.contains('hidden')) {
            this.renderer.hideHistoryModal();
            return;
        }
        this.renderer.showHistoryModal([]);  // shows "Memuat..." placeholder
        var self = this;
        this.api.getALHistory().then(function(data) {
            self.renderer.showHistoryModal(data.history || []);
        }).catch(function(err) {
            console.error('History fetch error:', err);
            self.renderer.showHistoryModal([]);
        });
    }

    toggleALMode() {
        var name = this.state.annotatorName;
        if (!name) {
            name = prompt('Masukkan nama Anda:');
            if (!name) return;
        }
        if (this.state.mode === 'al') {
            // Exit AL mode -> back to normal
            this.api.setupNormal(name)
                .then(() => { window.location.reload(); })
                .catch((err) => {
                    alert('Gagal keluar AL mode: ' + (err.message || 'unknown error'));
                });
        } else {
            // Enter AL mode -> ask for round (default 2)
            var roundStr = prompt('Round berapa? (default: 2)', '2');
            if (roundStr === null) return;
            var round = parseInt(roundStr, 10) || 2;
            this.api.setupAL(name, round)
                .then(() => { window.location.reload(); })
                .catch((err) => {
                    alert('Gagal masuk AL mode: ' + (err.message || 'unknown error'));
                });
        }
    }

    toggleReviewMode() {
        var name = this.state.annotatorName;
        if (!name) return;

        if (this.state.mode === 'review') {
            // Switch back to normal mode
            this.api.setupNormal(name)
                .then(() => {
                    window.location.reload();
                })
            .catch((err) => {
                console.error('Switch to normal failed:', err);
                this.renderer.showToast('Gagal kembali ke mode normal.', 'error');
            });
        } else {
            // Switch to review mode
            this.api.setupReview(name)
                .then(() => {
                    window.location.reload();
                })
                .catch((err) => {
                    this.renderer.showFlash('Download data review terlebih dahulu untuk memulai review.');
                });
        }
    }

    doSetup() {
        var name = this.nameInput.value.trim();
        if (!name) {
            this.setupError.textContent = 'Please enter your name';
            this.setupError.style.display = 'block';
            return;
        }
        this.api.doSetup(name)
            .then((data) => {
                this.state.annotatorName = data.name;
                this.state.mode = 'normal';
                this.annotatorNameEl.textContent = data.name;
                this.setupModal.classList.add('hidden');
                this.loadCurrentPatch();
                // Re-fetch full status (single source of truth) to ensure
                // buttons show even if /api/setup response is incomplete
                // (cached backend, future field drift, etc.)
                this.init();
            })
            .catch((err) => {
                this.setupError.textContent = err.message || 'Connection error';
                this.setupError.style.display = 'block';
            });
    }

    loadCurrentPatch() {
        if (this.state.mode === 'al') {
            this.api.getCurrentALPatch()
                .then((data) => {
                    if (data.done) {
                        this.state.done = true;
                        this.renderer.showALDoneScreen();
                        return;
                    }
                    this.state.currentPatch = data;
                    this.state.loading = false;
                    this.handleALLeafSwitch();
                })
                .catch((err) => {
                    console.error('Load AL error:', err);
                    this.renderer.showToast('Gagal memuat patch AL.', 'error');
                });
            return;
        }
        this.api.getCurrentPatch()
            .then((data) => {
                if (data.done) {
                    this.state.done = true;
                    this.renderer.showDoneScreen(this.state.mode);
                    return;
                }
                this.state.currentPatch = data;
                this.state.loading = false;
                this.handleLeafSwitch();
            })
            .catch((err) => {
                console.error('Load error:', err);
                this.renderer.showToast('Gagal memuat patch.', 'error');
            });
    }

    async handleALLeafSwitch() {
        var p = this.state.currentPatch;
        if (!p) return;
        var newStem = getLeafStem(p.patch_path);

        if (newStem === this.state.currentLeafStem && this.state.leafContext) {
            // Same leaf — only update center panel + grid overlay
            this.renderer.updateProgress(p);
            this.renderer.updateALCenterPanel(p);
            this.state.markCurrentPatchInContext(p.patch_path);
            this.renderer.renderRightPanel(this.state.leafContext);
            this.renderer.renderPatchStrip(this.state.leafContext);
        } else {
            // Different leaf or first load — rebuild full 3-panel layout
            this.state.currentLeafStem = newStem;
            this.renderer.updateProgress(p);
            this.renderer.buildFullALLayout(p);

            try {
                var ctx = await this.api.fetchLeafContext(p);
                this.state.leafContext = ctx;
                this.renderer.renderLeftPanel(ctx);
                this.renderer.renderRightPanel(ctx);
                this.renderer.renderPatchStrip(ctx);
            } catch (err) {
                console.error('Leaf context error:', err);
                this.renderer.handleLeafContextError();
                this.renderer.showToast('Gagal memuat konteks daun. Menampilkan tampilan ringkas.', 'warning');
                // Fallback: build a single-column AL layout (the old style)
                this.renderer.renderALPatch(p);
            }
        }
    }

    async handleLeafSwitch() {
        var p = this.state.currentPatch;
        if (!p) return;
        var newStem = getLeafStem(p.patch_path);

        if (newStem === this.state.currentLeafStem && this.state.leafContext) {
            // Same leaf — only update center panel + grid overlay
            this.renderer.updateProgress(p);
            this.renderer.updateCenterPanel(p);
            this.state.markCurrentPatchInContext(p.patch_path);
            this.renderer.renderRightPanel(this.state.leafContext);
            this.renderer.renderPatchStrip(this.state.leafContext);
        } else {
            // Different leaf or first load — rebuild everything
            this.state.currentLeafStem = newStem;
            this.renderer.updateProgress(p);
            this.renderer.buildFullLayout(p);

            try {
                var ctx = await this.api.fetchLeafContext(p);
                this.state.leafContext = ctx;
                this.renderer.renderLeftPanel(ctx);
                this.renderer.renderRightPanel(ctx);
                this.renderer.renderPatchStrip(ctx);
            } catch (err) {
                console.error('Leaf context error:', err);
                this.renderer.handleLeafContextError();
                this.renderer.showToast('Gagal memuat konteks daun.', 'warning');
            }
        }
    }

    annotate(label) {
        if (!this.state.currentPatch || this.state.loading) return;
        var patch = this.state.currentPatch;
        if (this.state.mode === 'al') {
            // For label_hitl, label is the annotator's choice ("healthy"/"unhealthy").
            // For verify_pseudo, is_correct is the annotator's choice (label is unused).
            this.state.loading = true;
            var isCorrect = (patch.task_type === 'verify_pseudo') ? (label === 'correct') : null;
            this.api.annotateAL(patch.patch_path, patch.task_type, label, isCorrect)
                .then((data) => {
                    this.state.history.push({
                        patch_path: patch.patch_path,
                        class_name: patch.class_name,
                        task_type: patch.task_type,
                        label: label
                    });
                    var flashText = (patch.task_type === 'verify_pseudo')
                        ? (isCorrect ? 'Correct' : 'Wrong')
                        : (label === 'healthy' ? 'Healthy' : 'Unhealthy');
                    this.renderer.showFlash(flashText);
                    if (data.done) {
                        this.state.done = true;
                        this.renderer.showALDoneScreen();
                    } else {
                        this.state.currentPatch = data;
                        this.state.loading = false;
                        this.handleALLeafSwitch();
                    }
                })
                .catch((err) => {
                    this.state.loading = false;
                    console.error('AL annotate error:', err);
                    this.renderer.showToast('Gagal menyimpan anotasi AL.', 'error');
                });
            return;
        }
        this.state.loading = true;

        this.api.annotate(this.state.currentPatch.patch_path, label)
            .then((data) => {
                var annotatedPath = this.state.currentPatch.patch_path;
                this.state.history.push({
                    patch_path: annotatedPath,
                    class_name: this.state.currentPatch.class_name,
                    label: label
                });
                this.state.updateCachedLeafLabel(annotatedPath, label);
                this.renderer.showFlash(label);

                if (data.done) {
                    this.state.done = true;
                    this.renderer.showDoneScreen(this.state.mode);
                } else {
                    this.state.currentPatch = data;
                    this.state.loading = false;
                    this.handleLeafSwitch();
                }
            })
            .catch((err) => {
                this.state.loading = false;
                console.error('Annotate error:', err);
                this.renderer.showToast('Gagal menyimpan anotasi.', 'error');
            });
    }

    skipPatch() {
        if (!this.state.currentPatch || this.state.loading) return;
        this.state.loading = true;

        var patchPath = this.state.currentPatch.patch_path;
        var skipPromise = (this.state.mode === 'al')
            ? this.api.skipALPatch(patchPath)
            : this.api.skipPatch(patchPath);

        skipPromise.then((data) => {
            if (this.state.mode !== 'al') {
                this.state.updateCachedLeafLabel(patchPath, 'skipped');
            }
            this.renderer.showFlash('skip');

            if (data.done) {
                this.state.done = true;
                if (this.state.mode === 'al') {
                    this.renderer.showALDoneScreen();
                } else {
                    this.renderer.showDoneScreen(this.state.mode);
                }
            } else {
                this.state.currentPatch = data;
                this.state.loading = false;
                if (this.state.mode === 'al') {
                    this.handleALLeafSwitch();
                } else {
                    this.handleLeafSwitch();
                }
            }
        })
        .catch((err) => {
            this.state.loading = false;
            console.error('Skip error:', err);
            this.renderer.showToast('Gagal skip patch.', 'error');
        });
    }

    undo() {
        if (this.state.history.length === 0 || this.state.loading) return;
        this.state.loading = true;

        this.api.undo()
            .then((data) => {
                this.state.history.pop();
                this.state.currentPatch = data;
                this.state.loading = false;
                // Phase 5.1: smooth undo — only nuke leaf context if patch moved
                // to a different leaf. Same-leaf undo just removes the cached label
                // and updates the center panel without a full reload.
                var newStem = getLeafStem(data.patch_path);
                if (this.state.currentLeafStem && newStem === this.state.currentLeafStem) {
                    this.state.removeCachedLeafLabel(data.patch_path);
                } else {
                    this.state.leafContext = null;
                    this.state.currentLeafStem = null;
                }
                if (this.state.mode === 'al') {
                    this.handleALLeafSwitch();
                } else {
                    this.handleLeafSwitch();
                }
            })
            .catch((err) => {
                this.state.loading = false;
                console.error('Undo error:', err);
                // Phase 5.2: distinguish history desync (server has no history)
                // from real network/parse errors
                var msg = (err && err.message) || 'Gagal undo.';
                if (msg.toLowerCase().includes('nothing to undo')) {
                    // Server has no history — sync our local state
                    this.state.history = [];
                    this.renderer.showToast('Tidak ada anotasi untuk di-undo.', 'warning');
                } else {
                    this.renderer.showToast('Gagal undo: ' + msg, 'error');
                }
            });
    }

    jumpToPatch(patchPath) {
        if (this.state.loading) return;
        this.state.loading = true;

        this.api.jumpToPatch(patchPath)
            .then((data) => {
                this.state.currentPatch = data;
                this.state.loading = false;
                if (this.state.mode === 'al') {
                    this.handleALLeafSwitch();
                } else {
                    this.handleLeafSwitch();
                }
            })
            .catch((err) => {
                this.state.loading = false;
                console.error('Jump error:', err);
                this.renderer.showToast('Gagal loncat ke patch.', 'error');
            });
    }

    setupKeyboardShortcuts() {
        var self = this;
        document.addEventListener('keydown', (e) => {
            // Esc closes any open modal
            if (e.key === 'Escape') {
                if (self.renderer.isAnyModalOpen()) {
                    e.preventDefault();
                    self.renderer.hideAllModals();
                    return;
                }
            }
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            if (!self.setupModal.classList.contains('hidden')) return;
            // Don't trigger annotation shortcuts if a modal is open
            if (self.renderer.isAnyModalOpen()) {
                // Only allow Escape (handled above) and '?' in modals
                if (e.key === '?') {
                    e.preventDefault();
                    self.renderer.showHelpModal();
                }
                return;
            }

            // ? opens help (Shift not required)
            if (e.key === '?') {
                e.preventDefault();
                self.renderer.showHelpModal();
                return;
            }
            // Shift+H opens history (AL mode only)
            if (e.key === 'H' && e.shiftKey) {
                e.preventDefault();
                self.toggleHistoryModal();
                return;
            }

            switch (e.key.toLowerCase()) {
                case 'h':
                    e.preventDefault();
                    this.annotate('healthy');
                    break;
                case 'u':
                    e.preventDefault();
                    this.annotate('unhealthy');
                    break;
                case 's':
                    e.preventDefault();
                    this.skipPatch();
                    break;
                case 'arrowleft':
                    e.preventDefault();
                    this.undo();
                    break;
            }
        });
    }
}
