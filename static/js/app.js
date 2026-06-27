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

        this.setupKeyboardShortcuts();
    }

    init() {
        this.api.getStatus()
            .then((data) => {
                if (data.setup) {
                    this.state.annotatorName = data.name;
                    this.state.mode = data.mode || 'normal';
                    this.state.round = data.round || null;
                    this.annotatorNameEl.textContent = data.name;
                    var revBanner = document.getElementById('reviewBanner');
                    var alBanner = document.getElementById('alBanner');
                    if (data.mode === 'review') {
                        this.annotatorNameEl.textContent = '[Review] ' + data.name;
                        if (revBanner) revBanner.classList.remove('hidden');
                    } else {
                        if (revBanner) revBanner.classList.add('hidden');
                    }
                    if (data.mode === 'al') {
                        this.annotatorNameEl.textContent =
                            '[AL R' + (data.round || '?') + '] ' + data.name;
                        if (alBanner) {
                            alBanner.classList.remove('hidden');
                            var roundEl = document.getElementById('alRound');
                            if (roundEl) roundEl.textContent = data.round || '?';
                        }
                    } else {
                        if (alBanner) alBanner.classList.add('hidden');
                    }
                    this.updateReviewButton(data.mode, data.has_disputed);
                    this.updateALButton(data.mode, data.round);
                    this.setupModal.classList.add('hidden');
                    this.loadCurrentPatch();
                } else {
                    this.setupModal.classList.remove('hidden');
                }
            })
            .catch((err) => {
                console.error('Init error:', err);
                this.setupModal.classList.remove('hidden');
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
                    this.renderer.renderALPatch(data);
                })
                .catch((err) => {
                    console.error('Load AL error:', err);
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
            });
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
                        this.renderer.renderALPatch(data);
                    }
                })
                .catch((err) => {
                    this.state.loading = false;
                    console.error('AL annotate error:', err);
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
                    this.renderer.renderALPatch(data);
                } else {
                    this.handleLeafSwitch();
                }
            }
        })
        .catch((err) => {
            this.state.loading = false;
            console.error('Skip error:', err);
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
                this.state.leafContext = null;
                this.state.currentLeafStem = null;
                this.handleLeafSwitch();
            })
            .catch((err) => {
                this.state.loading = false;
                console.error('Undo error:', err);
            });
    }

    jumpToPatch(patchPath) {
        if (this.state.loading) return;
        this.state.loading = true;

        this.api.jumpToPatch(patchPath)
            .then((data) => {
                this.state.currentPatch = data;
                this.state.loading = false;
                this.handleLeafSwitch();
            })
            .catch((err) => {
                this.state.loading = false;
                console.error('Jump error:', err);
            });
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            if (!this.setupModal.classList.contains('hidden')) return;

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
