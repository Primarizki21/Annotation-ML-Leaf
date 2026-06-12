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

        // Bind setup events
        this.setupBtn.addEventListener('click', () => this.doSetup());
        this.nameInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') this.doSetup();
        });

        this.setupKeyboardShortcuts();
    }

    init() {
        this.api.getStatus()
            .then((data) => {
                if (data.setup) {
                    this.state.annotatorName = data.name;
                    this.annotatorNameEl.textContent = data.name;
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
        this.api.getCurrentPatch()
            .then((data) => {
                if (data.done) {
                    this.state.done = true;
                    this.renderer.showDoneScreen();
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
                    this.renderer.showDoneScreen();
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

        this.api.skipPatch(this.state.currentPatch.patch_path)
            .then((data) => {
                this.state.updateCachedLeafLabel(this.state.currentPatch.patch_path, 'skipped');
                this.renderer.showFlash('skip');

                if (data.done) {
                    this.state.done = true;
                    this.renderer.showDoneScreen();
                } else {
                    this.state.currentPatch = data;
                    this.state.loading = false;
                    this.handleLeafSwitch();
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
