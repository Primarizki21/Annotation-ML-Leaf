/**
 * ApiClient — All HTTP calls to the FastAPI backend.
 *
 * Each method returns a Promise<Object>. Callers handle errors with .catch().
 * No DOM access, no state mutation.
 *
 * Endpoints:
 *   getStatus()         → GET  /api/status
 *   doSetup(name)       → POST /api/setup
 *   getCurrentPatch()   → GET  /api/patch/current
 *   annotate(path, lbl) → POST /api/annotate
 *   skipPatch(path)     → POST /api/skip
 *   undo()              → POST /api/undo
 *   jumpToPatch(path)   → GET  /api/jump-to-patch
 *   fetchLeafContext(patch) → GET /api/leaf-context/...
 *
 * @imports { getLeafStem } from './utils.js'
 * @exports { ApiClient }
 */

import { getLeafStem } from './utils.js';

export class ApiClient {
    getStatus() {
        return fetch('/api/status').then(function(res) { return res.json(); });
    }

    doSetup(name) {
        return fetch('/api/setup', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({name: name})
        }).then(function(res) {
            if (!res.ok) {
                return res.json().then(function(err) {
                    throw new Error(err.detail || 'Setup failed');
                });
            }
            return res.json();
        });
    }

    setupReview(name) {
        return fetch('/api/setup-review', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({name: name})
        }).then(function(res) {
            if (!res.ok) {
                return res.json().then(function(err) {
                    throw new Error(err.detail || 'Review setup failed');
                });
            }
            return res.json();
        });
    }

    setupNormal(name) {
        return fetch('/api/setup-normal', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({name: name})
        }).then(function(res) {
            if (!res.ok) {
                return res.json().then(function(err) {
                    throw new Error(err.detail || 'Normal setup failed');
                });
            }
            return res.json();
        });
    }

    getCurrentPatch() {
        return fetch('/api/patch/current').then(function(res) { return res.json(); });
    }

    annotate(patchPath, label) {
        return fetch('/api/annotate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({patch_path: patchPath, label: label})
        }).then(function(res) {
            if (!res.ok) throw new Error('Annotate failed');
            return res.json();
        });
    }

    skipPatch(patchPath) {
        return fetch('/api/skip', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({patch_path: patchPath})
        }).then(function(res) {
            if (!res.ok) throw new Error('Skip failed');
            return res.json();
        });
    }

    undo() {
        return fetch('/api/undo', {method: 'POST'}).then(function(res) {
            if (!res.ok) throw new Error('Undo failed');
            return res.json();
        });
    }

    jumpToPatch(patchPath) {
        return fetch('/api/jump-to-patch?patch_path=' + encodeURIComponent(patchPath))
            .then(function(res) {
                if (!res.ok) throw new Error('Jump failed');
                return res.json();
            });
    }

    fetchLeafContext(patch) {
        var stem = getLeafStem(patch.patch_path);
        var url = '/api/leaf-context/' + patch.split + '/' +
                  encodeURIComponent(patch.class_name) + '/' +
                  encodeURIComponent(stem);
        return fetch(url).then(function(res) {
            if (!res.ok) throw new Error('Leaf context fetch failed');
            return res.json();
        });
    }
}
