/**
 * State — Central data store for the annotation session.
 *
 * Holds all application state. No DOM access, no fetch calls.
 * Read/written by App, read by Renderer.
 *
 * Fields:
 *   currentPatch     — the patch being shown (Object from API)
 *   history          — stack of annotated patches for undo (Array)
 *   annotatorName    — user's name (String)
 *   done             — all patches annotated? (Boolean)
 *   loading          — request in flight? (Boolean)
 *   leafContext      — full leaf info: patches, image, grid size (Object)
 *   currentLeafStem  — filename stem of current leaf (String)
 *
 * @exports { State }
 */

export class State {
    constructor() {
        this.currentPatch = null;
        this.history = [];
        this.annotatorName = '';
        this.mode = 'normal';
        this.done = false;
        this.loading = true;
        this.leafContext = null;
        this.currentLeafStem = null;
    }

    updateCachedLeafLabel(patchPath, label) {
        if (!this.leafContext) return;
        for (var i = 0; i < this.leafContext.patches.length; i++) {
            if (this.leafContext.patches[i].patch_path === patchPath) {
                if (!this.leafContext.patches[i].label) {
                    this.leafContext.annotated_count++;
                }
                this.leafContext.patches[i].label = label;
                break;
            }
        }
    }

    markCurrentPatchInContext(patchPath) {
        if (!this.leafContext) return;
        this.leafContext.patches.forEach(function(pp) {
            pp.is_current = (pp.patch_path === patchPath);
        });
    }
}
