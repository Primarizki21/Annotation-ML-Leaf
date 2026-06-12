# JS Architecture — Annotation Page

## Read this first

This directory contains the JavaScript for the annotation page (`/`).
The code is organized as **ES Modules** — each file exports a single class
with a clear responsibility. No build tools needed.

## File map

| File | Job |
|------|-----|
| `index.js` | Entry point. Creates `App`, calls `init()`. **3 lines.** |
| `app.js` | **Orchestrator.** Wires State + ApiClient + Renderer together. Handles setup, keyboard shortcuts, and the main action flow. |
| `state.js` | **Data store.** Holds current patch, history, leaf context, loading flags. No DOM, no fetch. |
| `api.js` | **HTTP client.** All `fetch()` calls to the FastAPI backend. Each method returns a Promise. |
| `renderer.js` | **DOM artist.** Builds HTML, draws the canvas grid, shows flash messages. Reads from State, never calls the API. |
| `utils.js` | **Shared helpers.** `getLeafStem()`, `escapeHtml()`, `handleImageError()`. |

## Data flow

```
User clicks / presses key
       │
       ▼
    App.handleAction()          ← orchestrator
       │
       ├──▶ ApiClient.fetch()   ← talks to backend
       │       ◀── JSON
       │
       ├──▶ State.data = ...    ← update data
       │
       └──▶ Renderer.render()   ← redraw the page
```

## Who depends on whom

```
index.js  →  app.js
app.js    →  state.js  api.js  renderer.js  utils.js
api.js    →  utils.js
renderer.js  →  utils.js
state.js  →  (nothing)
utils.js  →  (nothing)
```

## Rules

- **State** never touches the DOM or calls fetch
- **ApiClient** never touches the DOM or mutates State
- **Renderer** reads State, never writes to it, never calls fetch
- **App** is the only class that calls all three — it owns the flow

## Adding a new feature

1. Does it need new backend data? → Add method to `api.js`
2. Does it need new stored data? → Add field to `State` in `state.js`
3. Does it need new UI? → Add method to `Renderer` in `renderer.js`
4. Does it change the user flow? → Add method to `App` in `app.js`

## Inline handlers

Because ES Modules don't expose functions globally, inline `onclick` and
`onerror` attributes are **not used**. Instead, buttons use `data-action`
attributes and event listeners are attached in `renderer.js`.
