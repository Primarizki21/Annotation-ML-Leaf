/**
 * index.js — Entry point for the annotation page.
 *
 * Imports the App class and boots the application.
 * This is the only file loaded directly by the HTML page.
 * All other modules are loaded via import statements.
 *
 * @imports { App } from './app.js'
 * @exports (none — this is the root module)
 */

import { App } from './app.js';

const app = new App();
app.init();
