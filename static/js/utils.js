/**
 * Utils — Shared helper functions used across modules.
 *
 * All functions are stateless and side-effect-free (except handleImageError
 * which mutates the DOM element's src as intended).
 *
 * @exports { getLeafStem, escapeHtml, handleImageError }
 */

export function getLeafStem(patchPath) {
    var filename = patchPath.split('/').pop();
    return filename.replace(/__r\d+_c\d+\.\w+$/, '');
}

export function escapeHtml(text) {
    var div = document.createElement('div');
    div.appendChild(document.createTextNode(text));
    return div.innerHTML;
}

export function handleImageError(img) {
    img.src = 'data:image/svg+xml,' + encodeURIComponent(
        '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64">' +
        '<rect width="64" height="64" fill="#333"/>' +
        '<text x="32" y="30" text-anchor="middle" fill="#888" font-size="9">Missing</text>' +
        '<text x="32" y="42" text-anchor="middle" fill="#888" font-size="9">Image</text>' +
        '</svg>'
    );
}
