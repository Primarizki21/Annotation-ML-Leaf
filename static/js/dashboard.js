var refreshTimer = null;

function loadData() {
    fetch('/api/dashboard-data')
        .then(function(res) { return res.json(); })
        .then(function(data) {
            renderDashboard(data);
        })
        .catch(function(err) {
            console.error('Dashboard error:', err);
            document.getElementById('content').innerHTML =
                '<div class="loading">Failed to load dashboard data</div>';
        });
}

function renderDashboard(data) {
    var annotators = data.annotators;
    var classes = data.classes;
    var overall = data.overall;
    var labels = data.labels;
    var alRounds = data.al_rounds || [];
    var totalLabels = labels.healthy + labels.unhealthy;
    var healthyPct = totalLabels > 0 ? (labels.healthy / totalLabels * 100).toFixed(1) : 0;
    var unhealthyPct = totalLabels > 0 ? (labels.unhealthy / totalLabels * 100).toFixed(1) : 0;

    var html = '';

    // Overall section
    html +=
        '<div class="section">' +
            '<h2>Overall Progress</h2>' +
            '<div class="overall-grid">' +
                '<div class="stat-card">' +
                    '<div class="stat-value">' + overall.done.toLocaleString() + '</div>' +
                    '<div class="stat-label">Annotated</div>' +
                '</div>' +
                '<div class="stat-card">' +
                    '<div class="stat-value">' + overall.total.toLocaleString() + '</div>' +
                    '<div class="stat-label">Total Patches</div>' +
                '</div>' +
                '<div class="stat-card">' +
                    '<div class="stat-value">' + overall.skipped.toLocaleString() + '</div>' +
                    '<div class="stat-label">Skipped</div>' +
                '</div>' +
                '<div class="stat-card">' +
                    '<div class="stat-value">' + overall.percent + '%</div>' +
                    '<div class="stat-label">Complete</div>' +
                '</div>' +
            '</div>' +
            '<div class="progress-bar-bg">' +
                '<div class="progress-bar-fill" style="width: ' + overall.percent + '%"></div>' +
            '</div>' +
            '<div class="progress-label">' + overall.done.toLocaleString() + ' of ' + overall.total.toLocaleString() + ' patches annotated</div>' +
        '</div>';

    // Label distribution
    html +=
        '<div class="section">' +
            '<h2>Label Distribution</h2>' +
            '<div class="label-grid">' +
                '<div class="label-card healthy">' +
                    '<div class="label-value">' + labels.healthy.toLocaleString() + '</div>' +
                    '<div class="label-pct">Healthy (' + healthyPct + '%)</div>' +
                '</div>' +
                '<div class="label-card unhealthy">' +
                    '<div class="label-value">' + labels.unhealthy.toLocaleString() + '</div>' +
                    '<div class="label-pct">Unhealthy (' + unhealthyPct + '%)</div>' +
                '</div>' +
            '</div>' +
        '</div>';

    // Per-annotator table
    if (annotators.length > 0) {
        html +=
            '<div class="section">' +
                '<h2>Per-Annotator Progress</h2>' +
                '<div class="table-container">' +
                    '<table>' +
                        '<thead>' +
                            '<tr>' +
                                '<th>Annotator</th>' +
                                '<th>Assigned</th>' +
                                '<th>Done</th>' +
                                '<th>Skipped</th>' +
                                '<th class="bar-cell">Progress</th>' +
                            '</tr>' +
                        '</thead>' +
                        '<tbody>';

        annotators.forEach(function(a) {
            html +=
                '<tr>' +
                    '<td>' + escapeHtml(a.name) + '</td>' +
                    '<td>' + a.assigned.toLocaleString() + '</td>' +
                    '<td>' + a.done.toLocaleString() + '</td>' +
                    '<td>' + a.skipped.toLocaleString() + '</td>' +
                    '<td class="bar-cell">' +
                        '<div class="mini-bar-bg">' +
                            '<div class="mini-bar-fill" style="width: ' + a.percent + '%"></div>' +
                        '</div>' +
                        '<div class="pct-text">' + a.percent + '%</div>' +
                    '</td>' +
                '</tr>';
        });

        html +=
                        '</tbody>' +
                    '</table>' +
                '</div>' +
            '</div>';
    }

    // Per-class table
    if (classes.length > 0) {
        html +=
            '<div class="section">' +
                '<h2>Per-Class Progress</h2>' +
                '<div class="table-container">' +
                    '<table>' +
                        '<thead>' +
                            '<tr>' +
                                '<th>Class Name</th>' +
                                '<th>Total</th>' +
                                '<th>Done</th>' +
                                '<th class="bar-cell">Progress</th>' +
                            '</tr>' +
                        '</thead>' +
                        '<tbody>';

        classes.forEach(function(c) {
            html +=
                '<tr>' +
                    '<td>' + escapeHtml(c.name) + '</td>' +
                    '<td>' + c.total.toLocaleString() + '</td>' +
                    '<td>' + c.done.toLocaleString() + '</td>' +
                    '<td class="bar-cell">' +
                        '<div class="mini-bar-bg">' +
                            '<div class="mini-bar-fill" style="width: ' + c.percent + '%"></div>' +
                        '</div>' +
                        '<div class="pct-text">' + c.percent + '%</div>' +
                    '</td>' +
                '</tr>';
        });

        html +=
                        '</tbody>' +
                    '</table>' +
                '</div>' +
            '</div>';
    }

    // Per-round AL progress (Phase 3.2)
    if (alRounds && alRounds.length > 0) {
        alRounds.forEach(function(al) {
            var roundN = al.round;
            var alAnn = al.annotators || [];
            var alDone = alAnn.reduce(function(sum, a) { return sum + a.done; }, 0);
            var alSkipped = alAnn.reduce(function(sum, a) { return sum + a.skipped; }, 0);
            var alAssigned = alAnn.reduce(function(sum, a) { return sum + a.assigned; }, 0);
            var alPct = alAssigned > 0 ? (alDone / alAssigned * 100).toFixed(1) : 0;
            html +=
                '<div class="section">' +
                    '<h2>Active Learning — Round ' + roundN + '</h2>' +
                    '<div class="overall-grid">' +
                        '<div class="stat-card">' +
                            '<div class="stat-value">' + alAssigned.toLocaleString() + '</div>' +
                            '<div class="stat-label">Total Patches (×' + alAnn.length + ' annotators)</div>' +
                        '</div>' +
                        '<div class="stat-card">' +
                            '<div class="stat-value">' + alDone.toLocaleString() + '</div>' +
                            '<div class="stat-label">Annotated</div>' +
                        '</div>' +
                        '<div class="stat-card">' +
                            '<div class="stat-value">' + alSkipped.toLocaleString() + '</div>' +
                            '<div class="stat-label">Skipped</div>' +
                        '</div>' +
                        '<div class="stat-card">' +
                            '<div class="stat-value">' + alPct + '%</div>' +
                            '<div class="stat-label">Complete</div>' +
                        '</div>' +
                    '</div>' +
                    '<div class="progress-bar-bg">' +
                        '<div class="progress-bar-fill" style="width: ' + alPct + '%"></div>' +
                    '</div>' +
                    '<div class="progress-label">' + alDone.toLocaleString() + ' of ' + alAssigned.toLocaleString() + ' patches annotated</div>';

            if (alAnn.length > 0) {
                html +=
                    '<div class="table-container" style="margin-top: 16px;">' +
                        '<table>' +
                            '<thead>' +
                                '<tr>' +
                                    '<th>Annotator</th>' +
                                    '<th>Assigned</th>' +
                                    '<th>Done</th>' +
                                    '<th>Skipped</th>' +
                                    '<th class="bar-cell">Progress</th>' +
                                '</tr>' +
                            '</thead>' +
                            '<tbody>';
                alAnn.forEach(function(a) {
                    html +=
                        '<tr>' +
                            '<td>' + escapeHtml(a.name) + '</td>' +
                            '<td>' + a.assigned.toLocaleString() + '</td>' +
                            '<td>' + a.done.toLocaleString() + '</td>' +
                            '<td>' + a.skipped.toLocaleString() + '</td>' +
                            '<td class="bar-cell">' +
                                '<div class="mini-bar-bg">' +
                                    '<div class="mini-bar-fill" style="width: ' + a.percent + '%"></div>' +
                                '</div>' +
                                '<div class="pct-text">' + a.percent + '%</div>' +
                            '</td>' +
                        '</tr>';
                });
                html +=
                            '</tbody>' +
                        '</table>' +
                    '</div>';
            }

            html += '</div>';
        });
    }

    document.getElementById('content').innerHTML = html;
}

function escapeHtml(text) {
    var div = document.createElement('div');
    div.appendChild(document.createTextNode(text));
    return div.innerHTML;
}

// Auto-refresh every 30 seconds
function startAutoRefresh() {
    if (refreshTimer) clearInterval(refreshTimer);
    refreshTimer = setInterval(loadData, 30000);
}

// Initialize
loadData();
startAutoRefresh();
