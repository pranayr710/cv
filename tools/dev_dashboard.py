import os
import sys
import json
import http.server
import socketserver
import subprocess
from urllib.parse import urlparse

# --- Pytest Collector Subprocess Entrypoint ---
if len(sys.argv) > 1 and sys.argv[1] == "--run-pytest-internal":
    import pytest
    class JSONCollector:
        def __init__(self):
            self.results = {}
        
        def pytest_itemcollected(self, item):
            self.results[item.nodeid] = {
                "nodeid": item.nodeid,
                "name": item.name,
                "file": item.location[0],
                "outcome": "pending",
                "duration": 0,
                "error": None
            }
            
        def pytest_runtest_logreport(self, report):
            if report.when == "call" or (report.when == "setup" and report.outcome == "skipped"):
                err = None
                if report.failed:
                    if isinstance(report.longrepr, str):
                        err = report.longrepr
                    else:
                        err = str(report.longrepr)
                
                # Retrieve or create entry
                self.results[report.nodeid] = {
                    "nodeid": report.nodeid,
                    "name": report.nodeid.split("::")[-1],
                    "file": report.nodeid.split("::")[0],
                    "outcome": report.outcome,
                    "duration": report.duration,
                    "error": err
                }

    collector = JSONCollector()
    pytest_args = ["-q"] + sys.argv[2:]
    pytest.main(pytest_args, plugins=[collector])
    print("---JSON_START---")
    print(json.dumps(list(collector.results.values())))
    print("---JSON_END---")
    sys.exit(0)

# --- HTTP Request Handler ---
HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ClassGraph Test Suite Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #0b0f19;
            --bg-card: rgba(17, 24, 39, 0.7);
            --border-color: rgba(255, 255, 255, 0.08);
            --primary: #8b5cf6;
            --primary-glow: rgba(139, 92, 246, 0.4);
            --success: #10b981;
            --success-glow: rgba(16, 185, 129, 0.2);
            --danger: #f43f5e;
            --danger-glow: rgba(244, 63, 94, 0.2);
            --warning: #f59e0b;
            --text-main: #f9fafb;
            --text-muted: #9ca3af;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Outfit', sans-serif;
            background: radial-gradient(circle at 50% 0%, #1e1b4b 0%, var(--bg-dark) 70%);
            color: var(--text-main);
            min-height: 100vh;
            padding: 2.5rem 1.5rem;
            overflow-x: hidden;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        /* Header Style */
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 3rem;
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            padding: 1.5rem 2rem;
            border-radius: 16px;
            border: 1px solid var(--border-color);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        }

        .header-title h1 {
            font-size: 1.8rem;
            font-weight: 700;
            background: linear-gradient(135deg, #a78bfa, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.25rem;
        }

        .header-title p {
            font-size: 0.9rem;
            color: var(--text-muted);
        }

        .status-badge {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.85rem;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.2);
            padding: 0.5rem 1rem;
            border-radius: 99px;
            color: var(--success);
            font-weight: 500;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background-color: var(--success);
            border-radius: 50%;
            box-shadow: 0 0 10px var(--success);
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }

        /* Stats Grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.5rem;
            margin-bottom: 3rem;
        }

        .stat-card {
            background: var(--bg-card);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
            position: relative;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
            transition: transform 0.3s ease, border-color 0.3s ease;
        }

        .stat-card:hover {
            transform: translateY(-2px);
            border-color: rgba(255, 255, 255, 0.15);
        }

        .stat-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: var(--primary);
        }

        .stat-card.success::before { background: var(--success); }
        .stat-card.danger::before { background: var(--danger); }
        .stat-card.warning::before { background: var(--warning); }

        .stat-label {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }

        .stat-value {
            font-size: 2.2rem;
            font-weight: 700;
            line-height: 1;
        }

        .stat-sub {
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-top: 0.5rem;
        }

        /* Action bar */
        .action-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
            gap: 1rem;
            flex-wrap: wrap;
        }

        .filter-tabs {
            display: flex;
            gap: 0.5rem;
            background: rgba(255, 255, 255, 0.03);
            padding: 0.25rem;
            border-radius: 10px;
            border: 1px solid var(--border-color);
        }

        .tab-btn {
            background: transparent;
            border: none;
            color: var(--text-muted);
            padding: 0.5rem 1.25rem;
            border-radius: 8px;
            cursor: pointer;
            font-family: inherit;
            font-size: 0.9rem;
            font-weight: 500;
            transition: all 0.2s ease;
        }

        .tab-btn:hover {
            color: var(--text-main);
        }

        .tab-btn.active {
            background: var(--primary);
            color: var(--text-main);
            box-shadow: 0 4px 12px var(--primary-glow);
        }

        .search-input {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-color);
            padding: 0.75rem 1.25rem;
            border-radius: 12px;
            color: var(--text-main);
            font-family: inherit;
            font-size: 0.9rem;
            width: 280px;
            transition: all 0.2s ease;
        }

        .search-input:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px var(--primary-glow);
        }

        .run-btn {
            background: linear-gradient(135deg, #8b5cf6, #6366f1);
            color: var(--text-main);
            border: none;
            padding: 0.8rem 2rem;
            border-radius: 12px;
            font-weight: 600;
            font-family: inherit;
            font-size: 0.95rem;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            box-shadow: 0 4px 15px var(--primary-glow);
            transition: all 0.2s ease;
        }

        .run-btn:hover:not(:disabled) {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px var(--primary-glow);
        }

        .run-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        /* Test Suite Accordion/Group */
        .suite-group {
            background: var(--bg-card);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            margin-bottom: 1.5rem;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        }

        .suite-header {
            padding: 1.25rem 1.5rem;
            background: rgba(255, 255, 255, 0.02);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
            user-select: none;
        }

        .suite-title {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            font-weight: 600;
            font-size: 1.05rem;
        }

        .suite-badge {
            font-size: 0.75rem;
            padding: 0.25rem 0.5rem;
            border-radius: 6px;
            background: rgba(255, 255, 255, 0.08);
            color: var(--text-muted);
        }

        .test-list {
            padding: 0.5rem 0;
        }

        .test-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 1.5rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.03);
            transition: background 0.2s ease;
        }

        .test-row:last-child {
            border-bottom: none;
        }

        .test-row:hover {
            background: rgba(255, 255, 255, 0.01);
        }

        .test-info {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .status-icon {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.75rem;
            font-weight: bold;
        }

        .status-icon.passed {
            background: rgba(16, 185, 129, 0.1);
            color: var(--success);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .status-icon.failed {
            background: rgba(244, 63, 94, 0.1);
            color: var(--danger);
            border: 1px solid rgba(244, 63, 94, 0.3);
        }

        .status-icon.skipped {
            background: rgba(245, 158, 11, 0.1);
            color: var(--warning);
            border: 1px solid rgba(245, 158, 11, 0.3);
        }

        .status-icon.pending {
            background: rgba(255, 255, 255, 0.05);
            color: var(--text-muted);
            border: 1px solid var(--border-color);
        }

        .test-name {
            font-weight: 500;
            font-size: 0.95rem;
        }

        .test-meta {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .test-duration {
            font-size: 0.85rem;
            color: var(--text-muted);
        }

        .view-err-btn {
            background: rgba(244, 63, 94, 0.1);
            border: 1px solid rgba(244, 63, 94, 0.3);
            color: var(--danger);
            padding: 0.35rem 0.75rem;
            border-radius: 6px;
            font-size: 0.8rem;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.2s ease;
        }

        .view-err-btn:hover {
            background: var(--danger);
            color: var(--text-main);
        }

        /* Traceback Drawer */
        .drawer-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.7);
            backdrop-filter: blur(4px);
            z-index: 100;
            display: none;
            opacity: 0;
            transition: opacity 0.3s ease;
        }

        .drawer {
            position: fixed;
            top: 0;
            right: -600px;
            width: 600px;
            max-width: 90%;
            height: 100%;
            background: #111827;
            border-left: 1px solid var(--border-color);
            box-shadow: -10px 0 30px rgba(0, 0, 0, 0.5);
            z-index: 101;
            padding: 2rem;
            display: flex;
            flex-direction: column;
            transition: right 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .drawer.open {
            right: 0;
        }

        .drawer-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 1rem;
        }

        .drawer-title {
            font-size: 1.2rem;
            font-weight: 600;
            color: var(--danger);
        }

        .close-drawer {
            background: transparent;
            border: none;
            color: var(--text-muted);
            font-size: 1.5rem;
            cursor: pointer;
        }

        .drawer-body {
            flex: 1;
            overflow-y: auto;
            background: #030712;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            padding: 1rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            white-space: pre-wrap;
            color: #f3f4f6;
        }

        /* Spinner */
        .spinner {
            width: 16px;
            height: 16px;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .empty-state {
            text-align: center;
            padding: 3rem;
            color: var(--text-muted);
            font-size: 1.1rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="header-title">
                <h1>ClassGraph Pipeline Testing</h1>
                <p>Verifying Stage 1 to Stage 4 Perception &amp; Interaction Pipelines</p>
            </div>
            <div class="status-badge">
                <div class="status-dot"></div>
                <span>Server Connected</span>
            </div>
        </header>

        <!-- Summary Statistics -->
        <div class="stats-grid">
            <div class="stat-card" id="card-rate">
                <div class="stat-label">Pass Rate</div>
                <div class="stat-value" id="val-rate">--</div>
                <div class="stat-sub" id="val-rate-sub">Fetching test list...</div>
            </div>
            <div class="stat-card success" id="card-passed">
                <div class="stat-label">Passed</div>
                <div class="stat-value" id="val-passed">0</div>
                <div class="stat-sub">Valid test cases</div>
            </div>
            <div class="stat-card danger" id="card-failed">
                <div class="stat-label">Failed</div>
                <div class="stat-value" id="val-failed">0</div>
                <div class="stat-sub">Errors to resolve</div>
            </div>
            <div class="stat-card warning" id="card-skipped">
                <div class="stat-label">Skipped / Total</div>
                <div class="stat-value" id="val-skipped-total">0 / 0</div>
                <div class="stat-sub" id="val-duration">Duration: --</div>
            </div>
        </div>

        <!-- Controls Action Bar -->
        <div class="action-bar">
            <div class="filter-tabs">
                <button class="tab-btn active" onclick="setFilter('all')">All</button>
                <button class="tab-btn" onclick="setFilter('passed')">Passed</button>
                <button class="tab-btn" onclick="setFilter('failed')">Failed</button>
                <button class="tab-btn" onclick="setFilter('skipped')">Skipped</button>
                <button class="tab-btn" onclick="setFilter('pending')">Pending</button>
            </div>
            <input type="text" class="search-input" placeholder="Search test cases..." oninput="handleSearch(this.value)">
            <button class="run-btn" id="run-btn" onclick="runTests()">
                <span>Run Test Suite</span>
            </button>
        </div>

        <!-- Test Groups Accordion -->
        <div id="test-suites-container">
            <div class="empty-state">Loading test suite structure...</div>
        </div>
    </div>

    <!-- Error Drawer -->
    <div class="drawer-overlay" id="drawer-overlay" onclick="closeErrorDrawer()"></div>
    <div class="drawer" id="drawer">
        <div class="drawer-header">
            <div class="drawer-title">Test Failure Traceback</div>
            <button class="close-drawer" onclick="closeErrorDrawer()">&times;</button>
        </div>
        <div class="drawer-body" id="drawer-body"></div>
    </div>

    <script>
        let testData = [];
        let currentFilter = 'all';
        let searchQuery = '';

        window.addEventListener('DOMContentLoaded', async () => {
            await fetchTests();
        });

        async function fetchTests() {
            try {
                const res = await fetch('/api/tests');
                if (!res.ok) throw new Error('Failed to fetch tests');
                const data = await res.json();
                testData = data;
                updateStats(data);
                renderSuites();
            } catch (err) {
                console.error(err);
                document.getElementById('test-suites-container').innerHTML = 
                    `<div class="empty-state" style="color: var(--danger)">Error loading tests: ${err.message}</div>`;
            }
        }

        function setFilter(filter) {
            currentFilter = filter;
            document.querySelectorAll('.tab-btn').forEach(btn => {
                btn.classList.toggle('active', btn.textContent.toLowerCase() === filter);
            });
            renderSuites();
        }

        function handleSearch(val) {
            searchQuery = val.toLowerCase();
            renderSuites();
        }

        function updateStats(results) {
            const total = results.length;
            const passed = results.filter(r => r.outcome === 'passed').length;
            const failed = results.filter(r => r.outcome === 'failed').length;
            const skipped = results.filter(r => r.outcome === 'skipped').length;
            const pending = results.filter(r => r.outcome === 'pending').length;
            const duration = results.reduce((acc, r) => acc + (r.duration || 0), 0);

            const activeRun = total - skipped - pending;
            const passRate = activeRun > 0 ? Math.round((passed / activeRun) * 100) : 0;

            document.getElementById('val-rate').textContent = activeRun > 0 ? `${passRate}%` : '--';
            document.getElementById('val-rate-sub').textContent = pending > 0 ? `${pending} tests pending execution` : `Based on ${activeRun} active tests`;
            document.getElementById('val-passed').textContent = passed;
            document.getElementById('val-failed').textContent = failed;
            document.getElementById('val-skipped-total').textContent = `${skipped} / ${total}`;
            document.getElementById('val-duration').textContent = pending > 0 ? `Not started` : `Duration: ${duration.toFixed(2)}s`;

            const rateCard = document.getElementById('card-rate');
            rateCard.className = 'stat-card';
            if (failed > 0) rateCard.classList.add('danger');
            else if (passed > 0 && pending === 0) rateCard.classList.add('success');
        }

        function renderSuites() {
            const container = document.getElementById('test-suites-container');
            container.innerHTML = '';

            // Group by file
            const groups = {};
            testData.forEach(item => {
                const matchFilter = currentFilter === 'all' || item.outcome === currentFilter;
                const matchSearch = item.name.toLowerCase().includes(searchQuery) || item.file.toLowerCase().includes(searchQuery);
                if (matchFilter && matchSearch) {
                    if (!groups[item.file]) groups[item.file] = [];
                    groups[item.file].push(item);
                }
            });

            const files = Object.keys(groups).sort();
            if (files.length === 0) {
                container.innerHTML = `<div class="empty-state">No test cases match the active filter/search.</div>`;
                return;
            }

            files.forEach(file => {
                const items = groups[file];
                const passedCount = items.filter(r => r.outcome === 'passed').length;
                const failedCount = items.filter(r => r.outcome === 'failed').length;
                const skippedCount = items.filter(r => r.outcome === 'skipped').length;
                const pendingCount = items.filter(r => r.outcome === 'pending').length;

                const groupDiv = document.createElement('div');
                groupDiv.className = 'suite-group';

                const header = document.createElement('div');
                header.className = 'suite-header';
                header.innerHTML = `
                    <div class="suite-title">
                        <span>${file}</span>
                        <span class="suite-badge">${items.length} tests</span>
                    </div>
                    <div style="display: flex; gap: 0.5rem; font-size: 0.9rem;">
                        ${passedCount > 0 ? `<span style="color: var(--success); font-weight: 600;">✓ ${passedCount}</span>` : ''}
                        ${failedCount > 0 ? `<span style="color: var(--danger); font-weight: 600;">✗ ${failedCount}</span>` : ''}
                        ${skippedCount > 0 ? `<span style="color: var(--warning); font-weight: 600;">⊘ ${skippedCount}</span>` : ''}
                        ${pendingCount > 0 ? `<span style="color: var(--text-muted); font-weight: 600;">⏰ ${pendingCount}</span>` : ''}
                    </div>
                `;

                const list = document.createElement('div');
                list.className = 'test-list';

                items.forEach(test => {
                    const row = document.createElement('div');
                    row.className = 'test-row';

                    let iconChar = '✓';
                    if (test.outcome === 'failed') iconChar = '✗';
                    if (test.outcome === 'skipped') iconChar = '⊘';
                    if (test.outcome === 'pending') iconChar = '⏰';

                    row.innerHTML = `
                        <div class="test-info">
                            <div class="status-icon ${test.outcome}">${iconChar}</div>
                            <span class="test-name">${test.name}</span>
                        </div>
                        <div class="test-meta">
                            <span class="test-duration">${test.outcome === 'pending' ? '--' : (test.duration * 1000).toFixed(0) + ' ms'}</span>
                            ${test.outcome === 'failed' ? `<button class="view-err-btn" onclick="showErrorDrawer('${escapeHtml(test.error)}')">Traceback</button>` : ''}
                        </div>
                    `;
                    list.appendChild(row);
                });

                groupDiv.appendChild(header);
                groupDiv.appendChild(list);
                container.appendChild(groupDiv);
            });
        }

        function escapeHtml(str) {
            if (!str) return '';
            return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
        }

        function showErrorDrawer(err) {
            document.getElementById('drawer-body').textContent = err;
            document.getElementById('drawer-overlay').style.display = 'block';
            setTimeout(() => {
                document.getElementById('drawer-overlay').style.opacity = '1';
                document.getElementById('drawer').classList.add('open');
            }, 10);
        }

        function closeErrorDrawer() {
            document.getElementById('drawer').classList.remove('open');
            document.getElementById('drawer-overlay').style.opacity = '0';
            setTimeout(() => {
                document.getElementById('drawer-overlay').style.display = 'none';
            }, 300);
        }

        async function runTests() {
            const btn = document.getElementById('run-btn');
            btn.disabled = true;
            btn.innerHTML = `<div class="spinner"></div><span>Running...</span>`;

            try {
                const res = await fetch('/api/run', { method: 'POST' });
                if (!res.ok) throw new Error('Failed to execute tests');
                const data = await res.json();
                testData = data;
                updateStats(data);
                renderSuites();
            } catch (err) {
                alert('Error running tests: ' + err.message);
            } finally {
                btn.disabled = false;
                btn.innerHTML = 'Run Test Suite';
            }
        }
    </script>
</body>
</html>
"""

class TestUIRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(HTML_CONTENT.encode("utf-8"))))
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode("utf-8"))
        elif url.path == "/api/tests":
            try:
                cmd = [sys.executable, __file__, "--run-pytest-internal", "--collect-only"]
                output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=30).decode("utf-8")
                
                # Parse JSON block from output
                lines = output.splitlines()
                json_str = ""
                in_json = False
                for line in lines:
                    if line == "---JSON_START---":
                        in_json = True
                        continue
                    if line == "---JSON_END---":
                        in_json = False
                        break
                    if in_json:
                        json_str += line + "\n"
                
                if json_str:
                    res_data = json.loads(json_str)
                else:
                    res_data = []
                
                response_body = json.dumps(res_data).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)
            except Exception as e:
                self.send_error(500, str(e))
        else:
            self.send_error(404, "File not found")

    def do_POST(self):
        url = urlparse(self.path)
        if url.path == "/api/run":
            try:
                cmd = [sys.executable, __file__, "--run-pytest-internal"]
                output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=60).decode("utf-8")
                
                # Parse JSON block from output
                lines = output.splitlines()
                json_str = ""
                in_json = False
                for line in lines:
                    if line == "---JSON_START---":
                        in_json = True
                        continue
                    if line == "---JSON_END---":
                        in_json = False
                        break
                    if in_json:
                        json_str += line + "\n"
                
                if json_str:
                    res_data = json.loads(json_str)
                else:
                    res_data = [{"nodeid": "error", "name": "Execution failure", "file": "system", "outcome": "failed", "duration": 0, "error": output}]
                
                response_body = json.dumps(res_data).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)
            except Exception as e:
                err_msg = f"Failed to run tests: {e}"
                res_data = [{"nodeid": "error", "name": "Execution failure", "file": "system", "outcome": "failed", "duration": 0, "error": err_msg}]
                response_body = json.dumps(res_data).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)
        else:
            self.send_error(404, "API endpoint not found")

# --- Start Server ---
def main():
    import argparse
    parser = argparse.ArgumentParser(description="ClassGraph Test Suite Dashboard UI")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the UI server on")
    args = parser.parse_args()

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", args.port), TestUIRequestHandler) as httpd:
        print(f"=========================================================")
        print(f" ClassGraph Test Suite Verification Dashboard ")
        print(f" Serving at: http://localhost:{args.port}/")
        print(f" Press Ctrl+C to stop ")
        print(f"=========================================================")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")

if __name__ == "__main__":
    main()
