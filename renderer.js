const API_BASE_URL = 'ws://localhost:8000/ws';

let websocket = null;

// DOM elements
const urlInput = document.getElementById('urlInput');
const outputDirInput = document.getElementById('outputDirInput');
const maxConcurrencyInput = document.getElementById('maxConcurrencyInput');
const maxDepthInput = document.getElementById('maxDepthInput');
const crawlButton = document.getElementById('crawlButton');
const crawlStatus = document.getElementById('crawlStatus');
const refreshStatusButton = document.getElementById('refreshStatusButton');
const machineStatusTableBody = document.querySelector('#machineStatusTable tbody');
const metricsOutput = document.getElementById('metricsOutput');
const resultsOutput = document.getElementById('results');

// Helper function to update status messages
function updateStatus(element, message, type = 'info') {
    element.innerHTML = message;
    element.className = `status-message ${type}`;
}

// Initialize WebSocket connection
function initializeWebSocket() {
    websocket = new WebSocket(API_BASE_URL);

    websocket.onopen = () => {
        console.log('WebSocket connection established');
        updateStatus(metricsOutput, 'Connected to WebSocket server', 'success');
        // Refresh machine statuses upon connection
        refreshAllMachineStatuses();
    };

    websocket.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleWebSocketMessage(data);
        } catch (error) {
            console.error('Error parsing WebSocket message:', error);
            updateStatus(crawlStatus, `Error parsing server message: ${error.message}`, 'error');
        }
    };

    websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        updateStatus(crawlStatus, 'WebSocket error occurred. Check console for details.', 'error');
    };

    websocket.onclose = () => {
        console.log('WebSocket connection closed');
        updateStatus(crawlStatus, 'WebSocket connection closed. Attempting to reconnect...', 'error');
        // Attempt to reconnect after a delay
        setTimeout(initializeWebSocket, 5000);
    };
}

// Handle incoming WebSocket messages
function handleWebSocketMessage(data) {
    const { type, message, payload, status, results, details } = data;

    switch (type) {
        case 'ack':
            updateStatus(crawlStatus, message, 'info');
            break;
        case 'crawl_progress':
            updateStatus(crawlStatus, message, 'loading');
            if (payload && payload.crawled_count && payload.queued_count) {
                updateStatus(crawlStatus, `${message} (${payload.crawled_count}/${payload.queued_count})`, 'loading');
            }
            break;
        case 'sitemap_progress':
            updateStatus(crawlStatus, message, payload?.error ? 'error' : 'info');
            break;
        case 'crawl_complete':
            if (status === 'success') {
                updateStatus(crawlStatus, 'Crawl process completed successfully!', 'success');
                resultsOutput.innerHTML = formatCrawlResults(results);
            } else {
                updateStatus(crawlStatus, `Crawl failed: ${message}`, 'error');
                resultsOutput.innerHTML = `Error: ${message}`;
            }
            break;
        case 'status_response':
            machineStatusTableBody.innerHTML = ''; // Clear previous rows
            addMachineRow(payload);
            updateStatus(metricsOutput, 'Machine status refreshed.', 'success');
            break;
        case 'error':
            updateStatus(crawlStatus, `Server error: ${message}`, 'error');
            if (details) {
                console.error('Error details:', details);
            }
            break;
        case 'info':
            updateStatus(crawlStatus, message, 'info');
            break;
        default:
            console.warn('Unknown WebSocket message type:', type);
    }
}

// Send a message via WebSocket
function sendWebSocketMessage(action, payload) {
    if (!websocket || websocket.readyState !== WebSocket.OPEN) {
        updateStatus(crawlStatus, 'WebSocket not connected. Please try again.', 'error');
        return false;
    }
    try {
        websocket.send(JSON.stringify({ action, payload }));
        return true;
    } catch (error) {
        console.error('Error sending WebSocket message:', error);
        updateStatus(crawlStatus, `Error sending message: ${error.message}`, 'error');
        return false;
    }
}

// Function to refresh all machine statuses
function refreshAllMachineStatuses() {
    updateStatus(metricsOutput, 'Fetching machine status...', 'loading');
    sendWebSocketMessage('get_status', {});
}

// Helper to add a row to the machine status table
function addMachineRow(machineData) {
    const row = machineStatusTableBody.insertRow();
    row.insertCell().textContent = machineData.machine_name || 'N/A';
    row.insertCell().textContent = machineData.crawling_status || 'N/A';
    row.insertCell().textContent = machineData.cpu_usage_percent ? `${machineData.cpu_usage_percent.toFixed(2)}%` : 'N/A';
    row.insertCell().textContent = machineData.memory_usage_percent ? `${machineData.memory_usage_percent.toFixed(2)}%` : 'N/A';
    row.insertCell().textContent = machineData.total_storage_gb ? machineData.total_storage_gb.toFixed(2) : 'N/A';
    row.insertCell().textContent = machineData.free_storage_gb ? machineData.free_storage_gb.toFixed(2) : 'N/A';
}

// Function to format crawl results
function formatCrawlResults(results) {
    let output = '<h2>Crawl Results Summary</h2>';
    if (results.status === 'processing') {
        output += '<p class="loading">Crawl is still in progress...</p>';
    } else if (results.status === 'completed') {
        output += '<p class="success">Crawl completed!</p>';
    } else if (results.status === 'error') {
        output += `<p class="error">Overall Crawl Error: ${results.message || 'Unknown error'}</p>`;
    }

    output += `<h3>Total URLs Requested: ${Object.keys(results.per_url_results || {}).length}</h3>`;

    if (results.per_url_results) {
        Object.keys(results.per_url_results).forEach(url => {
            const urlResult = results.per_url_results[url];
            output += `<h4>URL: ${url}</h4>`;
            output += `<div>Status: <span class="${urlResult.success.length > 0 ? 'success' : 'error'}">${urlResult.success.length > 0 ? 'success' : 'failed'}</span></div>`;
            if (urlResult.success && urlResult.success.length > 0) {
                output += '<p class="success">Successfully crawled URLs:</p><ul>';
                urlResult.success.forEach(sUrl => output += `<li>${sUrl}</li>`);
                output += '</ul>';
            }
            if (urlResult.failed && urlResult.failed.length > 0) {
                output += '<p class="error">Failed to crawl URLs:</p><ul>';
                urlResult.failed.forEach(fItem => {
                    output += `<li>URL: ${fItem.url || 'N/A'} - Error: ${fItem.error || 'Unknown error'}</li>`;
                });
                output += '</ul>';
            }
            if (urlResult.skipped_by_filter && urlResult.skipped_by_filter.length > 0) {
                output += '<p class="info">Skipped URLs:</p><ul>';
                urlResult.skipped_by_filter.forEach(skUrl => output += `<li>${skUrl}</li>`);
                output += '</ul>';
            }
            if (urlResult.message) {
                output += `<p>Message: ${urlResult.message}</p>`;
            }
            if (urlResult.sitemap_processing_results) {
                output += `<p>Sitemap Processing: <span class="${urlResult.sitemap_processing_results.status === 'success' ? 'success' : 'error'}">${urlResult.sitemap_processing_results.status}</span>`;
                if (urlResult.sitemap_processing_results.error) {
                    output += ` - Error: ${urlResult.sitemap_processing_results.error}`;
                }
                output += ` (Found: ${urlResult.sitemap_processing_results.sitemap_urls_found_with_valid_lastmod || 0} entries)</p>`;
            }
            if (urlResult.site_output_zip_file) {
                output += `<p>ZIP Archive: ${urlResult.site_output_zip_file}</p>`;
            }
            if (urlResult.metadata_file_location_note) {
                output += `<p>Metadata: ${urlResult.metadata_file_location_note}</p>`;
            }
        });
    }
    return output;
}

// Event listener for Start Crawl button
crawlButton.addEventListener('click', () => {
    updateStatus(crawlStatus, 'Starting crawl...', 'loading');
    resultsOutput.innerHTML = ''; // Clear previous results
    const urls = [urlInput.value];
    const output_dir = outputDirInput.value;
    const max_concurrency = parseInt(maxConcurrencyInput.value, 10);
    const max_depth = parseInt(maxDepthInput.value, 10);

    if (!urls[0] || !output_dir) {
        updateStatus(crawlStatus, 'Please enter a URL and output directory.', 'error');
        return;
    }

    const payload = {
        urls,
        output_dir,
        max_concurrency,
        max_depth,
    };

    const sent = sendWebSocketMessage('start_crawl', payload);
    if (sent) {
        updateStatus(crawlStatus, 'Crawl request sent. Awaiting server response...', 'loading');
    }
});

// Event listener for Refresh All Status button
refreshStatusButton.addEventListener('click', refreshAllMachineStatuses);

// Initialize WebSocket when the app starts
document.addEventListener('DOMContentLoaded', initializeWebSocket);