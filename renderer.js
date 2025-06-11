const API_BASE_URL = 'http://localhost:8000';

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

// Function to handle API requests
async function callApi(endpoint, method = 'GET', body = null) {
    try {
        const options = {
            method: method,
            headers: {
                'Content-Type': 'application/json',
            },
        };
        if (body) {
            options.body = JSON.stringify(body);
        }

        const response = await fetch(`${API_BASE_URL}${endpoint}`, options);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || `HTTP error! status: ${response.status}`);
        }
        return data;
    } catch (error) {
        console.error(`Error calling ${endpoint}:`, error);
        throw error;
    }
}

// Function to refresh all machine statuses
async function refreshAllMachineStatuses() {
    updateStatus(metricsOutput, 'Fetching all machine statuses...', 'loading');
    machineStatusTableBody.innerHTML = ''; // Clear previous rows

    try {
        // 1. Get Local Machine Metrics
        const localMetrics = await callApi('/metrics');
        if (localMetrics.status === 'success') {
            addMachineRow(localMetrics);
        } else {
            updateStatus(metricsOutput, `Failed to get local machine metrics: ${localMetrics.message || 'Unknown error'}`, 'error');
        }

        // 2. Get All Connected Machines Status from a Master Server (if applicable)
        // For this demonstration, we'll assume a master server running on localhost:8001
        // In a real distributed setup, the master_server_ip and master_server_port would be configurable
        const master_server_ip = '127.0.0.1'; 
        const master_server_port = 8001; 

        try {
            const allMachinesStatus = await callApi(`/machines_status?master_server_ip=${master_server_ip}&master_server_port=${master_server_port}`);
            if (allMachinesStatus.status === 'success' && allMachinesStatus.machines) {
                allMachinesStatus.machines.forEach(machine => {
                    addMachineRow(machine);
                });
            } else {
                updateStatus(metricsOutput, `Failed to get connected machines status from master: ${allMachinesStatus.message || 'Unknown error'}`, 'error');
            }
        } catch (error) {
            updateStatus(metricsOutput, `Could not connect to master server at ${master_server_ip}:${master_server_port}. (This is expected if no master is running): ${error.message}`, 'error');
        }

        updateStatus(metricsOutput, 'All machine statuses refreshed.', 'success');
    } catch (error) {
        updateStatus(metricsOutput, `Error refreshing all statuses: ${error.message}`, 'error');
    }
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
    
    output += `<h3>Total URLs Requested: ${results.total_urls_to_crawl}</h3>`;

    if (results.per_url_results) {
        Object.keys(results.per_url_results).forEach(url => {
            const urlResult = results.per_url_results[url];
            output += `<h4>URL: ${url}</h4>`;
            output += `<div>Status: <span class="${urlResult.status === 'success' ? 'success' : 'error'}">${urlResult.status}</span></div>`;
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
             if (urlResult.metadata_file_location_note) {
                output += `<p>Metadata: ${urlResult.metadata_file_location_note}</p>`;
            }
        });
    }
    return output;
}

// Event listener for Start Crawl button
crawlButton.addEventListener('click', async () => {
    updateStatus(crawlStatus, 'Starting crawl...', 'loading');
    resultsOutput.innerHTML = '' // Changed to innerHTML
    const urls = [urlInput.value]; // FastAPI expects a list of URLs
    const output_dir = outputDirInput.value;
    const max_concurrency = parseInt(maxConcurrencyInput.value, 10);
    const max_depth = parseInt(maxDepthInput.value, 10);

    if (!urls[0] || !output_dir) {
        updateStatus(crawlStatus, 'Please enter a URL and output directory.', 'error');
        return;
    }

    try {
        const response = await callApi('/start_crawl', 'POST', {
            urls: urls,
            output_dir: output_dir,
            max_concurrency: max_concurrency,
            max_depth: max_depth,
        });
        updateStatus(crawlStatus, 'Crawl process initiated successfully! Check results below.', 'success');
        resultsOutput.innerHTML = formatCrawlResults(response); // Changed to innerHTML and formatted
    } catch (error) {
        updateStatus(crawlStatus, `Crawl failed: ${error.message}`, 'error');
        resultsOutput.innerHTML = `Error: ${error.message}`; // Changed to innerHTML
    }
});

// Event listener for Refresh All Status button
refreshStatusButton.addEventListener('click', refreshAllMachineStatuses);

// Initial load: refresh all statuses when the app starts
document.addEventListener('DOMContentLoaded', refreshAllMachineStatuses); 