const enabledEl = document.getElementById('enabled');
const thresholdEl = document.getElementById('threshold');
const apiBaseEl = document.getElementById('apiBase');
const saveBtn = document.getElementById('save');

chrome.storage.sync.get(['enabled','threshold','apiBase'], ({enabled, threshold, apiBase}) => {
  enabledEl.checked = enabled ?? true;
  thresholdEl.value = threshold ?? 0.4;
  apiBaseEl.value = apiBase ?? 'http://localhost:8000';
});

saveBtn.addEventListener('click', () => {
  chrome.storage.sync.set({
    enabled: enabledEl.checked,
    threshold: parseFloat(thresholdEl.value),
    apiBase: apiBaseEl.value.trim()
  });
});
