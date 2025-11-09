// Simple relay for future enhancements; currently not intercepting
chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.sync.set({ enabled: true, threshold: 0.4, apiBase: 'http://localhost:8000' });
});
