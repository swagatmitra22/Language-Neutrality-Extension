chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "analyzeBias",
    title: "Analyze text for bias",
    contexts: ["selection"]
  });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId === "analyzeBias" && info.selectionText) {
    const selectedText = info.selectionText.trim();

    // Retrieve settings
    const s = await chrome.storage.sync.get({
      apiBase: "http://localhost:8000",
      threshold: 0.4
    });

    try {
      // Step 1: Analyze bias
      const analyzeRes = await fetch(`${s.apiBase}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: selectedText, threshold: s.threshold })
      });
      const analyzeData = await analyzeRes.json();

      const bias = analyzeData.prediction.predicted_class;
      const conf = analyzeData.prediction.confidence;
      const probs = analyzeData.prediction.all_probabilities;

      let suggestion = "";

      // Step 2: Only get suggestion if bias ≠ Neutral
      if (bias !== "Neutral") {
        const suggestRes = await fetch(`${s.apiBase}/suggest`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sentence: selectedText,
            bias_type: bias,
            confidence: conf
          })
        });
        const suggestData = await suggestRes.json();
        suggestion = suggestData.suggestion || "";
      }

      // Step 3: Inject popup into the page
      chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: showDialog,
        args: [selectedText, bias, conf, probs, suggestion]
      });
    } catch (err) {
      console.error("Error analyzing text:", err);
    }
  }
});

// Function injected into the page
function showDialog(text, bias, conf, probs, suggestion) {
  const existing = document.querySelector(".ln-dialog");
  if (existing) existing.remove();

  const dialog = document.createElement("div");
  dialog.className = "ln-dialog";
  Object.assign(dialog.style, {
    position: "fixed",
    top: "50%",
    left: "50%",
    transform: "translate(-50%, -50%)",
    background: "#fff",
    color: "#000",
    padding: "20px",
    borderRadius: "12px",
    boxShadow: "0 6px 30px rgba(0,0,0,0.4)",
    zIndex: 999999,
    width: "420px",
    maxWidth: "90%"
  });

  const title = document.createElement("h3");
  title.innerText = `Detected Bias: ${bias}`;
  dialog.appendChild(title);

  const confLine = document.createElement("div");
  confLine.innerText = `Confidence: ${(conf * 100).toFixed(1)}%`;
  dialog.appendChild(confLine);

  const ul = document.createElement("ul");
  ul.style.listStyleType = "none";
  ul.style.padding = "0";
  for (const [k, v] of Object.entries(probs)) {
    const li = document.createElement("li");
    li.innerText = `${k}: ${(v * 100).toFixed(1)}%`;
    ul.appendChild(li);
  }
  dialog.appendChild(ul);

  // If it's not neutral, show suggestion
  if (bias !== "Neutral") {
    const suggestionDiv = document.createElement("div");
    suggestionDiv.innerHTML = `<strong>Suggested Neutral:</strong><br>${suggestion || "(no suggestion)"}`;
    dialog.appendChild(suggestionDiv);
  } else {
    const neutralMsg = document.createElement("div");
    neutralMsg.innerHTML = `<em>No bias detected — the text appears neutral.</em>`;
    neutralMsg.style.marginTop = "10px";
    neutralMsg.style.color = "#555";
    dialog.appendChild(neutralMsg);
  }

  const closeBtn = document.createElement("button");
  closeBtn.innerText = "Close";
  closeBtn.style.marginTop = "12px";
  closeBtn.style.background = "#ffcc00";
  closeBtn.style.border = "none";
  closeBtn.style.padding = "6px 12px";
  closeBtn.style.borderRadius = "6px";
  closeBtn.style.cursor = "pointer";
  closeBtn.addEventListener("click", () => dialog.remove());
  dialog.appendChild(closeBtn);

  document.body.appendChild(dialog);
}
