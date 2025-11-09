(function () {
  const ICON_PATH = chrome.runtime.getURL("assets/warn.png");
  let settings = { enabled: true, threshold: 0.4, apiBase: "http://localhost:8000" };
  let analyzing = false;
  let lastText = "";

  chrome.storage.sync.get(["enabled", "threshold", "apiBase"], (s) => Object.assign(settings, s));

  chrome.storage.onChanged.addListener((changes) => {
    if (changes.enabled) settings.enabled = changes.enabled.newValue;
    if (changes.threshold) settings.threshold = changes.threshold.newValue;
    if (changes.apiBase) settings.apiBase = changes.apiBase.newValue;
  });

  const observer = new MutationObserver(debounce(scanPage, 1200));
  observer.observe(document.documentElement, { subtree: true, childList: true, characterData: true });
  document.addEventListener("input", debounce(scanPage, 1200), true);

  function debounce(fn, ms) {
    let t;
    return (...a) => {
      clearTimeout(t);
      t = setTimeout(() => fn.apply(null, a), ms);
    };
  }

  function scanPage() {
    if (!settings.enabled || analyzing) return;
    const fields = getEditableFields();
    for (const el of fields) analyzeField(el);
  }

  function getEditableFields() {
    const inputs = Array.from(document.querySelectorAll('textarea, input[type="text"], input[type="search"]'));
    const editable = Array.from(document.querySelectorAll('[contenteditable="true"], [role="textbox"]'));
    return [...inputs, ...editable].filter(isVisible);
  }

  function isVisible(el) {
    const r = el.getBoundingClientRect();
    return r.width > 0 && r.height > 0;
  }

  async function analyzeField(el) {
    const text = readText(el).trim();
    if (!text || text === lastText) return;
    lastText = text;

    try {
      analyzing = true;
      const res = await fetch(`${settings.apiBase}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, threshold: settings.threshold }),
      });

      const data = await res.json();
      renderHighlight(el, text, data);
    } catch (e) {
      console.warn("Analyze error:", e);
    } finally {
      analyzing = false;
    }
  }

  function readText(el) {
    return el.value !== undefined ? el.value : el.innerText || "";
  }

  function renderHighlight(el, text, data) {
    clearDecorations(el);
    if (!data || !data.prediction) return;

    const bias = data.prediction.predicted_class;
    if (bias === "Neutral") return; // nothing to highlight

    const conf = data.prediction.confidence;
    const allProbs = data.prediction.all_probabilities;

    // Highlight the entire sentence
    const container = document.createElement("span");
    container.className = "ln-highlight-container";
    container.style.position = "relative";
    container.style.display = "inline-block";
    container.style.backgroundColor = "rgba(255, 200, 0, 0.35)";
    container.style.borderRadius = "4px";
    container.style.padding = "2px 4px";

    const textNode = document.createElement("span");
    textNode.innerText = text;
    container.appendChild(textNode);

    const icon = document.createElement("img");
    icon.src = ICON_PATH;
    icon.className = "ln-icon";
    icon.style.width = "18px";
    icon.style.height = "18px";
    icon.style.marginLeft = "6px";
    icon.style.cursor = "pointer";
    icon.style.verticalAlign = "middle";
    icon.addEventListener("click", (e) => {
      e.stopPropagation();
      showDialog(bias, conf, allProbs, text, el);
    });

    container.appendChild(icon);

    // Replace text visually (non-destructive)
    if (el.isContentEditable) {
      el.innerHTML = "";
      el.appendChild(container);
    } else {
      const overlay = makeOverlay(el, container);
      document.body.appendChild(overlay);
    }
  }

  function clearDecorations(root) {
    document.querySelectorAll(".ln-dialog").forEach((d) => d.remove());
  }

  function showDialog(bias, conf, allProbs, text, el) {
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
      width: "400px",
      maxWidth: "90%",
    });

    const title = document.createElement("h3");
    title.innerText = `Detected Bias: ${bias}`;
    dialog.appendChild(title);

    const confLine = document.createElement("div");
    confLine.innerText = `Confidence: ${(conf * 100).toFixed(1)}%`;
    dialog.appendChild(confLine);

    const list = document.createElement("ul");
    for (const [label, val] of Object.entries(allProbs)) {
      const li = document.createElement("li");
      li.innerText = `${label}: ${(val * 100).toFixed(1)}%`;
      list.appendChild(li);
    }
    dialog.appendChild(list);

    const suggestionDiv = document.createElement("div");
    suggestionDiv.innerText = "Loading neutral suggestion...";
    dialog.appendChild(suggestionDiv);

    const closeBtn = document.createElement("button");
    closeBtn.innerText = "Close";
    closeBtn.style.marginTop = "12px";
    closeBtn.addEventListener("click", () => dialog.remove());
    dialog.appendChild(closeBtn);

    document.body.appendChild(dialog);

    // Fetch suggestion asynchronously
    fetch(`${settings.apiBase}/suggest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sentence: text, bias_type: bias, confidence: conf }),
    })
      .then((r) => r.json())
      .then((j) => {
        suggestionDiv.innerHTML = `<strong>Suggested Neutral:</strong><br>${j.suggestion || "(no suggestion)"}`;
      })
      .catch(() => {
        suggestionDiv.innerText = "(Failed to fetch suggestion)";
      });
  }

  function makeOverlay(el, content) {
    const rect = el.getBoundingClientRect();
    const overlay = document.createElement("div");
    Object.assign(overlay.style, {
      position: "absolute",
      left: `${window.scrollX + rect.left}px`,
      top: `${window.scrollY + rect.top}px`,
      width: `${rect.width}px`,
      height: `${rect.height}px`,
      pointerEvents: "none",
      zIndex: 99999,
    });
    overlay.appendChild(content);
    return overlay;
  }
})();
