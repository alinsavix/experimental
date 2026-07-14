// GIR Clipper — popup.js
// Posts the current page (URL + title + readable text) to the Hermes webhook.
// Webhook URL and HMAC secret are read from browser.storage.local (set via options page).

const DEFAULT_WEBHOOK_URL = "https://vootcruiser.lupine.org/clip";

// --- HMAC-SHA256 signing (SubtleCrypto) ---
async function hmacSign(secret, message) {
  const enc = new TextEncoder();
  const key = await crypto.subtle.importKey(
    "raw", enc.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false, ["sign"]
  );
  const sig = await crypto.subtle.sign("HMAC", key, enc.encode(message));
  return Array.from(new Uint8Array(sig))
    .map(b => b.toString(16).padStart(2, "0")).join("");
}

// --- Extract full page readable text ---
function extractPageText(tabId) {
  return new Promise((resolve) => {
    browser.tabs.executeScript(tabId, {
      code: `
        (function() {
          const clone = document.cloneNode(true);
          for (const el of clone.querySelectorAll(
            'script,style,nav,header,footer,aside,[role="banner"],[role="navigation"],[role="complementary"],iframe,noscript'
          )) el.remove();
          const main = clone.querySelector('article, [role="main"], main') || clone.body;
          const text = (main || clone.body).innerText || clone.body.textContent || '';
          return text.replace(/[ \\t]+/g, ' ').replace(/\\n{3,}/g, '\\n\\n').trim().slice(0, 15000);
        })()
      `
    }, (results) => resolve((results && results[0]) || ""));
  });
}

// --- Extract current selection text ---
function extractSelection(tabId) {
  return new Promise((resolve) => {
    browser.tabs.executeScript(tabId, {
      code: `window.getSelection().toString().trim()`
    }, (results) => resolve((results && results[0]) || ""));
  });
}

// --- Send to webhook ---
async function sendToWebhook(webhookUrl, hmacSecret, tab, content, isSelection) {
  const payload = JSON.stringify({
    url:     tab.url,
    title:   tab.title,
    content: content,
    clipped: isSelection ? "selection" : "full_page",
  });

  const sig = await hmacSign(hmacSecret, payload);

  const resp = await fetch(webhookUrl, {
    method:  "POST",
    headers: {
      "Content-Type":        "application/json",
      "X-Hub-Signature-256": "sha256=" + sig,
    },
    body: payload,
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`${resp.status}: ${text.slice(0, 120)}`);
  }
}

// --- Main ---
document.addEventListener("DOMContentLoaded", async () => {
  const btnPage      = document.getElementById("btn-page");
  const btnSelection = document.getElementById("btn-selection");
  const btnSettings  = document.getElementById("btn-settings");
  const status       = document.getElementById("status");
  const urlPreview   = document.getElementById("url-preview");
  const selHint      = document.getElementById("selection-hint");

  const [tab] = await browser.tabs.query({ active: true, currentWindow: true });
  urlPreview.textContent = tab.url;

  // Load settings
  const stored = await browser.storage.local.get(["webhookUrl", "hmacSecret"]);
  const webhookUrl = stored.webhookUrl || DEFAULT_WEBHOOK_URL;
  const hmacSecret = stored.hmacSecret || "";

  if (!hmacSecret) {
    status.className = "error";
    status.textContent = "⚠️ No secret set — open settings first!";
    btnPage.disabled = true;
  }

  // Check if there's a selection
  const selection = await extractSelection(tab.id);
  if (selection) {
    btnSelection.disabled = false;
    selHint.textContent = `✂️ ${selection.length} chars selected`;
  } else {
    selHint.textContent = "No text selected";
  }

  async function clip(useSelection) {
    if (!hmacSecret) return;
    btnPage.disabled = true;
    btnSelection.disabled = true;
    status.className = "sending";
    status.textContent = useSelection ? "Sending selection…" : "Extracting page…";

    try {
      let content;
      if (useSelection) {
        content = selection;
      } else {
        content = await extractPageText(tab.id);
        status.textContent = "Sending to GIR…";
      }

      await sendToWebhook(webhookUrl, hmacSecret, tab, content, useSelection);

      status.className = "success";
      status.textContent = "✓ Sent! GIR is saving it now.";
      btnPage.textContent = "✓ Done";
    } catch (err) {
      status.className = "error";
      status.textContent = `Failed: ${err.message}`;
      btnPage.disabled = false;
      if (selection) btnSelection.disabled = false;
    }
  }

  btnPage.addEventListener("click",      () => clip(false));
  btnSelection.addEventListener("click", () => clip(true));
  btnSettings.addEventListener("click",  () => browser.runtime.openOptionsPage());
});
