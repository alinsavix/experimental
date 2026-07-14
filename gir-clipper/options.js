// options.js — load/save settings to browser.storage.local

const DEFAULT_WEBHOOK_URL = "https://vootcruiser.lupine.org/clip";

const urlInput    = document.getElementById("webhook-url");
const secretInput = document.getElementById("hmac-secret");
const saveBtn     = document.getElementById("save-btn");
const statusEl    = document.getElementById("status");

// Load saved settings on open
browser.storage.local.get(["webhookUrl", "hmacSecret"]).then((result) => {
  urlInput.value    = result.webhookUrl    || DEFAULT_WEBHOOK_URL;
  secretInput.value = result.hmacSecret   || "";
});

// Save on button click
saveBtn.addEventListener("click", () => {
  const webhookUrl = urlInput.value.trim();
  const hmacSecret = secretInput.value.trim();

  if (!webhookUrl) {
    statusEl.className = "error";
    statusEl.textContent = "Webhook URL is required.";
    return;
  }
  if (!hmacSecret) {
    statusEl.className = "error";
    statusEl.textContent = "HMAC secret is required.";
    return;
  }

  browser.storage.local.set({ webhookUrl, hmacSecret }).then(() => {
    statusEl.className = "";
    statusEl.textContent = "✓ Saved!";
    setTimeout(() => { statusEl.textContent = ""; }, 2000);
  });
});
