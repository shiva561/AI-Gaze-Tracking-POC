// script.js

// Generate or reuse persistent sessionId
let sessionId = localStorage.getItem("sessionId");
if (!sessionId) {
  sessionId = "session_" + Date.now();
  localStorage.setItem("sessionId", sessionId);
}

function sendRunningApps() {
  // Simulated app list — replace with actual dynamic data if needed
  const runningApps = [
    "chrome.exe",
    "zoom.exe",
    "vscode.exe",
    "notepad.exe"
  ];

  fetch('http://127.0.0.1:5000/api/report_apps', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      sessionId: sessionId,
      apps: runningApps
    }),
  })
  .then(response => response.json())
  .then(data => {
    document.getElementById("appListOutput").innerText =
      "📦 Sent app list to server for session: " + sessionId + "\n" +
      data.received.join("\n");
  })
  .catch(err => {
    console.error("Failed to send app list:", err);
    document.getElementById("appListOutput").innerText = "❌ Failed to send data.";
  });
}
