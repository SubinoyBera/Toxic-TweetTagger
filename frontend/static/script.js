const form = document.querySelector("form");
const button = document.querySelector(".predict-btn");

form.addEventListener("submit", () => {
  button.innerText = "Analyzing...";
  button.disabled = true;
});


const params = new URLSearchParams(window.location.search);
const errorMessage = params.get("error");

if (errorMessage) {

    const toast = document.getElementById("toast");
    const toastText = document.getElementById("toast-text");

    toastText.innerText = "⚠️ " + errorMessage;

    toast.classList.add("show");

    setTimeout(() => {
        toast.classList.remove("show");
    }, 4000);
}

window.history.replaceState({}, document.title, window.location.pathname);


// confidence bar
document.addEventListener("DOMContentLoaded", function () {
  const bar = document.querySelector(".confidence-bar");
  const fill = document.querySelector(".confidence-fill");

  if (!bar || !fill) {
    console.log("Confidence elements not found");
    return;
  }

  const confidence = parseFloat(bar.dataset.confidence);

  if (isNaN(confidence)) {
    console.log("No confidence value yet");
    return;
  }

  console.log("Confidence:", confidence);

  // Set fill width based on confidence
  fill.style.width = confidence + "%";

  // Apply gradient to the FILL element using fixed color stops
  // at 40% and 70% of the fill's own width, scaled to the actual confidence.
  // e.g. if confidence=99.74%, the red zone ends at (40/99.74)*100 ≈ 40.1% of the fill
  const redEnd   = (40 / confidence) * 100;
  const greenStart = (70 / confidence) * 100;

  fill.style.background = `linear-gradient(to right,
    #ff0000 0%,
    #ff6600 ${redEnd * 0.5}%,
    #ffb300 ${redEnd}%,
    #ffff07 ${greenStart}%,
    #a9c838 ${(greenStart + 100) / 2}%,
    #04c00a 100%
  )`;
});


// Show feedback popup if feedback was submitted
document.addEventListener("DOMContentLoaded", function () {

  const params = new URLSearchParams(window.location.search);

  function showPopup(message, bgColor) {
    const popup = document.createElement("div");
    popup.innerHTML = message;
    popup.style.position = "fixed";
    popup.style.top = "20px";
    popup.style.left = "50%";
    popup.style.transform = "translateX(-50%)";
    popup.style.background = bgColor;
    popup.style.color = "white";
    popup.style.padding = "15px 20px";
    popup.style.borderRadius = "8px";
    popup.style.boxShadow = "0 4px 10px rgba(0,0,0,0.2)";
    popup.style.zIndex = "9999";
    popup.style.opacity = "1";
    popup.style.transition = "opacity 0.5s ease";

    document.body.appendChild(popup);

    setTimeout(() => {
        popup.style.opacity = "0";
        setTimeout(() => popup.remove(), 500);
    }, 5500);
  }

  // Success popup
  if (params.get("feedback_submitted") === "true") {
      showPopup("✅ Feedback recorded successfully. Thank you for your feedback! 🙏", "#3da240");
  }

  // Error popup
  if (params.get("feedback_error") === "true") {
      showPopup("❌ Feedback service is currently unavailable. Please try again later.", "#f44336");
  }

  // Clean URL after showing popup
  if (params.has("feedback_submitted") || params.has("feedback_error")) {
      window.history.replaceState({}, document.title, "/");
  }

});