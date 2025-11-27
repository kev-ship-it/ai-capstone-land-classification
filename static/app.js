const form = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const submitBtn = document.getElementById("submit-btn");

const preview = document.getElementById("preview");
const previewImg = document.getElementById("preview-img");

const result = document.getElementById("result");
const predictionText = document.getElementById("prediction-text");
const probAgriBar = document.getElementById("prob-agri-bar");
const probDetails = document.getElementById("prob-details");

const errorBox = document.getElementById("error");

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  preview.classList.remove("hidden");
  errorBox.classList.add("hidden");
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = fileInput.files[0];
  if (!file) {
    showError("Please select an image first.");
    return;
  }

  submitBtn.disabled = true;
  submitBtn.textContent = "Predicting...";

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error || `Request failed with ${res.status}`);
    }

    const data = await res.json();
    const pAgri = data.prob_agri ?? 0;
    const pNonAgri = data.prob_non_agri ?? 0;

    predictionText.textContent = `Predicted class: ${data.prediction}`;
    probAgriBar.style.width = `${(pAgri * 100).toFixed(1)}%`;
    probDetails.textContent =
      `Agricultural: ${(pAgri * 100).toFixed(1)}% • ` +
      `Non‑agricultural: ${(pNonAgri * 100).toFixed(1)}%`;

    result.classList.remove("hidden");
    errorBox.classList.add("hidden");
  } catch (err) {
    showError(err.message);
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Predict";
  }
});

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.classList.remove("hidden");
}
