async function predictDisease() {
  const fileInput = document.getElementById("imageUpload");
  const resultDiv = document.getElementById("result");
  const preview = document.getElementById("preview");

  if (!fileInput.files.length) {
    resultDiv.innerHTML = "Please upload an image first!";
    return;
  }

  const file = fileInput.files[0];
  preview.src = URL.createObjectURL(file);
  preview.style.display = "block";

  const formData = new FormData();
  formData.append("file", file);

  resultDiv.innerHTML = "Analyzing image... ⏳";

  const response = await fetch("/predict", {
    method: "POST",
    body: formData
  });

  const data = await response.json();

  resultDiv.innerHTML = `
    <h2>Disease: ${data.disease}</h2>
    <p><strong>Prevention:</strong> ${data.prevention}</p>
  `;
}
const file = fileInput.files[0];
preview.src = URL.createObjectURL(file);
preview.classList.add("show");

