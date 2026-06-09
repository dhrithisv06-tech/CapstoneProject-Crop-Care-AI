const detectBtn =
document.getElementById("detectBtn");

detectBtn.addEventListener(
"click", async () => {

    const file =
    fileInput.files[0];

    if (!file) {
        alert("Please upload image");
        return;
    }

    detectBtn.innerHTML =
    "AI Scanning...";

    const formData =
    new FormData();

    formData.append(
    "image", file);

    try {

        const response =
        await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data =
        await response.json();

        document.querySelector(
        ".result-card"
        ).innerHTML = `
            <h3>${data.disease}</h3>
            <p>
            <strong>Confidence:</strong>
            ${data.confidence}
            </p>
            <p>
            <strong>Status:</strong>
            Disease Detected
            </p>
        `;

        detectBtn.innerHTML =
        "✅ Done";

    } catch(error) {

        alert("Prediction Failed");
    }
});