let file = null,
  file_data = null;

const thumbnailElement = document.getElementById("thumb");
fileInput = document.getElementById("customFile");

const imgForm = document.getElementById("img-form");
const imgMeta = document.getElementById("img-meta");
const uploadBtn = document.getElementById("upload-btn");
const predictBtn = document.getElementById("predict-btn");
const spinnerBtn = document.getElementById("spinner-div");
const resultDiv = document.getElementById("result-div");
const resultLabel = document.getElementById("result-label");

const enableThumbnail = file => {
  thumbnailElement.style.display = "block";
  thumbnailElement.src = file;
};

const upload = () => {
  const reader = new FileReader();
  const fileList = document.getElementById("customFile").files;
  file = fileList[0];

  if (file) {
    imgMeta.textContent = file.name;
    reader.readAsDataURL(file);
    reader.onload = () => {
      file_data = reader.result;
      enableThumbnail(file_data);
      resultDiv.style.display = "none";
    };
  }
};

const handleFileChange = () => {
  uploadBtn.disabled = false;
  predictBtn.disabled = false;
};

const handleSubmit = e => {
  e.preventDefault();

  const fileInput = document.getElementById("customFile");

  if (file_data) {
    base64_image = file_data.replace("data:" + file.type + ";base64,", "");

    spinnerBtn.style.display = "block";

    fetch("/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data: base64_image, filename: file.name })
    })
      .then(response => response.json())
      .then(payload => {
        spinnerBtn.style.display = "none";
        if (payload.code === 400) throw payload.message; // a bad request
        resultDiv.style.display = "block";
        resultLabel.textContent = payload.message; //showing api response

        uploadBtn.disabled = true;
        predictBtn.disabled = true;
        fileInput.value = "";
      })
      .catch(err => console.log(err));
  }
};

window.onload = () => {
  spinnerBtn.style.display = "none";
  imgForm.addEventListener("submit", handleSubmit);
  fileInput.addEventListener("change", handleFileChange);
  resultDiv.style.display = "none";
};
