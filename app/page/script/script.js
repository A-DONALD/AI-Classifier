// button
const fileInput = document.getElementById('file-input');
const uploadButton = document.getElementById('upload-button');

uploadButton.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', () => {
    const files = fileInput.files;
    handleSelectedFiles(files);
});

function handleSelectedFiles(files) {
    // Gérer les fichiers sélectionnés ici
    console.log(files);
}

//dropzone
const dropzone = document.getElementById('dropzone');

dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
});

dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
});

dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');

    const files = e.dataTransfer.files;
    handleDroppedFiles(files);
});

function handleDroppedFiles(files) {
    // Gérer les fichiers déposés ici
    console.log(files);
}






