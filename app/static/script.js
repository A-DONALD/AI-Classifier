const dropZone = document.getElementById('dropZone');
const uploadBtn = document.getElementById('upload-button');
const fileInput = document.getElementById('fileInput');


let d = new DataTransfer();
// Handle uploaded files
function handleFiles(files) {
      let fileFound = false;
      for (const file of files) {
        for(const f of d.files){
          if(file.name == f.name){
            fileFound = true;
            break;
          }
        }
        if(fileFound) continue;
        const div = document.createElement('div');
        const li = document.createElement('li');
        div.textContent = file.name;
        const deleteBtn = document.createElement('button');
        deleteBtn.textContent = 'X';
        deleteBtn.classList.add('remove-btn');
        deleteBtn.addEventListener('click', (e) => {
        e.preventDefault();
        li.remove(); // Remove the file from the file list
        removeFileFromFileList(div.textContent.slice(0,-3));
        });

        //Save in files list variable
        d.items.add(file);
      }
      //console.log(d.files);
      fileInput.files = d.files;
      //console.log(d.items);

}
function removeFileFromFileList(filename) {
  const dt = new DataTransfer();
  //const input = document.getElementById('fileInput')
  const files = d.files;

  for (let i = 0; i < files.length; i++) {
    const file = files[i]
    if (file.name != filename){
      dt.items.add(file) // here you exclude the file. thus removing it.
    }
  }
  d = dt // Update the list
  fileInput.files = d.files;
}