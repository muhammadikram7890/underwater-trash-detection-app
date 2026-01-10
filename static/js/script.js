
  const fileInput = document.getElementById('file-input');
  const uploadBtn = document.getElementById('upload-btn');
  const form = document.getElementById('upload-form');

  uploadBtn.addEventListener('click', (e) => {
    e.preventDefault();
    fileInput.click();
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
      document.getElementById('preview-hidden').value = 'true';
      form.submit();
    }
  });