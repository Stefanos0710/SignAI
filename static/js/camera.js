// Zugriff auf die Kamera anfordern
navigator.mediaDevices.getUserMedia({ video: true, audio: false })
  .then(function(stream) {
    const video = document.getElementById('video');
    video.srcObject = stream;
  })
  .catch(function(error) {
    alert("Access to the camera was denied or is not available.");
  });