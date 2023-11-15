var image;
document.querySelector('form').addEventListener('submit', () => {
    const img = document.querySelector('img')
    const file = document.querySelector('#customFile').files[0]
    const reader = new FileReader()
    reader.onload = () => {
        img.src = reader.result
        console.log("before")
        document.querySelector('#readOnly').value = reader.result
        console.log("after")
        image = reader.result;
        console.log(image);
    }
    reader.readAsDataURL(file)
    //const data = { username: 'example' };

    fetch('http://0.0.0.0:5000/', {
    method: 'POST', // or 'PUT'
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify(image),
    })
    .then((response) => response.json())
    .then((image) => {
    console.log('Success:', image);
    })
    .catch((error) => {
    console.error('Error:', error);
    });
})

// copy to clipboard
const copyToClipboard=()=>{
    const imageValue=document.getElementById('readOnly')
    imageValue.select();
    imageValue.setSelectionRange(0, 99999) // for mobile devices
    document.execCommand('copy')
}