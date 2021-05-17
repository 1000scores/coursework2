import requests
import json
import io
from PIL import Image
from serve import run_model


url1 = 'http://92.63.105.87:8080/neuron/getPhoto'
TOKEN = '123'

headers1 = {'token': TOKEN}
response_image = requests.get(url1, headers=headers1)


if response_image.status_code != 200:
    print('Server response_path error!')
    exit()

id = response_image.headers['id']

image_bytes = io.BytesIO(response_image.content)

img = Image.open(image_bytes)

img, labels = run_model(img)

img.show()

data2 = {
    'labels': labels
}

headers2 = {
    'token': TOKEN,
    'id': id,
}

buf = io.BytesIO()
img.save(buf, format='JPEG')
buf.seek(0)

url2 = 'http://92.63.105.87:8080/neuron/sendPhotoInfo'


response2 = requests.post(url2, files={'file': ('image.jpeg', buf, 'image/jpeg')}, data=data2, headers=headers2)

kek = 0