from common import get_distance
import requests
import json
import io
from PIL import Image
from serve import run_model
import cv2
import PIL
import base64
import io
import time

def recognize_image_from_server(url1, url2):

    #url1 = 'http://92.63.105.87:8080/neuron/getPhoto'
    TOKEN = '123'

    headers1 = {'token': TOKEN}
    response_image = requests.get(url1, headers=headers1)

    if response_image.status_code != 200:
        print('Server response_path error!')
        exit()

    id = response_image.headers['id']

    image_bytes = io.BytesIO(response_image.content)

    img = Image.open(image_bytes)

    img, labels, bboxes = run_model(img)

    data2 = {
        'labels': ' '.join(labels)
    }
    print(data2)
    headers2 = {
        'token': TOKEN,
        'id': id,
    }
    #url2 = 'http://92.63.105.87:8080/neuron/sendPhotoInfo'

    img.save('temp/tmp.jpg')
    
    with open("temp/tmp.jpg", "rb") as img_file:
        img_str = base64.b64encode(img_file.read())
        
    r = requests.post(url2, data={
    'image': img_str,
    'labels': labels,
    'distances': get_distance(bboxes)
    }, headers=headers2)


if __name__ == '__main__':
    done = 0
    url1 = input()
    url2 = input()
    
    while True:
        try: 
            
            recognize_image_from_server(url1, url2)
            time.sleep(100)
            done += 1
            print(f'Done : {done}')
        except BaseException as ex:
            error = 'Error'
