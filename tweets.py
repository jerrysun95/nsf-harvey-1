import requests
import keyring
import json

def get_image_from_url (url, filename):
    image = requests.get(url).content

    access_token = keyring.get_password("system", "BOX_ACCESS_TOKEN")
    print(access_token)
    parent_id = 0

    headers = { 'Authorization' : 'Bearer {0}'.format(access_token) }
    url = 'https://upload.box.com/api/2.0/files/content'
    files = { 'filename': (filename, image) }
    data = { "parent_id": parent_id }
    response = requests.post(url, data=data, files=files, headers=headers)
    file_info = response
    print(file_info)

get_image_from_url('https://pbs.twimg.com/media/DJ4D7ZJV4AA_Zsi.jpg', 'test')
