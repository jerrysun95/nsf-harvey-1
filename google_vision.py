import keyring
import base64
import os
import json
import string
from utils import Service, encode_image

OUTPUT_FILE = "output/vision.json"

# Stored API key as environment variable called VISION_API
#
# Generate API key from Google Cloud Dashboard -> APIs and Services ->
# Credentials -> Create Credentials -> API Key

# Creates JSON given a list of labels from Google Vision
def create_json(labels, image_name):
    image_json = {}
    index = image_name.rfind('.')
    image_json["piece_number"] = image_name[:index]
    picture_attributes = []
    picture_attributes_scores = []
    print("Creating JSON for " + image_name)

    for label in labels:
        print(label['description'] + ": " + str(label['score']))
        picture_attributes.append(str(label['description']))
        picture_attributes_scores.append(label['score'])

    image_json["picture_attributes"] = picture_attributes
    image_json["picture_attributes_scores"] = picture_attributes_scores

    print("")
    print("JSON created")
    print(image_json)
    print("")

    return image_json

# Outputs JSON to file given a JSON of features
def output_to_file(image_data):
    print("Outputting to file...")

    json_data = None
    with open(OUTPUT_FILE, 'a+') as output_file:

        # read in existing data if present
        data = output_file.read()
        try:
            json_data = json.loads(data)
        except:
            print('exception')
            json_data = []

        # append new image to json and write back to file
        json_data.append(image_data)
        json_data = json.dumps(json_data, indent=4)
        print(json_data)
        
    with open(OUTPUT_FILE, 'w') as output_file:
        output_file.write(json_data)
    
    print("Outputted to file")
    print("")

# Send image to Google Vision from local file
def vision_from_file(image_name, photo_file):
    print("Sending to Google Vision from file...")
    access_token = keyring.get_password("system", "VISION_API_KEY")
    service = Service('vision', 'v1', access_token=access_token)

    with open(photo_file, 'rb') as image:
        base64_image = encode_image(image)
        body = {
            'requests': [{
                'image': {
                    'content': base64_image,
                },
                'features': [
                    {
                        'type': 'LABEL_DETECTION',
                        'maxResults': 1000,
                    }
                ]

            }]
        }
        response = service.execute(body=body)
        labels = response['responses'][0]['labelAnnotations']
        for label in labels:
            print(label['description'] + ": " + str(label['score']))
        #print(response)

# Send image data to Google Vision
def vision_from_data(image_name, image_content):
    print("Sending to Google Vision from Box...")
    access_token = keyring.get_password("system", "VISION_API_KEY")
    service = Service('vision', 'v1', access_token=access_token)

    base64_image = base64.b64encode(image_content).decode()
    body = {
        'requests': [{
            'image': {
                'content': base64_image,
            },
            'features': [
                {
                    'type': 'LABEL_DETECTION',
                    'maxResults': 1000,
                }
            ]

        }]
    }
    response = service.execute(body=body)
    print("Response received from Google Vision")
    print("")
    labels = response['responses'][0]['labelAnnotations']
    image_json = create_json(labels, image_name)
    output_to_file(image_json)
