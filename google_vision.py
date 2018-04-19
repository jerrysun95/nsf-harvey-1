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
    image_json["name"] = image_name
    attributes = []
    attributes_scores = []
    print("Creating JSON for " + image_name)

    for label in labels:
        print(label['description'] + ": " + str(label['score']))
        attributes.append(label['description'].encode('utf-8'))
        attributes_scores.append(label['score'])

    image_json["attributes"] = attributes
    image_json["attributes_scores"] = attributes_scores

    print("")
    print("JSON (image) created")
    print(image_json)
    print("")

    return image_json

# Creates JSON given a text from Google Vision
def create_json_text(text, image_name):
    print('Creating JSON for text ' + image_name)
    
    text_json = {}
    text_json['piece_name'] = image_name
    text_json['text'] = text.replace('\n', ' ')

    print('')
    print('JSON (text) created')
    print(text_json)
    print('')

    return text_json

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
        # print(response)
        return create_json(labels, image_name)

# Send image data to Google Vision
def vision_from_data(image_name, image_content, request_type):
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
                    'type': request_type
                }
            ]

        }]
    }
    return service.execute(body=body)

# Send image to Google Vision
# Generate and return formatted json with labels and image name
def vision_from_data_image(image_name, image_content):
    response = vision_from_data(image_name, image_content, 'LABEL_DETECTION')

    print('Response received from Google Vision')
    print('')

    labels = {}
    try:
        labels = response['responses'][0]['labelAnnotations']
    except:
        print('Exception during handling of Google Vision image response')
        pass
    return create_json(labels, image_name)

# Send text image to Google Vision
# Generate and return formatted json with text and image name
def vision_from_data_text(image_name, image_content):
    response = vision_from_data(image_name, image_content, 'TEXT_DETECTION')

    print('Response received from Google Vision')
    print('')

    text = {}
    try:
        text = response['responses'][0]['textAnnotations'][0]['description']
    except:
        print('Exception during handling of Google Vision text response')
        pass
    return create_json_text(text, image_name)
