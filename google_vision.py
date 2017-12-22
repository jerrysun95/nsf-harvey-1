import keyring
import base64
import os
import json
import string
from utils import Service, encode_image

OUTPUT_FILE = "output/output.json"

# Stored API key as environment variable called VISION_API
#
# Generate API key from Google Cloud Dashboard -> APIs and Services ->
# Credentials -> Create Credentials -> API Key

class GoogleVision:

    # Creates JSON given a list of labels from Google Vision
    def create_json(self, labels, image_name):
        image_json = {}
        image_json["piece_number"] = image_name
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
    def output_to_file(self, image_json):
        print("Outputting to file...")
        if os.path.isfile(OUTPUT_FILE):
            print("there")
            with open(OUTPUT_FILE, 'r') as output_file:
                current_file = output_file.read()
            with open(OUTPUT_FILE, 'w') as output_file:
                current_file = current_file.replace("]  ", "," + str(image_json) + "]  ")
                output_file.write(current_file)
        else:
            print("here")
            with open(OUTPUT_FILE, 'w') as output_file:
                output_file.write("[" + str(image_json) + "]  ")
        print("Outputted to file")
        print("")

    # Send image to Google Vision from local file
    def vision_from_file(self, image_name, photo_file):
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
    def vision_from_data(self, image_name, image_content):
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
        image_json = self.create_json(labels, image_name)
        self.output_to_file(image_json)
