import keyring
import base64
from utils import Service, encode_image

"""Run a face detection request on a single image"""

# Stored API key as environment variable called VISION_API
#
# Generate API key from Google Cloud Dashboard -> APIs and Services ->
# Credentials -> Create Credentials -> API Key

class GoogleVision:

    # Send image to Google Vision from local file
    def vision_from_file(self, photo_file):
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
    def vision_from_data(self, image_content):
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
        labels = response['responses'][0]['labelAnnotations']
        for label in labels:
            print(label['description'] + ": " + str(label['score']))
        #print(response)
