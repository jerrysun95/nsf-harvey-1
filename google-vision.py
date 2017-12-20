import keyring
from utils import Service, encode_image

def main(photo_file):
    """Run a face detection request on a single image"""

    # Stored API key as environment variable called VISION_API
    #
    # Generate API key from Google Cloud Dashboard -> APIs and Services ->
    # Credentials -> Create Credentials -> API Key

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

if __name__ == '__main__':
    main("airport.jpg")
