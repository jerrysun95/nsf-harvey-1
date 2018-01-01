from boxpython import BoxAuthenticateFlow, BoxSession, BoxError
from request import BoxRestRequest
import google_vision as gv
import keyring
import requests

#Tokens Changed Callback
def tokens_changed(refresh_token, access_token):
    keyring.set_password("system", "BOX_ACCESS_TOKEN", access_token),
    keyring.set_password("system", "BOX_REFRESH_TOKEN", refresh_token)

#Upload File
def upload(file_name, folder_id, file_location):
    print("Uploading " + file_name + " to folder id " + str(folder_id) + " from " + file_location + "...")
    response = box.upload_file(file_name, folder_id, file_location)
    print('File ID: %s' % response['entries'][0]['id'])
    print('Uploaded ' + file_name)
    print("")
    return response['entries'][0]['id']

#Download File
def download(file_id, file_location):
    print("Downloading file id " + file_id + " to " + file_location + "...")
    response = box.download_file(file_id, file_location)
    print("Downloaded file")
    print("")

#Delete File
def delete(file_id):
    print("Deleting " + file_id + "...")
    response = box.delete_file(file_id)
    print("Deleted file id " + file_id)
    print("")

def items(folder_id, lim=100, ofs=0):
    print("Getting items in folder " + folder_id + "...")
    response = box.get_folder_items(folder_id, limit=lim, offset=ofs, fields_list=['name', 'type', 'id'])
    print("Got items in folder " + folder_id)
    print("")
    return response

#Helper method to manage request to Google Vision from send_to_vision
def request(method, command):
    data = None
    querystring = None
    files = None
    headers = None
    stream = None
    json_data = True
    if files or (data and isinstance(data, MultipartUploadWrapper)):
        url_prefix = BoxRestRequest.API_UPLOAD_PREFIX
    else:
        url_prefix = BoxRestRequest.API_PREFIX

    if headers is None:
        headers = {}
    headers['Authorization'] = 'Bearer %s' % keyring.get_password("system", "BOX_ACCESS_TOKEN")

    url = '%s/%s' % (url_prefix, command)

    if json_data and data is not None:
        data = json.dumps(data)

    kwargs = { 'headers' : headers }
    if data is not None: kwargs['data'] = data
    if querystring is not None: kwargs['params'] = querystring
    if files is not None: kwargs['files'] = files
    if stream is not None: kwargs['stream'] = stream
    #if self.timeout is not None: kwargs['timeout'] = self.timeout

    # returns a requests.Response object
    return requests.request(method=method, url=url, **kwargs)


#Send to Google Vision
def send_to_vision(file_name, file_id, chunk_size=1034*1034*1):
    req = request("GET", "files/%s/content" % (file_id, ))
    total = -1
    image_content = ''
    if hasattr(req, 'headers'):
        lower_headers = {k.lower():v for k,v in req.headers.items()}
        if 'content-length' in lower_headers:
            total = lower_headers['content-length']

    transferred = 0
    for chunk in req.iter_content(chunk_size=1034*1034*1):
        if chunk:
            #fp.write(chunk)
            #fp.flush()
            image_content += chunk
            transferred += len(chunk)

    #print(image)
    gv.vision_from_data(file_name, image_content)

# Set up a box session
def setup_box():
    # Generate BoxAuthenticationFlow
    flow = BoxAuthenticateFlow(keyring.get_password("system", "BOX_CLIENT_ID"), keyring.get_password("system", "BOX_CLIENT_SECRET"))
    flow.get_authorization_url()
    access_token = keyring.get_password("system", "BOX_ACCESS_TOKEN")
    refresh_token = keyring.get_password("system", "BOX_REFRESH_TOKEN")

    #Uncomment this to get a new access and refresh token from a code
    # access_token, refresh_token = flow.get_access_tokens('uolXRxIJQynMGmLglAe5oGjXoIlTixVs')
    # keyring.set_password("system", "BOX_ACCESS_TOKEN", access_token)
    # keyring.set_password("system", "BOX_REFRESH_TOKEN", refresh_token)

    # Generate BoxSession
    return BoxSession(keyring.get_password("system", "BOX_CLIENT_ID"), 
        keyring.get_password("system", "BOX_CLIENT_SECRET"), refresh_token, access_token, tokens_changed)

#----------------------------------------------------------------------------------------------
# Access a BoxSession, upload file to box, send file to Google Vision, and delete file from Box
#----------------------------------------------------------------------------------------------

box = setup_box()

# # Generate BoxAuthenticationFlow
# flow = BoxAuthenticateFlow(keyring.get_password("system", "BOX_CLIENT_ID"), keyring.get_password("system", "BOX_CLIENT_SECRET"))
# flow.get_authorization_url()
# access_token = keyring.get_password("system", "BOX_ACCESS_TOKEN")
# refresh_token = keyring.get_password("system", "BOX_REFRESH_TOKEN")

# #Uncomment this to get a new access and refresh token from a code
# # access_token, refresh_token = flow.get_access_tokens('uolXRxIJQynMGmLglAe5oGjXoIlTixVs')
# # keyring.set_password("system", "BOX_ACCESS_TOKEN", access_token)
# # keyring.set_password("system", "BOX_REFRESH_TOKEN", refresh_token)

# # Generate BoxSession
# box = BoxSession(keyring.get_password("system", "BOX_CLIENT_ID"), keyring.get_password("system", "BOX_CLIENT_SECRET"), refresh_token, access_token, tokens_changed)

# # Uplaod file to Box
# new_file_id = upload('obama.jpeg', 0, 'test_images/obama.jpeg')
# # # Send file to Google Vision
# try:
#     send_to_vision('obama.jpeg', new_file_id)
# finally:
#     delete(new_file_id)
    
# # Delete file from Google Vision
