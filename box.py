from boxpython import BoxAuthenticateFlow, BoxSession, BoxError
from request import BoxRestRequest
from StringIO import StringIO
import google_vision as gv
import keyring, requests, read_csv, text, os, json
from multiprocessing import Pool as ThreadPool

# Tokens Changed Callback
def tokens_changed(refresh_token, access_token):
    keyring.set_password("system", "BOX_ACCESS_TOKEN", access_token),
    keyring.set_password("system", "BOX_REFRESH_TOKEN", refresh_token)

# Upload File
def upload(file_name, folder_id, file_location):
    print("Uploading " + file_name + " to folder id " + str(folder_id) + " from " + file_location + "...")
    response = box.upload_file(file_name, folder_id, file_location)
    print('File ID: %s' % response['entries'][0]['id'])
    print('Uploaded ' + file_name)
    print("")
    return response['entries'][0]['id']

# Download File
def download(file_id, file_location):
    print("Downloading file id " + str(file_id) + " to " + file_location + "...")
    response = box.download_file(file_id, file_location)
    print("Downloaded file")
    print("")

# Delete File
def delete(file_id):
    print("Deleting " + file_id + "...")
    response = box.delete_file(file_id)
    print("Deleted file id " + file_id)
    print("")

# Copy File
def copy(file_id, dest_folder_id):
    print("Copying file " + str(file_id) + " to " + str(dest_folder_id) + "...")
    response = box.copy_file(file_id, dest_folder_id)
    print("Copied file")
    print("")

# List files in folder
def items(folder_id, lim=5000, ofs=0):
    print("Getting items in folder " + str(folder_id) + "...")
    response = box.get_folder_items(folder_id, limit=lim, offset=ofs, fields_list=['name', 'type', 'id'])
    print("Got " + str(len(response['entries'])) + " items in folder " + str(folder_id))
    print("")
    return response

# Helper method to manage request to Google Vision from send_to_vision
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


# Send to Google Vision
def send_to_vision(file_name, file_id, image_type='image', chunk_size=1034*1034*1):
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
            image_content += chunk
            transferred += len(chunk)

    #print(image)
    if image_type == 'image':
        return gv.vision_from_data_image(file_name, image_content)
    elif image_type == 'text':
        return gv.vision_from_data_text(file_name, image_content)
    else:
        return None

# Read file data from box
def get_file_data(file_id, chunk_size=1034*1034*1):
    file_content = ''
    req = request("GET", "files/%s/content" % (file_id))
    total = -1
    if hasattr(req, 'headers'):
        lower_headers = {k.lower():v for k,v in req.headers.items()}
        if 'content-length' in lower_headers:
            total = lower_headers['content-length']

    transferred = 0
    for chunk in req.iter_content(chunk_size=chunk_size):
        if chunk: # filter out keep-alive new chunks
            file_content += chunk
            transferred += len(chunk)
    return file_content

# Parse excel file from box
def parse_excel(file_id, chunk_size=1034*1034*1):
    print('Parsing excel file ' + str(file_id) + '...')
    file_data = get_file_data(file_id, chunk_size)
    dataframe = read_csv.read_excel(StringIO(file_data))
    data = read_csv.parse_from_data(dataframe.values, dataframe.axes[1])
    print('Parsed excel file')
    print('')
    return data

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

# Runs all images found in box source folder through google vision label detection
# Writes results to output file
def vision(src_folder_id, t):
    vision_data = []

    # Read folder to get file names and ids
    response = items(src_folder_id)
    entries = response['entries']

    # # Send each file to google vision for label detection
    # for entry in entries:
    #     name = entry['name'].lower()
    #     if entry['type'] == 'file' and '.jpg' in name or '.png' in name or '.jpeg' in name:
    #         try:
    #             result = send_to_vision(name, entry['id'])
    #             vision_data.append(result)
    #         except:
    #             pass

    pool = ThreadPool(16)
    vision_data = pool.map(vision_thread, entries)

    # Write results to output file
    with open('output/' + t + '.json', 'w') as f:
        f.write(json.dumps(vision_data, indent=4))

def vision_thread(entry):
    # Send each file to google vision for label detection
    result = {}
    name = entry['name'].lower()
    print('---------' + name + '---------')
    if entry['type'] == 'file' and '.jpg' in name or '.png' in name or '.jpeg' in name:
        try:
            result = send_to_vision(name, entry['id'])
        except:
            result = {'name':name, 'attributes_scores':[], 'attributes':[], 'error':True}
    return result

def vision_local_thread(entry):
    name = entry[entry.rfind('/') + 1:].lower()
    print('---------' + name + '---------')

    # try:
    result = gv.vision_from_file(name, entry)
    # except:
    #     result = {'name':name, 'attributes_scores':[], 'attributes':[], 'error':True}
    return result

def vision_local(src_folder, t, files):
    # entries = os.listdir(src_folder)
    entries = [src_folder + '/' + x for x in files]

    pool = ThreadPool(4)
    vision_data = pool.map(vision_local_thread, entries)

    # Write results to output file
    with open('output/' + t + '.json', 'w') as f:
        f.write(json.dumps(vision_data, indent=4))

# Runs all text images found in box source folder through google vision text detection
# Writes results to output file
def vision_text(src_folder_id, t):
    vision_data = []

    # Read folder to get file names and ids
    response = box.items(src_folder_id)
    entries = response['entries']

    # Send each file to google vision for text detection
    for entry in entries:
        name = entry['name'].lower()
        if entry['type'] == 'file' and '.jpg' in name or '.png' in name or '.jpeg' in name:
            try:
                result = send_to_vision(name, entry['id'], image_type='text')
                t = text.parse_text([result['text']], result['piece_name'])
                result['attributes'] = t
                vision_data.append(result)
            except:
                pass

    # Write results to output file
    with open('output/' + t + '_text.json', 'w') as f:
        f.write(json.dumps(vision_data, indent=4))

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

