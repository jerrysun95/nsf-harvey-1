from boxpython import BoxAuthenticateFlow, BoxSession, BoxError
import keyring

flow = BoxAuthenticateFlow(keyring.get_password("system", "BOX_CLIENT_ID"), keyring.get_password("system", "BOX_CLIENT_SECRET"))
flow.get_authorization_url()
access_token = keyring.get_password("system", "BOX_ACCESS_TOKEN")
refresh_token = keyring.get_password("system", "BOX_REFRESH_TOKEN")

#Uncomment this to get a new access and refresh token from a code
#access_token, refresh_token = flow.get_access_tokens('P6jKDBZXFYGuGoAkxLTlaAxmxwv3e0cD')

box = BoxSession(keyring.get_password("system", "BOX_CLIENT_ID"), keyring.get_password("system", "BOX_CLIENT_SECRET"), refresh_token, access_token, tokens_changed)

#Upload File
def upload(file_name, folder_id, file_location):
    response = box.upload_file(file_name, folder_id, file_location)
    print('File ID: %s' % response['entries'][0]['id'])

#Delete File
def delete(file_id):
    response = box.delete_file(file_id)
    print("Success")

#Download File
def download(file_id, file_location):
    response = box.download_file(file_id, file_location)
    print("Success")

#Tokens Changed Callback
def tokens_changed(refresh_token, access_token):
    keyring.set_password("system", "BOX_ACCESS_TOKEN", access_token),
    keyring.set_password("system", "BOX_REFRESH_TOKEN", refresh_token)
