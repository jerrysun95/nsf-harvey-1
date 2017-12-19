from boxpython import BoxAuthenticateFlow, BoxSession, BoxError
import keyring

flow = BoxAuthenticateFlow(keyring.get_password("system", "BOX_CLIENT_ID"), keyring.get_password("system", "BOX_CLIENT_SECRET"))
flow.get_authorization_url()
access_token = keyring.get_password("system", "BOX_ACCESS_TOKEN")
refresh_token = keyring.get_password("system", "BOX_REFRESH_TOKEN")

#access_token, refresh_token = flow.get_access_tokens('P6jKDBZXFYGuGoAkxLTlaAxmxwv3e0cD')

def tokens_changed(refresh_token, access_token):
    keyring.set_password("system", "BOX_ACCESS_TOKEN", access_token),
    keyring.set_password("system", "BOX_REFRESH_TOKEN", refresh_token)

box = BoxSession(keyring.get_password("system", "BOX_CLIENT_ID"), keyring.get_password("system", "BOX_CLIENT_SECRET"), refresh_token, access_token, tokens_changed)

print(box.get_folder_info(0))
