'''import keyring
import requests
import json
import os
import sys
import traceback
from boxsdk import OAuth2
from boxsdk import Client

def main():
    AUTH_CODE = 'Jpqw1i28X7JHBif8gyrc0awiuDaX3sJA'

    oauth = OAuth2(
      client_id=keyring.get_password("system", "BOX_CLIENT_ID"),
      client_secret=keyring.get_password("system", "BOX_CLIENT_SECRET"),
      access_token=keyring.get_password("system", "BOX_ACCESS_TOKEN"),
      refresh_token=keyring.get_password("system", "BOX_REFRESH_TOKEN"),
      store_tokens=callback,
    )
    '''

'''
    ----------------------------------------------------------------------------
    If error message is received that token is expired or this is your first time running the code, uncomment the code below.
    Go to https://app.box.com/api/oauth2/authorize?response_type=code&client_id=cd0xj5a7m6x5fqzvt6ej5gm9dkoux7zw
    and grant access to the Box. Once this occurs, you should be redirected to an
    error page. The URL of this error page should be http://127.0.0.1/?code=_HERE_IS_YOUR_NEW_CODE.
    Your new code is at the end of the URL. Copy that into the variable AUTH_CODE and run the code again and it
    will go through. Comment out the code below once new token is set.
    ----------------------------------------------------------------------------
'''

'''
    #UNCOMMENT THIS CODE

    auth_url, csrf_token = oauth.get_authorization_url('http://127.0.0.1')

    access_token, refresh_token = oauth.authenticate(AUTH_CODE)

    keyring.set_password("system", "BOX_ACCESS_TOKEN", access_token),
    keyring.set_password("system", "BOX_REFRESH_TOKEN", refresh_token)
'''
'''
    client = Client(oauth)

    me = client.user(user_id = 'me').get()
    print 'user_login: ' + me['login']

    client.folder(folder_id='0').create_subfolder('L1')

def callback(access_token, refresh_token):
    print("Callback")

main()
'''

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
