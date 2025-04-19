import requests
import json
from typing import Union

from server_config import LOCALHOST, PORT

URL_HOME = 'http://' + LOCALHOST + ':' + str(PORT)
URL_FUNC = URL_HOME + '/func'
INFO = '''Choose an action:
1. Get home page: home.
2. Get prediction with GET request by providing one feature vector: vector. 
3. Get predictions with POST request by providing multiple feature vectors: matrix.
Press CTRL + C to exit the client app.
'''

def send_request(request_type: str, url: str, data: Union[str, None] = None, params: dict = None):
    print(f'Sending {request_type} request...', end='')
    if request_type == 'GET':
        r = requests.get(url, json=data, params=params)
    else:
        r = requests.post(url, json=data, params=params)
    print(r.ok)
    if r.headers['content-type'] == 'application/json':
        print('JSON content:\n', r.json())
    elif r.headers['content-type'] == 'text/html':
        print('HTML content:\n', r.text)
    else:
        print('Text content:\n', r.text)
    print('------------------------------------------------')

def get_home_page():
    send_request(request_type='GET', url=URL_HOME)

def get_vector_prediction():
    vector = dict()
    for pair in input('provide URL parameters: ').replace(' ', '').split(','):
        key, value = pair.split('=')
        vector[key] = value
    send_request(request_type='GET', url=URL_FUNC, params=vector)

def get_matrix_prediction():
    matrix = json.loads(input('Provide feature vectors: '))
    send_request(request_type='POST', url=URL_FUNC, data=matrix)

def main():
    print('Open client app...')
    while True:
        try:
            action = input(INFO)
            action = action.replace(' ', '').lower()
            if action == 'home':
                get_home_page()
            elif action == 'vector':
                get_vector_prediction()
            elif action == 'matrix':
                get_matrix_prediction()
            else:
                print('Invalid action. Please try again.')
        except KeyboardInterrupt:
            print('Close client app... ')
            break

if __name__ == '__main__':
    main()