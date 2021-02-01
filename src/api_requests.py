"""
This package contains requests (GET) to and from the Muncipality's API. It is used by the extract_api.py script.

NOT USED IN THE DELIVERABLE
"""

import requests


def return_buurt(code):

    # Make a GET request to the API
    response = requests.get(f"https://api.data.amsterdam.nl/bag/v1.1/verblijfsobject/?buurt__vollcode={code}")

    # Convert the response to a JSON file
    return response.json()


def return_vbo(vbo_url):

    # Make a GET request to the API
    response = requests.get(vbo_url)

    # Convert the response to a JSON file
    return response.json()


def return_pand(pand_url):

    # Make a GET request to the API
    response = requests.get(pand_url + '&detailed=1')

    # Convert the response to a JSON file
    return response.json()


def return_stad(code):

    # Make a GET request to the API
    response = requests.get(f"https://api.data.amsterdam.nl/bag/v1.1/woonplaats/{code}/")

    # Convert the response to a JSON file
    return response.json()


def return_pand():
    pass


def _url(path):

    return 'https://api.data.amsterdam.nl/bag/v1.1/' + path