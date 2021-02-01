"""
This package coordinates and parses GET requests to the Muncipality's API.

NOT USED IN THE DELIVERABLE
"""

# Import Modules
import pandas as pd
import pprint
import src.api_requests as ar


def woonfunctie(gebruiksdoel: list) -> bool:

    # Loop over all gebruiksdoelen of a Verblijfsobject
    for _gebruiksdoel in gebruiksdoel:

        # If any of the gebruiksdoelen are 'woonfunctie', return True
        if _gebruiksdoel == 'woonfunctie':
            return True

    # If none of the gebruiksdoelen are 'woonfunctie', return False
    return False


# Create a pretty printer object to print results
pp = pprint.PrettyPrinter(indent=4)

# Query the API for a certain neighborhood
api_output_buurt = ar.return_buurt('A02c')

# Loop over all 'Verblijfsobjecten' within the 'results' list of the api_output_buurt
for vbo_url in api_output_buurt['results']:

    # Perform a GET request on that particular Verblijfsobject
    vbo_url = vbo_url['_links']['self']['href']
    vbo = ar.return_vbo(vbo_url)

    # Check if the Verblijfsobject has a 'gebruiksdoel' of 'woonfunctie'
    if woonfunctie([vbo['gebruiksdoel']]):
        # If so, do not use this Verblijfsobject, and move on to the next
        break

    # Perform a GET request on the 'panden' category of the Verblijfsobject
    pand_url = vbo['panden']['href']
    pand = ar.return_pand(pand_url)


# Create a list to hold the Coordinates as tuples
coordinates = []

# Iterate over each coordinate (X, Y), stored in the data returned from the API
for feature in pand['geometrie']['coordinates'][0][0]:

    # Convert the X, Y coordinates from
    coordinates_tuple = tuple(feature)
    coordinates.append(coordinates_tuple)

# Create a DataFrame from the coordinates
stad_df = pd.DataFrame(columns=['longitude', 'latitude'], data=coordinates)
BBox = ((stad_df.longitude.min(),   stad_df.longitude.max(),
         stad_df.latitude.min(), stad_df.latitude.max()))