"""
This package creates GeoJSON outputs for the POINT and POLYGON geometric objects, used by Tableau in visualizing
scenario 1

Inputs:
"'outputs' + os.sep + 'geodata_all_scenarios.csv'" - A CSV stored in the outputs directory, created by the csv_parser.py

Outputs:
'geojson_scenario_1_point.geojson' - A GeoJSON file which contains all the information on the 'POINT' geometric objects
for scenario 1
'geojson_scenario_1_polygon.geojson' - A GeoJSON file which contains all the information on the 'POLYGON' geometric
objects for scenario 1
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.cluster import dbscan
import os
import json
from sklearn.metrics.pairwise import haversine_distances
from math import radians
import pathlib
import os


def neighbours(df: pd.DataFrame, cluster: int, max_distance=0.015) -> pd.DataFrame:
    """
    This function takes in a dataframe, with lon lat coordinates, and a cluster number
    and makes a boolean matrix for a single cluster with False if 2 'identificatie' are not neighbours
    and True if they are neighbours

    :return:
        a pandas dataframe, containing 'identificatie' as both row and column names,
        with booleans between those 'identificatie' with True = neighbour and False = no neighbour
    """

    # Call gen_distance_matrix to get the distance matrix for the cluster
    distance_matrix = gen_distance_matrix(df, cluster)

    # Return boolean array with True if distance < max_distance, default max_distance is 15m
    # and if the distance is not equal to zero, so not taking itself as neighbour
    df_ones = (distance_matrix <= max_distance).astype(int)
    np.fill_diagonal(df_ones.values, 0)

    return df_ones


def gen_distance_matrix(df: pd.DataFrame, cluster: int) -> pd.DataFrame:
    """
    This function takes in a dataframe, with lon lat coordinates, and a cluster number
    and calculates the distance matrix between points for a single cluster

    :return:
        return_df: a pandas dataframe, containing 'identificatie' as both row and column names,
        with the distance between those 'identificatie' as value in kilometers
    """

    # Take only the data belonging to the cluster we want
    only_cluster = df[df['Cluster'] == cluster]

    # Take out only the coordinates
    cluster_coords = only_cluster[['x_coordinate', 'y_coordinate']]

    # Calculate radians for the haversine function
    in_radians = [[radians(coord[0]), radians(coord[1])] for coord in cluster_coords.values]

    # Calculate distances with the haversine function, and multiply by the circumference of earth to get kilometers
    result = haversine_distances(in_radians) * 6371.0088

    # Add 'identificatie' as column and row names
    return_df = pd.DataFrame(result, columns=only_cluster.identificatie_vbo, index=only_cluster.identificatie_vbo)

    return return_df


def rd_to_wgs(x: float, y: float) -> [float, float]:
    """
    Convert rijksdriehoekcoordinates (used by the Municipality) into WGS84 coordinates (used by Tableau)
    """

    X0 = 155000
    Y0 = 463000
    PHI0 = 52.15517440
    LAM0 = 5.38720621

    if isinstance(x, (list, tuple)):
        x, y = x

    pqk = [(0, 1, 3235.65389),
           (2, 0, -32.58297),
           (0, 2, -0.24750),
           (2, 1, -0.84978),
           (0, 3, -0.06550),
           (2, 2, -0.01709),
           (1, 0, -0.00738),
           (4, 0, 0.00530),
           (2, 3, -0.00039),
           (4, 1, 0.00033),
           (1, 1, -0.00012)]

    pql = [(1, 0, 5260.52916),
           (1, 1, 105.94684),
           (1, 2, 2.45656),
           (3, 0, -0.81885),
           (1, 3, 0.05594),
           (3, 1, -0.05607),
           (0, 1, 0.01199),
           (3, 2, -0.00256),
           (1, 4, 0.00128),
           (0, 2, 0.00022),
           (2, 0, -0.00022),
           (5, 0, 0.00026)]

    dx = 1E-5 * (x - X0)
    dy = 1E-5 * (y - Y0)

    phi = PHI0
    lam = LAM0

    for p, q, k in pqk:
        phi += k * dx ** p * dy ** q / 3600

    for p, q, l in pql:
        lam += l * dx ** p * dy ** q / 3600

    return [lam, phi]


def run_dbscan(geodata: pd.DataFrame, eps: int, minpts: int) -> pd.DataFrame:
    """
    Used to actually run the DBSCAN Algorithm, using the user-supplied Epsilon
    """

    eps = eps / 1000

    kms_per_radian = 6371.0088
    epsilon = eps / kms_per_radian
    minsamples = minpts
    radians = np.radians(geodata[['x_coordinate', 'y_coordinate']])

    # DBSCAN
    preds = dbscan(radians, eps=epsilon, min_samples=minsamples, algorithm='ball_tree', metric='haversine')[1]
    dbscan_coords = np.append(radians, preds.reshape(-1, 1), axis=1)
    pd.DataFrame(dbscan_coords).plot(x=1, y=0, kind="scatter", c=2, colorbar=True,
                                     title="DBSCAN (eps= 15m, min_points=5)", marker="+", colormap="tab20b")

    geodata['Cluster'] = pd.DataFrame(dbscan_coords)[2]

    return geodata


def add_x_y_dbscan(geodata: pd.DataFrame) -> pd.DataFrame:

    x_coordinates = []
    y_coordinates = []

    for entry in geodata['geometrie_vbo']:
        entry = entry.strip("POINT()")
        entry = entry[2:]
        entry = entry.split()
        y, x = rd_to_wgs(float(entry[0]), float(entry[1]))
        x_coordinates.append(x)
        y_coordinates.append(y)

    geodata["x_coordinate"] = np.array(x_coordinates).astype(np.float64)
    geodata["y_coordinate"] = np.array(y_coordinates).astype(np.float64)

    return geodata


def generate_risico_scores(geodata: pd.DataFrame) -> pd.DataFrame:

    # Generate random values (risicoscores)

    a, b = 0, 1
    mu, sigma = 0.7, 0.15
    dist = stats.truncnorm((a - mu) / sigma, (b - mu) / sigma, loc=mu, scale=sigma)

    values = dist.rvs(len(geodata))

    geodata["score"] = values

    return geodata


def read_geodata() -> pd.DataFrame:

    # Read in geodata
    PROJECT_PATH = pathlib.Path(os.path.abspath(__file__)).parent.parent
    geodata_path = os.path.join(PROJECT_PATH, "outputs", "geodata_all_scenarios.csv")
    geodata = pd.read_csv(geodata_path)

    return geodata


def parse_polygon_coordinates(geodata: pd.DataFrame) -> [[float, float]]:
    output_series = geodata['geometrie_pand']
    output_list = output_series.to_list()
    output_poly = []

    for poly in output_list:
        coords = poly[10:-2].split(',')
        coords = [ele.lstrip() for ele in coords]
        coords = [ele.split(' ') for ele in coords]
        coords = [(float(ele[0].replace('(', '')), float(ele[1].replace(')', ''))) for ele in coords]
        coords = [list(rd_to_wgs(ele[0], ele[1])) for ele in coords]
        coords_final = []
        for ele in coords:
            coords_final.append([float("{:.5f}".format(ele[0])), float("{:.5f}".format(ele[1]))])

        output_poly.append(coords_final)

    return output_poly


def parse_polygon_coordinates_(geodata: pd.DataFrame) -> [[float, float]]:
    output_series = geodata['geometrie_pand']
    output_list = output_series.to_list()
    output_poly = []

    for index, poly in enumerate(output_list):

        final_string = ''
        coords_string = ''

        coords = poly[10:-2].split(',')
        coords = [ele.lstrip() for ele in coords]
        coords = [ele.split(' ') for ele in coords]
        coords = [(float(ele[0].replace('(', '')), float(ele[1].replace(')', ''))) for ele in coords]
        coords = [tuple(rd_to_wgs(ele[1], ele[0])) for ele in coords]
        coords_final = []

        temp = ()

        for ind, ele in enumerate(coords):
            coords_final.append((float("{:.5f}".format(ele[0])), float("{:.5f}".format(ele[1]))))
            if ind == 0:
                temp = ele

        coords_final.append((float("{:.5f}".format(temp[0])), float("{:.5f}".format(temp[1]))))

        final_string = 'POLYGON (('
        coords_string = ''
        for pair in coords_final:
            coords_string += str(pair[0]) + ' ' + str(pair[1]) + ', '

        coords_string = coords_string[:-2]

        final_string = final_string + coords_string + '))'

        output_poly.append(final_string)

    return output_poly


def parse_point_coordinates(geodata: pd.DataFrame) -> [[float, float]]:

    # First, select the relevant columns for the POINT coordinates
    output_series_x = geodata['x_coordinate']
    output_series_y = geodata['y_coordinate']

    # Then create two lists containing these coordinates
    output_list_x = output_series_x.to_list()
    output_list_y = output_series_y.to_list()

    # Create a list to contain the merged points
    output_poly = []

    # Loop over both lists, and append the coordinates
    for x, y in zip(output_list_x, output_list_y):
        output_poly.append([y, x])

    return output_poly


def create_geojson(polygon_coords: [[float, float]], cluster_list: list, point_coords: [[float, float]], score: [float],
                   avg_neighbour_score: [float], vbo_id_list: [str], bouwjaar_list: [str], stadsdeel_list: [str],
                   gebruiksdoel: [str], rel_score_veroorzaker, rel_score_slachtoffer, rel_cluster_veroorzaker,
                   rel_cluster_slachtoffer) -> (dict, dict):
    # Create GEOJSON objects from each element in output_poly
    feature_collection_polygon = {
        "type": "FeatureCollection",
        "features": []
    }

    feature_collection_point = {
        "type": "FeatureCollection",
        "features": []
    }

    for index, element in enumerate(polygon_coords):
        output = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                        "coordinates": [
                            element
                        ]
                    },
            "properties": {
                "cluster": str(cluster_list[index]),
                "vbo_id": str(vbo_id_list[index]),
                "bouwjaar": str(bouwjaar_list[index])
            }
        }
        feature_collection_polygon["features"].append(output)

    for i, ele in enumerate(point_coords):
        output = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates":
                    ele

            },
            "properties": {
                "cluster": str(cluster_list[i]),
                "score": str(score[i]),
                "neighbour_score": str(avg_neighbour_score[i]),
                "vbo_id": str(vbo_id_list[i]),
                "stadsdeel": str(stadsdeel_list[i]),
                "gebruiksdoel": str(gebruiksdoel[i]),
                "rel_score_veroorzaker": str(rel_score_veroorzaker[i]),
                "rel_score_slachtoffer": str(rel_score_slachtoffer[i]),
                "rel_cluster_veroorzaker": str(rel_cluster_veroorzaker[i]),
                "rel_cluster_slachtoffer": str(rel_cluster_slachtoffer[i])
            }
        }
        feature_collection_point["features"].append(output)

    return feature_collection_polygon, feature_collection_point


def write_geojson(feature_collection: dict, geometry: str):
    with open(f'geojson_scenario_1_{geometry}.geojson', 'w') as f:
        json.dump(feature_collection, f)


def average_neighbour_scores(df: pd.DataFrame) -> ([int], [int]):

    # Get list of all cluster numbers
    cluster_numbers = df['Cluster'].unique().astype(int).tolist()
    # Create lists for features
    all_average_risk_score = []
    neighbour_counts = []

    for cluster in cluster_numbers:

        # Make every cluster score of cluster -1 0, because much computational power is needed for a cluster
        # of length 25426 while cluster -1 represents the vbo's without a cluster.
        if cluster == -1:
            average_risk_score = [0 for x in range(len(df[df['Cluster'] == -1]))]
            neighbour_count = [0 for x in range(len(df[df['Cluster'] == -1]))]

        else:
            # Get the scores of the vbo's in the adjacency matrix
            column_scores = {vbo: df['score'][df['identificatie_vbo'] == vbo] for vbo in neighbours(df, cluster).keys()}

            # Temporarily list to store risk scores of neighbours
            neighbour_scores = [0 for score in column_scores]

            for vbo, score in column_scores.items():
                # Multiply the risk score with the adjacency matrix to get the risk scores of the neighbours
                vbo_column = neighbours(df, cluster)[vbo].apply(lambda x: x * score).values.flatten().tolist()
                # Adding up the risk scores of the neighbours
                neighbour_scores = [x + y for x, y in zip(neighbour_scores, vbo_column)]

            # Get the count of neighbours of a vbo
            neighbour_count = neighbours(df, cluster).apply(sum).values.flatten().tolist()
            # Calculate average with the count of non zero neighbour scores, because a vbo can't be it's own neighbour
            average_risk_score = [x / y for x, y in zip(neighbour_scores, neighbour_count)]

        # Append features to list (average risk score, neighbour count)
        all_average_risk_score.extend(average_risk_score)
        neighbour_counts.extend(neighbour_count)

    return all_average_risk_score, neighbour_counts


def calculate_neighbours(geodata: pd.DataFrame) -> pd.DataFrame:

    # Make a subset of clusters and non-clusters
    geodata_cluster = geodata.loc[geodata['Cluster'] > -1]
    geodata_no_cluster = geodata.loc[geodata['Cluster'] == -1]
    geodata_no_cluster['average_neighour_score'] = 0
    geodata_no_cluster['neighbour_count'] = 0

    # Add calculation for average neighbour scores
    adjusted_scores = average_neighbour_scores(geodata_cluster)
    geodata_cluster['average_neighour_score'] = adjusted_scores[0]
    geodata_cluster['neighbour_count'] = adjusted_scores[1]
    geodata = pd.concat([geodata_no_cluster, geodata_cluster]).sort_index()

    return geodata


def calculate_metrics(geodata: pd.DataFrame) -> pd.DataFrame:

    # Case 1 (veroorzaker van risico)
    # Variant 1 / Hogere risicoscore dan neighbours = hogere relatieve score
    geodata['rel_score_veroorzaker_1'] = geodata['score'] / geodata['average_neighour_score']
    geodata['rel_score_veroorzaker_2'] = geodata['neighbour_count'] + 1
    geodata['rel_score_veroorzaker_3'] = geodata['neighbour_count'] / geodata['rel_score_veroorzaker_2']
    geodata['rel_score_veroorzaker'] = geodata['rel_score_veroorzaker_1'] * geodata['rel_score_veroorzaker_3']

    # Variant 2 / Meer neighbours = hogere relatieve score
    geodata['rel_score_slachtoffer_1'] = geodata['average_neighour_score'] / geodata['score']
    geodata['rel_score_slachtoffer_2'] = geodata['neighbour_count'] + 1
    geodata['rel_score_slachtoffer_3'] = geodata['neighbour_count'] / geodata['rel_score_slachtoffer_2']
    geodata['rel_score_slachtoffer'] = geodata['rel_score_slachtoffer_1'] * geodata['rel_score_slachtoffer_3']

    # Case 2 (slachtoffer van risico)
    # Variant 1 / geeft een hogere score als eigen score lager is dan neighbours gemiddeld
    geodata['cluster_average'] = geodata.groupby('Cluster')['score'].transform('mean')
    geodata['rel_cluster_veroorzaker'] = geodata['score'] / geodata['cluster_average']

    # Variant 2 / met dezelfde n_neighbour factor als in case 1
    geodata['rel_cluster_slachtoffer'] = geodata['cluster_average'] / geodata['score']

    # Now fill the NA Values
    geodata = geodata.fillna(value=0)

    return geodata


def run_script(eps: int, minpts: int):

    # First read in the geodata, stored in a CSV file in the 'outputs' directory
    geodata = read_geodata()

    # Extract an X and Y coordinate from the 'Point' column, in the Geodata DataFrame and add these as two seperate cols
    geodata = add_x_y_dbscan(geodata)

    # Generate random risk scores, using the same distribution as the test data and add this as a seperate column
    geodata = generate_risico_scores(geodata)

    # Create clusters of VBO's, based on their X and Y coordinates
    geodata = run_dbscan(geodata, eps, minpts)

    # Construct a list containing polygon coordinates, in the right format (longitude, latitude)
    polygon_coords = parse_polygon_coordinates(geodata)

    # Construct a list containing Point coordinates, in the right format
    point_coords = parse_point_coordinates(geodata)

    # Create a list containing all the cluster values
    cluster_list = geodata['Cluster'].to_list()

    # Create a list containing the identificatie_vbo
    vbo_id_list = geodata['identificatie_vbo'].to_list()

    # Create a list containing scores
    score = geodata['score'].to_list()

    # Create a list containing bouwjaar
    bouwjaar = geodata['oorspronkelijkBouwjaar'].to_list()

    # Create a list containting stadsdeel
    stadsdeel = geodata['ligtIn:GBD.SDL.naam'].to_list()

    # Create a list to hold the gebruiksdoel
    gebruiksdoel = geodata['gebruiksdoel'].to_list()

    # Calculate the number of neighbours
    geodata = calculate_neighbours(geodata)

    # Create a list containing average_neighbour_scores
    avg_neighbour_score = geodata['average_neighour_score'].to_list()

    # Now calculate the metrics used by the Visualisation
    geodata = calculate_metrics(geodata)

    # Now fill the NA Values
    geodata = geodata.fillna(value=0)

    # Now transform these 4 columns to lists
    rel_score_veroorzaker = geodata['rel_score_veroorzaker'].to_list()
    rel_score_slachtoffer = geodata['rel_score_slachtoffer'].to_list()
    rel_cluster_veroorzaker = geodata['rel_cluster_veroorzaker'].to_list()
    rel_cluster_slachtoffer = geodata['rel_cluster_slachtoffer'].to_list()

    # Adjust the X and Y coordinates to align with the Tableau required format
    geodata['x_coordinate_temp'] = geodata['x_coordinate']
    geodata['y_coordinate_temp'] = geodata['y_coordinate']
    geodata['x_coordinate'] = geodata['y_coordinate_temp']
    geodata['y_coordinate'] = geodata['x_coordinate_temp']
    geodata.drop(columns=['x_coordinate_temp', 'y_coordinate_temp'], inplace=True)

    # Create a Feature Collection of Polygons with some additional information (such as the Cluster)
    feature_collection_polygon = create_geojson(polygon_coords, cluster_list, point_coords, score, avg_neighbour_score,
                                                vbo_id_list, bouwjaar, stadsdeel, gebruiksdoel, rel_score_veroorzaker,
                                                rel_score_slachtoffer, rel_cluster_veroorzaker, rel_cluster_slachtoffer)[0]
    feature_collection_point = create_geojson(polygon_coords, cluster_list, point_coords, score, avg_neighbour_score,
                                              vbo_id_list, bouwjaar, stadsdeel, gebruiksdoel, rel_score_veroorzaker,
                                                rel_score_slachtoffer, rel_cluster_veroorzaker, rel_cluster_slachtoffer)[1]

    # Finally, write the Feature Collection to a GeoJSON file
    write_geojson(feature_collection_polygon, 'polygon')
    write_geojson(feature_collection_point, 'point')


if __name__ == "__main__":

    print("Running Scenario 1")
    print("Please note that the tool has only been tested with a value for Epsilon of 15, and a value for the minimum"
          "amount of VBO's in a Cluster of 5.")
    eps = input("Please provide input for the Epsilon, or press Enter to use a value of 15: ") or 15
    eps = int(eps)
    minpts = input("Please provide input for minimum amount of VBO's in a Cluster, "
                       "or press Enter to use a value of 5: ") or 5
    minpts = int(minpts)
    run_script(eps, minpts)
