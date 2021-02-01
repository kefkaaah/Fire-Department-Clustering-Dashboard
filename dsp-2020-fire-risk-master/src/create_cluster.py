"""
This package can be used for manually creating clusters.

NOT USED IN THE DELIVERABLE
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from shapely import wkt
import networkx
from networkx.algorithms.components.connected import connected_components


def create_intersections():

    geodata = pd.read_csv('output_met_poly_OMGEDRAAID_V2.csv')

    # Create the correct Polygon format
    geodata["geometry"] = geodata["geometrie_pand"].apply(wkt.loads)

    # create the geopandas DF needed for functionality
    geodata = gpd.GeoDataFrame(geodata, crs ="epsg:4326")

    # Convert to CRS 3395 (used in tutorial to be able to use buffer of 5 meter)
    geodata["geometry_buffed"] = geodata["geometry"].to_crs('epsg:3395').buffer(5)

    # Convert back to the epsg code(?) of Netherlands of wg84 system
    geodata["geometry_buffed"] = geodata["geometry_buffed"].to_crs("epsg:4326")

    # drop all vbos that share the same building geometry
    geodata = geodata.drop_duplicates(subset=["ligtIn:BAG.PND.identificatie"])

    # Convert buffed with 5 meters geometry back to wg 84 system
    geodata["geometry_buffed"] = geodata["geometry_buffed"].to_crs("epsg:4326")

    # also convert geometry to 4326 system
    geodata["geometry"] = geodata["geometry"].to_crs("epsg:4326")

    # Create df of buffed polygons
    df1= gpd.GeoDataFrame(geodata[["geometry_buffed", "ligtIn:BAG.PND.identificatie"]])

    # create df of non buffed polygon to see if buffed polygon overlap with non-buffed
    df2 = gpd.GeoDataFrame(geodata[["geometry", "ligtIn:BAG.PND.identificatie"]])

    # Create 2 df to be used in intersect function
    df1["geometry"] = df1["geometry_buffed"].to_crs("epsg:4326")
    df2["geometry"] = df2["geometry"].to_crs("epsg:4326")

    # Function needs a df with column name geometry so original name is copied and dropped
    df1.drop(columns=["geometry_buffed"], axis=1, inplace=True)

    intersections = gpd.overlay(df1, df2, how="intersection")

    return geodata, intersections


def create_clusters(intersections):

    intersections = intersections[
        intersections['ligtIn:BAG.PND.identificatie_1'] != intersections['ligtIn:BAG.PND.identificatie_2']]

    intersect_1 = intersections['ligtIn:BAG.PND.identificatie_1'].to_list()
    intersect_2 = intersections['ligtIn:BAG.PND.identificatie_2'].to_list()

    clusters = dict()

    for index in range(len(intersect_1)):
        if intersect_1[index] not in clusters:
            clusters[intersect_1[index]] = [intersect_2[index]]

        else:
            clusters[intersect_1[index]].append(intersect_2[index])

    input_list = []

    for key, elements in clusters.items():

        temp_list = []
        temp_list.append(key)
        for ele in elements:
            temp_list.append(ele)

        input_list.append(temp_list)

    G = to_graph(input_list)

    return G


def to_graph(l):
    G = networkx.Graph()
    for part in l:
        # each sublist is a bunch of nodes
        G.add_nodes_from(part)
        # it also imlies a number of edges:
        G.add_edges_from(to_edges(part))
    return G


def to_edges(l):
    """
        treat `l` as a Graph and returns it's edges
        to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
    """
    it = iter(l)
    last = next(it)

    for current in it:
        yield last, current
        last = current


geodata_, intersections_ = create_intersections()
g = create_clusters(intersections_)

dfs = []

for index, item in enumerate(networkx.algorithms.connected_components(g)):
    temp = pd.DataFrame(data=[item]).transpose()
    temp['cluster'] = index + 1
    dfs.append(temp)


final_df = pd.concat(dfs).reset_index(drop=True)

output = pd.merge(final_df, geodata_, left_on=[0], right_on=['identificatie'], how='right')
output = output.fillna(value=-1)

output.to_csv('eerste_handmatige.csv', index=False)