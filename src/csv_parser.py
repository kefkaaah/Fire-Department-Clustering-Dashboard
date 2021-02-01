"""
This package creates the CSV Files used by the Scenario 1, 2, and 3 clustering Algorithms.

Inputs:
"BAG_pand_Actueel.csv" - Retrievable using FileZilla
"BAG_verblijfsobject_Actueel.csv" - Retrievable using FileZilla

Outputs:
"geodata_all_scenarios.csv" - Used as input for the Scenario 1, 2 and 3 clustering Algorithms
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from shapely import wkt


def csv_read() -> (pd.DataFrame, pd.DataFrame):
    """
    This function reads in the two CSV files containing information on the Verblijfsobjecten and Panden, in Amsterdam.
    These files should be in the same directory as the script.

    :return:
        pand: Contains information on the Panden in Amsterdam
        vbo: Contains information on the Verblijfsobjecten in Amsterdam

    """

    # Create file path, using the separator depending on the OS (e.g. '/' for MacOS / Linux, and '\' for windows)
    cwd = os.getcwd()
    PAND = cwd + os.sep + 'inputs' + os.sep + 'BAG_pand_Actueel.csv'
    VBO = cwd + os.sep + 'inputs' + os.sep + 'BAG_verblijfsobject_Actueel.csv'

    # Read in both CSV Files
    pand = pd.read_csv(PAND, usecols=['identificatie', 'geometrie', 'oorspronkelijkBouwjaar'],
                       encoding='utf-8', sep=';', low_memory=False)
    vbo = pd.read_csv(VBO, usecols=['identificatie', 'gebruiksdoel', 'ligtIn:BAG.PND.identificatie', 'geometrie',
                                    'ligtIn:GBD.SDL.naam'], encoding='utf-8', sep=';',
                      low_memory=False).rename(columns={'identificatie': 'identificatie_vbo'})

    return pand, vbo


def format_dataframe(pand: pd.DataFrame, vbo: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    This function parses and then formats the content of the 'pand' and 'vbo' DataFrames, as to ensure that they
    are compatible with the GeoPandas data format

    :param pand: Contains information on the Panden in Amsterdam
    :param vbo: Contains information on the Verblijfsobjecten in Amsterdam
    :return:
        pand: Contains information on the Panden in Amsterdam, parsed and formatted
        vbo: Contains information on the Verblijfsobjecten in Amsterdam, parsed and formatted
    """

    # Drop all 'Verblijfsobjecten' that have a 'woonfunctie' as 'gebruiksdoel'
    vbo = vbo[vbo['gebruiksdoel'] != 'woonfunctie']

    # Remove leading 0 from the 'ligtIn:BAG.PND.identificatie' column
    vbo['ligtIn:BAG.PND.identificatie'] = vbo['ligtIn:BAG.PND.identificatie'].str.lstrip('0')

    # Set all columns of both DataFrames to String ('object')
    pand = pand.astype('str')
    vbo = vbo.astype('str')

    return pand, vbo


def create_geodata(pand: pd.DataFrame, vbo: pd.DataFrame) -> (gpd.GeoSeries, pd.DataFrame):
    """
    This function merges the 'vbo' and 'pand' DataFrames, and creates the necessary object to plot the geographical
    information contained in the merged DataFrame

    :param pand: Contains information on Panden in Amsterdam, parsed and formatted
    :param vbo: Contains information on Verblijfsobjecten in Amsterdam, parsed and formatted
    :return:
        polygon: A GeoSeries object, containing geographical information on Panden in Amsterdam
        geodata: A DataFrame containing the merged vbo and pand DataFrames
    """

    # Merge the two dataframes (pand and vbo) together, using the 'ligtIn:BAG.PND.identificatie' and 'identificatie'
    geodata = vbo.merge(pand, left_on='ligtIn:BAG.PND.identificatie',right_on='identificatie',
                        how='left').rename(columns={'geometrie_x': 'geometrie_vbo', 'geometrie_y': 'geometrie_pand'})

    # Drop rows that contain 'nan' values in the 'geometrie_pand' column
    geodata = geodata.dropna(subset=['geometrie_pand'])

    # Now modify the 'geometrie_pand' column, to be compatible with GeoPandas
    geodata['geometrie_pand'] = geodata['geometrie_pand'].apply(wkt.loads)

    # Create a GeoPandas GeoDataFrame in order to plot the geometrie_pand coordinates on a map
    geo_df = gpd.GeoDataFrame(geodata, geometry='geometrie_pand')

    # Create a GeoSeries object, containing the 'geometrie_pand' coordinates
    polygon = gpd.GeoSeries(geo_df.geometrie_pand)

    return geodata


def create_plot() -> gpd.GeoSeries:
    """
    This function reads in the 'geodata' CSV file, and uses it to create a plot of Panden in Amsterdam

    :return:
        polygon: A GeoPandas GeoSeries object, containing the Polygons representing Panden in Amsterdam
    """

    # Read in the CSV containing the geodata of Panden in Amsterdam
    geodata = pd.read_csv('geodata_full.csv')

    # Now modify the 'geometrie_pand' column, to be compatible with GeoPandas
    geodata['geometrie_pand'] = geodata['geometrie_pand'].apply(wkt.loads)

    # Create a GeoPandas GeoDataFrame in order to plot the geometrie_pand coordinates on a map
    geo_df = gpd.GeoDataFrame(geodata, geometry='geometrie_pand')

    # Create a GeoSeries object, containing the 'geometrie_pand' coordinates
    polygon = gpd.GeoSeries(geo_df.geometrie_pand)

    # Create plot, using the polygon
    polygon.plot()
    plt.show()

    return polygon


def run_csv_parser():

    # Read in the CSV's (Pand and Verblijfsobject)
    pand, vbo = csv_read()
    pand, vbo = format_dataframe(pand, vbo)
    geodata = create_geodata(pand, vbo)

    # Create a CSV from the geodata, and write
    geodata.to_csv('geodata_all_scenarios.csv', header=True, index=False)


if __name__ == "__main__":
    run_csv_parser()
