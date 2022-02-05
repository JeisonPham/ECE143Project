from dataclasses import dataclass

import pandas as pd
import numpy as np
import plotly.express as px

from urllib.request import urlopen
import json
import quandl


def get_by_state(x, id):
    """
    Helper function for filter that selects data by state for geojson
    :param x: geojson data
    :param id: id (str)
    :return: bool
    """

    assert isinstance(id, str)
    assert isinstance(x, dict)

    if x["properties"]['STATE'] == id:
        return True
    return False


def get_center(features):
    """
    Returns the center of the map
    :param features: features of geojson
    :return: dict of center
    """

    assert isinstance(features, dict)

    coords = []
    for feature in features:
        geometry = np.array(feature['geometry']['coordinates'])
        if len(geometry.shape) == 3:
            coords.append(geometry[0])

    coords = np.concatenate(coords)
    return {"lon": int(coords[:, 0].mean()), "lat": int(coords[:, 1].mean())}




class DataVisualization:
    def __init__(self, data):
        with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
            self.counties = json.load(response)

        self.data = data
        quandl.ApiConfig.api_key = 'MyCDqbqLL2vis32f4v8r'

        self.region_id = self.generate_StateData(quandl.get_table("ZILLOW/REGIONS", paginate=True))



    def plot_state(self, id):
        """
        Plots a state given an id
        :param id: id of state
        :return:
        """

        fig = px.choropleth_mapbox(self.data, geojson=self.counties, locations='fips', color='unemp',
                                   color_continuous_scale="Viridis",
                                   range_color=(0, 12),
                                   zoom=4.9, center=get_center(self.counties['features']),
                                   opacity=0.5,
                                   mapbox_style="white-bg",
                                   labels={'unemp': 'unemployment rate'}
                                   )
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        fig.show()




@dataclass
class RegionData:
    id: int
    value: str

@dataclass
class StateData:
    abr_name: str
    zip: list
    city: list
    county: list

class HouseData:
    def __init__(self, data):
        quandl.ApiConfig.api_key = 'MyCDqbqLL2vis32f4v8r'

        self.data = quandl.get_table("ZILLOW/REGIONS", paginate=True)
        self.data[['value', 'state', 'name', 'etc']] = self.data['region'].str.split("; ", expand=True)
        self.region_id = self.generate_StateData(self.data)

    def generate_StateData(self, data):
        states = data[data['region_type'] == 'state']

        state_info = dict()
        for state in states['region']:
            state = state.split(";")
            full_name = state[0]
            abr = state[1][1:]

            zip = self.get_zips(data, full_name, abr)


    def get_zips(self, data, full_name, abr):
        zips = data[data['region_type'] == 'zip']
        zip_list = list()


        for i, row in zips.iterrows():
            region = row['region'].split("; ")
            if abr in region:
                zip_list.append(RegionData(row['region_id'], region[0]))
        return zip_list


if __name__ == "__main__":
    data = HouseData(None)