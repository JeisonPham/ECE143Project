import plotly.express as px
from AnimationClass import Animation
from urllib.request import urlopen
import json
import pandas as pd
import numpy as np

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


class ExamplePlot(Animation):
    def __init__(self, *args, **kwargs):
        super(ExamplePlot, self).__init__(*args, **kwargs)

        with urlopen("https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json") as response:
            self.counties = json.load(response)

        def filter_func(x, id):
            if x["properties"]['STATE'] == id:
                return True
            return False

        self.counties['features'] = list(filter(lambda x: filter_func(x, "06"), self.counties['features']))

        self.df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/fips-unemp-16.csv",
                              dtype={"fips": str})

    def get_next_data(self):
        yield 5 # return length first
        for i in range(5):
            data = self.df.copy()
            data['unemp'] *= i
            yield data

    def plot(self, data):
        fig = px.choropleth_mapbox(data, geojson=self.counties, locations='fips', color='unemp',
                                   color_continuous_scale="Viridis",
                                   range_color=(0, 12),
                                   zoom=4.9, center={'lon': -120, 'lat': 37},
                                   opacity=0.5,
                                   mapbox_style="white-bg",
                                   labels={'unemp': 'unemployment rate'}
                                   )
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        return fig


if __name__ == "__main__":
    example = ExamplePlot("images")
    example.render(delete_images_after_render=False,
                   fps=1,
                   title="Example")
