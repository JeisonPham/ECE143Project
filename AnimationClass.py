from abc import ABC, abstractmethod
import os


class Animation(ABC):
    """
    Animation Library class that will be used as the base template for other plot objects. This way everything
    can call a render function and generate the corresponding visual

    example usage: class StatePlot(Animation)

   :param image_folder: file location to store images that will be converted to video file
   :type image_folder: str
    """

    def __init__(self, image_folder):
        self.image_folder = image_folder

        if not os.path.exists(self.image_folder):
            os.mkdir(self.image_folder)

    @abstractmethod
    def get_next_data(self):
        """
        A generator function that will YIELD the data that is to be plotted. Typical format to be returned is two columns
        or a pandas dataframe.

        Example:

        ZIP CODE    |  Jan  |   Feb   |   Mar   |   Apr   | ...
        -------------------------------------------------------

        92832       |   90  |   80  |   70  |   60  | ...

        will yield (92832, 90) first
        then (92832, 80) second
        then (92832, 70) third
        etc


        :yield: data that is to be plotted
        """
        pass

    @abstractmethod
    def plot(self, data):
        """
        Plots the current data, will be specified at a later date
        :param data: data to be plotted
        :type data: pandas dataframe
        :return: return plotly figure type
        """
        pass

    def render(self, delete_images_after_render, fps, title):
        """
        Renders the plots and saves them to the image folder

        :param delete_images_after_render: bool to either keep or delete images after rendering
        :param fps: fps of video
        :param title: video file name
        :type delete_images_after_render: bool
        :type fps: int
        :type title: str
        :return:
        """

        counter = 0
        gen = self.get_next_data()
        length = next(gen)

        for counter in range(length):
            data = next(gen)
            print(f"Rendering Frame: {counter + 1}")
            fig = self.plot(data)
            fig.write_image(os.path.join(self.image_folder, f'{counter:05d}.png'))

        palcmd = f"ffmpeg -i {os.path.join(self.image_folder, '%05d.png')} -vf palettegen palette.png"
        cmd = f"ffmpeg -framerate {fps} -i {os.path.join(self.image_folder, '%05d.png')} -i palette.png -lavfi paletteuse {title}.gif -y"
        os.system(palcmd)
        os.system(cmd)

        if delete_images_after_render:
            for i in range(length):
                path = os.path.join(self.image_folder, f"{i:05d}.png")
                if os.path.exists(path):
                    os.remove(path)


