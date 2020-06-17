import pathlib
import os
from config import config

class Experiment():

    def __init__(self, run_n=0 ):
        self.run_n = run_n
        self.result_dir = os.path.join(config['result_dir'], 'run_' + str(self.run_n))
        self.model_dir = os.path.join(config['model_dir'],'run_' + str(self.run_n))
        self.fig_dir = os.path.join(config['fig_dir'], 'run_' + str(self.run_n))
        self.fig_dir_event = os.path.join(self.fig_dir,'analysis_event')

    def setup(self, fig_dir=False, result_dir=False, tensorboard_dir=False, model_dir=False):

        if fig_dir:
            pathlib.Path(self.fig_dir).mkdir(parents=True, exist_ok=True)
            self.fig_dir_img = os.path.join(self.fig_dir,'analysis_image')
            pathlib.Path(self.fig_dir_img).mkdir(parents=True, exist_ok=True)
            pathlib.Path(self.fig_dir_event).mkdir(parents=True, exist_ok=True)

        if result_dir:
            pathlib.Path(self.result_dir).mkdir(parents=True, exist_ok=True)

        if tensorboard_dir:
            self.tensorboard_dir = os.path.join(config['tensorboard_dir'],'run_' + str(self.run_n))
            pathlib.Path(self.tensorboard_dir).mkdir(parents=True, exist_ok=True)

        if model_dir:
            pathlib.Path(self.model_dir).mkdir(parents=True, exist_ok=True)

        return self