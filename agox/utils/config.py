import configparser
import inspect
import os
from pathlib import Path

class Config:

    def __init__(self, config_path):
        self.parser = configparser.ConfigParser()
        self.parser.read(config_path)
    
def get_default_config_path():
    agox_path = os.path.abspath(__file__)
    agox_path = Path(os.path.dirname(os.path.dirname(os.path.dirname(agox_path))))
    return agox_path / 'config.ini'

config_path = os.environ.get('AGOX_CONFIG_PATH')
if config_path is None:
    config_path = get_default_config_path()

print('Using config file: {}'.format(config_path))

cfg = Config(config_path)


