from environment.board import Diamond
from environment.hex import Hex
from environment.visualizer import Visualizer

with open("config.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

hex = Hex(cfg, True)
hex.perform_action(0,0, (0,1))
