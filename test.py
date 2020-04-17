
from environment.board import Diamond
from environment.hex import Hex
from environment.visualizer import Visualizer
import yaml

with open("config.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

hex = Hex(cfg, True)
hex.perform_action((0, 0))
hex.change_player()
hex.perform_action((0, 4))
hex.change_player()
hex.perform_action((2, 2))

print(hex.board.get_open_cells())
print(hex.board.get_filled_cells())

visualizer = Visualizer(hex.board, hex.size, cfg["display"])
visualizer.fill_nodes(hex.board.get_filled_cells())
