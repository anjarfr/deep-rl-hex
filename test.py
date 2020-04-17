from environment.board import Diamond
from environment.hex import Hex
from environment.visualizer import Visualizer
import yaml

with open("config.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

hex = Hex(cfg, True)
hex.perform_action((1, 1))
hex.change_player()
hex.perform_action((1, 2))
hex.change_player()
hex.perform_action((2, 1))
hex.change_player()
hex.perform_action((3, 2))
hex.change_player()
hex.perform_action((2, 3))
hex.change_player()
hex.perform_action((0, 2))
hex.change_player()
hex.perform_action((2, 2))
hex.change_player()
hex.perform_action((1, 3))
hex.change_player()
hex.perform_action((1, 0))
print(hex.player)

visualizer = Visualizer(hex.board, hex.size, cfg["display"])
visualizer.fill_nodes(hex.board.get_filled_cells())
