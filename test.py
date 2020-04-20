from environment.board import Diamond
from environment.hex import Hex
from environment.visualizer import Visualizer
import yaml

with open("config.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

hex = Hex(cfg, True)
state = Diamond(hex.size)

state = hex.perform_action(state, (1, 1))
hex.change_player()
state = hex.perform_action(state, (1, 2))
hex.change_player()
state = hex.perform_action(state, (2, 1))
hex.change_player()
state = hex.perform_action(state, (2, 0))
hex.change_player()
state = hex.perform_action(state, (2, 3))
hex.change_player()
state = hex.perform_action(state, (0, 2))
hex.change_player()
state = hex.perform_action(state, (2, 2))
hex.change_player()
state = hex.perform_action(state, (1, 3))
hex.change_player()
state = hex.perform_action(state, (1, 0))

print(hex.game_over(state))

visualizer = Visualizer(state, hex.size, cfg["display"])
visualizer.fill_nodes(state.get_filled_cells())

hex.change_player()
children = hex.generate_child_states(state)
for child in children:
    visualizer.fill_nodes(child[0].get_filled_cells())
