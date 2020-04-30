# |---------- Environment parameters ----------|
board_size = 5  # between 3 and 10
player = (
    3  # starting player option, 1, 2 or 3 (mixed) Vet ikke om vi skal ha med 3
)
verbose = False  # Whether the details of moves for each game is shown

node_size = 1500
initial_color = "white"
edge_color = "darkgrey"
p1_color = "lightsteelblue"
p2_color = "darkseagreen"
delay = 0.5  # seconds
display_last_game = True  # True or False
plot_window_size = 20

# |------------- Agent parameters -------------|

m = 10  # interval of ANETs to be cached for playing TOPP
g = 25  # number of games played in TOPP

activation_hidden = "relu"
optimizer = "adam"

episodes = 500  # number of episodes
simulations = 500
epochs = 50
dimensions = [64, 64, 64]
learning_rate = 0.0005
batch_size = 128
max_buffer_length = 2500

save_directory = "ep{}_sim{}_epo{}_dim{}_lr{}_bs{}_max{}".format(
    episodes,
    simulations,
    epochs,
    "".join([str(i) for i in dimensions]),
    learning_rate,
    batch_size,
    max_buffer_length,
)

load_directory = save_directory
