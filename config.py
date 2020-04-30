# |---------- Environment parameters ----------|
board_size = 3  # between 3 and 10
player = 3  # starting player option, 1, 2 or 3 (mixed)
verbose = True  # Whether the details of moves for each game is shown
stochastic = False

node_size = 1500
initial_color = "white"
edge_color = "darkgrey"
p1_color = "lightsteelblue"
p2_color = "darkseagreen"
delay = 0.5  # seconds
display_last_game = True  # True or False
plot_window_size = 20

# |------------- Agent parameters -------------|

episodes = 10  # number of episodes
simulations = 800  # number of simulations
m = 4  # interval of ANETs to be cached for playing TOPP

learning_rate = 0.0005
dimensions = [128, 128, 64, 64]
activation_hidden = "relu"  # linear sigmoid tanh relu
optimizer = "adam"  # adagrad sgd rmsprop adam
epochs = 50
batch_size = 32
max_buffer_length = 5000
epsilon = 1
epsilon_decay = 0.97

directory = "ep{}_sim{}_epo{}_dim{}_lr{}_bs{}_max{}".format(
    episodes,
    simulations,
    epochs,
    "".join([str(i) for i in dimensions]),
    learning_rate,
    batch_size,
    max_buffer_length,
)

save_directory = 'demo'
load_directory = save_directory

c = 1  # exploration constant
mcts_epsilon = 0
