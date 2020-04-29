# |---------- Environment parameters ----------|
board_size = 3 # between 3 and 10
player = 3 # starting player option, 1, 2 or 3 (mixed) Vet ikke om vi skal ha med 3
verbose = False # Whether the details of moves for each game is shown

node_size = 1500
initial_color = 'white'
edge_color = 'darkgrey'
p1_color = 'lightsteelblue'
p2_color = 'darkseagreen'
delay = 0.5 #seconds
display_last_game = True # True or False
plot_window_size = 20

# |------------- Agent parameters -------------|

c = 1 # exploration constant
episodes = 200 # number of episodes
simulations = 1000 # number of simulations (and hence rollouts) per actual game move
m = 5 # interval of ANETs to be cached for playing TOPP
g = 1 # number of games played in TOPP
mcts_epsilon = 0.05

learning_rate = 0.0005
epsilon = 1
epsilon_decay = 0.97
dimensions = [64, 64, 64] # Number of nodes in each layer goood = en del layers med f�rre eller 1 med flere 1000
activation_hidden = 'relu' # Choose between 'linear', 'sigmoid', 'tanh' and 'relu'
optimizer = 'adam' # Choose between 'adagrad', 'sgd', 'rmsprop' and 'adam'
epochs = 10
batch_size = 128
max_buffer_length = 500
save_directory = 'models'
load_directory = 'models'
