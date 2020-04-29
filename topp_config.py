# |---------- Environment parameters ----------|

board_size = 4 # between 3 and 10
verbose = False # Whether the details of moves for each game is show

node_size = 1500
initial_color = 'white'
edge_color = 'darkgrey'
p1_color = 'lightsteelblue'
p2_color = 'darkseagreen'
delay = 0.5 #seconds
display_last_game = True # True or False
plot_window_size = 20

# |------------- Agent parameters -------------|

episodes = 200
m = 5 # interval of ANETs to be cached for playing TOPP
g = 25 # number of games played in TOPP

learning_rate = 0.0005
dimensions = [64, 64, 64] # Number of nodes in each layer goood = en del layers med færre eller 1 med flere 1000
activation_hidden = 'relu' # Choose between 'linear', 'sigmoid', 'tanh' and 'relu'
optimizer = 'adam' # Choose between 'adagrad', 'sgd', 'rmsprop' and 'adam'
save_directory = 'models'
load_directory = 'models'
