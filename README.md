# On-Policy Monte Carlo Tree Search for Game Playing

## TODO
- [ ] Hex logikk 
- [ ] ANET 
- [ ] RBUF 
- [ ] Finne ut hvordan illegal moves settes til 0 
- [ ] Implementere e-greedy 
- [ ] Litt randomness på target policy også (sigma)  
- [x] Finne ut hvilke variabler vi trenger i config 
- [ ] Implementere at vi både får evaluation og prob dist som output fra ANET som vi kan bruke som critic 
- [ ] Update visualization

Spørre om: 
- [ ] Kan vi ha et critic og et actor nett? Hvis vi skal ha ett, hvordan trener vi det opp for å gjøre begge deler? 

This project consist of four main parts: the game, the state manager, the actor and the MCTS system.

### State manager
- Store replay buffer as hashmap or array. Consisting of (state, distribution) couplets from each move in the game. 
Use this to select mini-batches for the NN to train on after each completed episode/game. 

### Actor
- The actor consists of a Neural Network, that is trained by the MCTS to output successful moves.
- The error between the MCTS targets and the NN output is used as basis for weight update/backprop. 
- In addition to state, the NN takes in the player ID. The player ID is represented by two neurons
- The NN is trained after each complete actual game 

**Comment: Can the actor and MCTS communicate directly or must it be through state manager?**

### MCTS
- The most important role for the MCTS is to provide probability distributions as target for the NN. 
- Higher visit count -> higher probability. Must sum to 1. 
- After an actual move is made, we keep the subtree below the chosen node, and throw away the rest of the tree. 

#### Policies
- Three policy: Like last time (Q + U)
- Default policy in tree: Use actor epsilon greedy of random move, else best move.  
- Training/default policy: Use actor with randomness/epsilon (more in the beginning)
- Target policy: Use actor but with a smaller chance of randomness. 