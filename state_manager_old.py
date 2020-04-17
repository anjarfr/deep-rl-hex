import matplotlib.pyplot as plt
import numpy as np
import yaml

from agent.actor import Actor
from agent.torch_critic import NeuralNetCritic
from environment.hex import Hex
from environment.visualizer import Visualizer

with open("config.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)


class StateManager:
    """
    Player is responsible for communication between environment and agent
    """

    def __init__(self):
        self.verbose = cfg['game']['verbose']
        self.episodes = cfg["agent"]["episodes"]

        self.game = Hex(cfg, self.verbose)
        self.actor = Actor(cfg)

        self.visualizer = Visualizer(self.game.board, self.game.size, cfg["display"])
        self.display_last_game = cfg["display"]["display_last_game"]
        self.iterations = []
        self.epsilon_greedy = True

    def initialize_critic(self):
        critic = NeuralNetCritic(cfg, self.game.board.generate_state())
        return critic

    def plot_result(self):
        """ Plots the remaining pegs after each episode and the moving average """
        window_size = min(cfg["display"]["plot_window_size"], int(self.episodes / 2))

        plt.figure()
        plt.plot(self.iterations, self.remaining_pegs, color='#638ed4')
        plt.plot(self.iterations[int(window_size / 2):-int((window_size - 1) / 2)],
                 np.convolve(self.remaining_pegs, np.ones((window_size,)) / window_size, mode='valid'), color='#003182')
        plt.xlabel('Episode')
        plt.ylabel('Remaining pegs')
        plt.show()

    def play_final_game(self, SAP_history):
        """ Visualize the policy learnt by the agent """
        self.game = self.initialize_game()
        self.visualizer.fill_nodes(self.game.board.get_filled_cells())

        for sap in SAP_history:
            action = sap[1]
            self.game.perform_action(action)
            self.visualizer.fill_nodes(
                self.game.board.get_filled_cells(), action[0], action[1]
            )

    def reset_eligibilities_and_history(self):
        """ Reset all eligibilities to 0 at the start of a new game,
        and reset the path taken from start to end state """

        self.critic.reset_eligibility()
        self.actor.reset_eligibilities()
        self.SAP_history = []

    def initialize_new_game(self):
        """ Create a new game for each episode and return the initial state and initial action """
        self.game = self.initialize_game()
        init_state = self.game.board.generate_state()  # String with state
        possible_actions = (self.game.get_all_legal_actions())  # List of actions [(from), (to)]
        self.actor.initialize_policy(init_state, possible_actions)  # Creates dictionary {string: tuple of tuple}
        init_action = self.actor.choose_action(init_state, possible_actions,
                                               epsilon_greedy=self.epsilon_greedy)  # Tuple of tuple
        if self.table_critic:
            self.critic.initialize_value_function(init_state)  # Creates dictionary {string: value}

        return init_state, init_action

    def get_reward_and_succ_state(self, state, action):
        """
        Perform action from state s and bring the game to state s'
        Initializes the critic's value function if critic is a table
        Returns the reward received and successor state
        """

        """ Initializes value of new states found to 0 at the start of an episode """
        if self.table_critic:
            self.critic.initialize_value_function(state)

        """ Do action a from state s to s', and receive reward r """
        reward = self.game.perform_action(action)  # Int
        succ_state = self.game.board.generate_state()  # String

        """ Add performed action to SAP history"""
        self.SAP_history.append((state, action))

        """ Dynamically update value function if succ_state is never seen before """
        if self.table_critic:
            self.critic.initialize_value_function(succ_state)

        return reward, succ_state

    def get_succ_action(self, succ_state):
        """ Dictate a' from the current policy for s' """

        possible_succ_actions = self.game.get_all_legal_actions()

        succ_action = None
        if len(possible_succ_actions) > 0:
            """
            If succ_state has any legal actions:
            Dynamically update value function of succ_state and its possible actions
            """
            self.actor.initialize_policy(succ_state, possible_succ_actions)
            succ_action = self.actor.choose_action(succ_state, possible_succ_actions,
                                                   epsilon_greedy=self.epsilon_greedy)  # tuple of tuple
        return succ_action

    def set_eligibilities_for_current_state(self, state, action):
        """ Set eligibility of a and s to 1 (get all the reward or punishment)"""
        self.actor.set_current_eligibility(state, action)

        """ Set eligibility of current state to 1 """
        if self.table_critic:
            self.critic.set_current_eligibility(state)

    def update_eligibility_traces(self, reward, TD_error, succ_state):
        """ For each state-action-pairs, update their eligibility """
        for SAP in self.SAP_history:
            state = SAP[0]
            action = SAP[1]

            self.critic.update_value_function(state, TD_error, reward, succ_state)
            if self.table_critic:
                self.critic.update_eligibility(state)

            self.actor.update_policy(state, action, TD_error)
            self.actor.update_eligibility(state, action)

    def play_game(self):
        for i in range(self.episodes):

            if i == self.episodes - 1: #If final episode
                print('Switched to following policy')
                self.epsilon_greedy = False

            """ Reset all elegibilities to 0 """
            self.reset_eligibilities_and_history()

            """Initialize new game"""
            init_state, init_action = self.initialize_new_game()

            state = init_state  # String
            action = init_action  # Tuple of tuple

            """ Play game until termination, a.k.a. one episode """
            while not self.game.is_finished():

                reward, succ_state = self.get_reward_and_succ_state(state, action)
                succ_action = self.get_succ_action(succ_state)
                self.set_eligibilities_for_current_state(state, action)

                """ Compute TD error """
                TD_error = self.critic.calculate_TD_error(state, succ_state, reward)

                """ Update the eligibility traces, value function and policy for the path taken up to this point """
                self.update_eligibility_traces(reward, TD_error, succ_state)

                state = succ_state
                if succ_action:
                    action = succ_action

            """ Decay epsilon for each episode """
            self.actor.epsilon = self.actor.epsilon * self.actor.epsilon_decay

            pegs = self.game.get_pegs()
            print(i, ": ", pegs, ' pegs were left. Epsilon: ', self.actor.epsilon)

            """ Store result for learning plot """
            self.remaining_pegs.append(pegs)
            self.iterations.append(i)

        if self.display_last_game:
            self.play_final_game(self.SAP_history)

        self.plot_result()


def main():
    player = StateManager()
    #player.play_game()


if __name__ == "__main__":
    main()
