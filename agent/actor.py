import random


class Actor:
    """
    Get current state and legal actions from player
    Decide action
    Update policy based on TD error
    """

    def __init__(self, cfg):
        self.policy = {}
        self.eligibility = {}

        self.alpha = cfg["actor"]["learning_rate"]
        self.eligibility_decay = cfg["actor"]["eligibility_decay"]
        self.discount_factor = cfg["actor"]["discount_factor"]
        self.epsilon = cfg["actor"]["init_epsilon"]
        self.epsilon_decay = cfg["actor"]["epsilon_decay"]

    def initialize_policy(self, state: str, possible_actions):
        """
        Initializes policy of state, if not already in the policy
        """
        for action in possible_actions:
            if self.policy.get((state, action)) is None:
                self.policy[(state, action)] = 0

    def reset_eligibilities(self):
        """
        Reset eligibilities to 0 for all s,a in self.eligibility
        """
        for key in self.eligibility:
            self.eligibility[key] = 0

    def set_current_eligibility(self, state: str, action: tuple):
        """
        Sets eligibility of the current state to 1
        """
        self.eligibility[(state, action)] = 1

    def update_eligibility(self, state, action):
        """
        Update eligibility with discount and decay
        """
        self.eligibility[(state, action)] = (
            self.discount_factor
            * self.eligibility_decay
            * self.eligibility[(state, action)]
        )

    def choose_action(self, state: str, actions: list, epsilon_greedy=True):
        """
        Epsilon Greepy Policy for choosing action
        Choose the best possible action with a probability 1-e,
        and a random action with probability e
        """

        greedy_number = random.uniform(0, 1)
        chosen_action = None

        if len(actions) > 0:
            chosen_action = actions[0]
            if epsilon_greedy:
                if greedy_number < self.epsilon:
                    random_index = random.randint(0, len(actions) - 1)
                    chosen_action = actions[random_index]
                    return chosen_action

            best = self.policy[(state, actions[0])]
            for action in actions:
                if self.policy[(state, action)] >= best:
                    best = self.policy[(state, action)]
                    chosen_action = action

        return chosen_action

    def update_policy(self, state: str, action: tuple, td_error: float):
        """
        Updates the policy for a given state and action based on the TD error
        computed by the Critic
        Also updates the epsilon for every iteration
        """
        curr_value = self.policy.get((state, action))
        self.policy[(state, action)] = curr_value + self.alpha * td_error * self.eligibility[(state, action)]
