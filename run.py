from state_manager import StateManager
import config


def run_multiple():
    config.lr = 0.0001  # [0.0001, 0.0005, 0.001, 0.005]
    epochs = [10, 50]
    for e in epochs:
        config.epochs = e
        player = StateManager()
        player.play_game()


if __name__ == "__main__":
    run_multiple()
