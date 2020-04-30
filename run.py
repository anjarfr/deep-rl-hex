from state_manager import StateManager
import config


def run_multiple():
    epochs = [10, 50]
    for e in epochs:
        config.epochs = e
        print(config.epochs)
        player = StateManager()
        player.play_game()


if __name__ == "__main__":
    run_multiple()
