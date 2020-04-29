from state_manager import StateManager
import config


def run_multiple():
    config.episodes = 500  # 500
    lr = 0.0001  # [0.0001, 0.0005, 0.001, 0.005]
    epochs = [10, 50]
    for l in lr:
        config.lr = l
        for e in epochs:
            config.epochs = e
            player = StateManager()
            player.play_game()


if __name__ == "__main__":
    run_multiple()
