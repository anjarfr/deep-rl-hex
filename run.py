from state_manager import StateManager
import config


def run_multiple():
    config.episodes = 1000  # 500
    lr = 0.0001  # [0.0001, 0.0005, 0.001, 0.005]
    epochs = [1, 10, 50]
    batch_size = [64, 128]
    for l in lr:
        config.lr = l
        for e in epochs:
            config.epochs = e
            for b in batch_size:
                config.batch_size = b
                player = StateManager()
                player.play_game()


if __name__ == "__main__":
    run_multiple()
