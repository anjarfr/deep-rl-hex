from state_manager import StateManager
import config


def run_multiple():
    episodes = [2, 5, 7]
    for e in episodes:
        config.episodes = e
        player = StateManager()
        player.play_game()


if __name__ == "__main__":
    run_multiple()
