from unityagents import UnityEnvironment
import os


class Banana:
    @staticmethod
    def get_banana_env():
        directory, _ = os.path.split(__file__)
        return UnityEnvironment(os.path.join(directory, 'data', 'Banana.app'))
