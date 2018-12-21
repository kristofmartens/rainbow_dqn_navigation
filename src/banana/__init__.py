from gym.envs.registration import register
from .banana import Banana

register(id='Banana-v1', entry_point='banana:Banana')
