import numpy as np, time
from env.wrapper import CarRacingWrapper

env = CarRacingWrapper()
obs = env.reset()
assert obs.shape == (19,)
ret, done, steps = 0, False, 0
while not done and steps < 300:
    a = np.random.randint(env.n_actions)
    obs, r, done, _ = env.step(a)
    ret += r
    steps += 1
print(f"random reward {ret:.1f}  steps {steps}")
env.close()