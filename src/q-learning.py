import numpy as np
import matplotlib.pyplot as plt
import pickle
from main import Tienda

MAX_DEMANDA = 30

def run(episodes, is_training=True, render=False):

    env = Tienda(30)

    # Divide position and velocity into segments
    money_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 4401)    # Between -1.2 and 0.6
    stock_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 151)   
    price_space = np.linspace(env.observation_space.low[2], env.observation_space.high[2], 1)    # Between -0.07 and 0.07
    day_space = np.linspace(env.observation_space.low[3], env.observation_space.high[3], 31)    # Between -0.07 and 0.07

    if(is_training):
        q = np.zeros((len(money_space), len(stock_space), len(price_space), len(day_space), env.action_space.n)) # init a 20x20x3 array
    else:
        f = open('mountain_car.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.1 # alpha or learning rate
    discount_factor_g = 0.99 # gamma or discount factor.

    epsilon = 1         # 1 = 100% random actions
    epsilon_decay_rate = 2/episodes # epsilon decay rate
    rng = np.random.default_rng()   # random number generator

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]      # Starting position, starting velocity always 0
        state_m = np.digitize(state[0], money_space) - 1
        state_s = np.digitize(state[1], stock_space) - 1
        state_p = 0
        state_d = np.digitize(state[3], day_space) - 1

        terminated = False          # True when reached goal

        rewards=0

        while not terminated:
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state_m, state_s, state_p, state_d, :])

            new_state,reward,terminated,_,_ = env.step(action)
            new_state_m = np.digitize(new_state[0], money_space) - 1
            new_state_s = np.digitize(new_state[1], stock_space) - 1
            new_state_p = 0
            new_state_d = np.digitize(new_state[3], day_space) - 1

            if is_training:
                q[state_m, state_s, state_p, state_d, action] = q[state_m, state_s, state_p, state_d, action] + learning_rate_a * (
                    reward + discount_factor_g*np.max(q[new_state_m, new_state_s, new_state_p, new_state_d, :]) - q[state_m, state_s, state_p, state_d, action]
                )

            state = new_state
            state_m = new_state_m
            state_s = new_state_s
            state_p = new_state_p
            state_d = new_state_d

            rewards+=reward

            if render:
                env.render()

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        rewards_per_episode[i] = rewards

        print(f'FIN ITERACION {i}')



    env.close()

    # Save Q table to file
    if is_training:
        f = open('mountain_car.pkl','wb')
        pickle.dump(q, f)
        f.close()

    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(mean_rewards)
    plt.savefig(f'mountain_car.png')

if __name__ == '__main__':
    # run(1000, is_training=True, render=False)

    run(5000, is_training=True, render=True)