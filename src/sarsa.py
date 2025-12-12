import numpy as np
import matplotlib.pyplot as plt
from main import Tienda

MAX_DEMANDA = 30

def run(episodes, render=False):

    env = Tienda(30)

    # Divide position and velocity into segments
    money_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 2333)    # Between -1.2 and 0.6
    stock_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 151)   
    price_space = np.linspace(env.observation_space.low[2], env.observation_space.high[2], 7)    # Between -0.07 and 0.07
    day_space = np.linspace(env.observation_space.low[3], env.observation_space.high[3], 31)    # Between -0.07 and 0.07

    q = np.zeros((len(money_space), len(stock_space), len(price_space), len(day_space), env.action_space.n)) # init a 20x20x3 array

    learning_rate_a = 0.1 # alpha or learning rate
    discount_factor_g = 0.99 # gamma or discount factor.

    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.05
    rng = np.random.default_rng()

    max_money = 0
    max_stock = 0

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]      # Starting position
        state_m = np.digitize(state[0], money_space) - 1
        state_s = np.digitize(state[1], stock_space) - 1
        state_p = np.digitize(state[2], price_space) - 1
        state_d = np.digitize(state[3], day_space) - 1

        # Elegir accion inicial
        if rng.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q[state_m, state_s, state_p, state_d, :])

        terminated = False          # True when reached goal

        rewards=0

        while not terminated:
            # ejecutar la accion actual
            new_state,reward,terminated,_,_ = env.step(action)
            new_state_m = np.digitize(new_state[0], money_space) - 1
            new_state_s = np.digitize(new_state[1], stock_space) - 1
            new_state_p = np.digitize(new_state[2], price_space) - 1
            new_state_d = np.digitize(new_state[3], day_space) - 1

            if terminated:
                td_target = reward
                # actualizar q
                q[state_m, state_s, state_p, state_d, action] = q[state_m, state_s, state_p, state_d, action] + learning_rate_a * (
                    td_target + discount_factor_g * q[new_state_m, new_state_s, new_state_p, new_state_d, action] - q[state_m, state_s, state_p, state_d, action]
                )
                rewards += reward
                break

            # eliges la siguiente accion (a') segun la misma política ε-greedy
            if rng.random() < epsilon:
                new_action = env.action_space.sample()
            else:
                new_action = np.argmax(q[new_state_m, new_state_s, new_state_p, new_state_d, :])

            q[state_m, state_s, state_p, state_d, action] = q[state_m, state_s, state_p, state_d, action] + learning_rate_a * (
                reward + discount_factor_g * q[new_state_m, new_state_s, new_state_p, new_state_d, new_action] - q[state_m, state_s, state_p, state_d, action]
            )

            state = new_state
            state_m = new_state_m
            state_s = new_state_s
            state_p = new_state_p
            state_d = new_state_d

            # la accion siguiente pasa a ser la actual
            action = new_action

            rewards+=reward

            if render:
                env.render()

            if max_money < money_space[state_m]:
                max_money = money_space[state_m]
            if max_stock < stock_space[state_s]:
                max_stock = stock_space[state_s]

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        rewards_per_episode[i] = rewards

        if render:
            print(f'FIN ITERACION {i + 1}')



    env.close()

    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_per_episode[max(0, t-100):(t+1)])
    
    plt.plot(mean_rewards, label=f"Recompensa promedio por Meses")
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Línea en y=0
    plt.legend()
    plt.savefig('sarsa.png')


if __name__ == '__main__':
    # run(1000, is_training=True, render=False)

    run(20000, render=True)