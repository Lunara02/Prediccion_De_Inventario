import gymnasium as gym
from gymnasium import spaces
import numpy as np

class Tienda(gym.Env):
    def __init__(self, max_day):
        super(Tienda, self).__init__()
        # Estado inicial
        self.max_day = max_day
        self.money = 100
        self.price = 10
        self.stock = 0
        self.day = 0
        #Acciones posibles, comprar de 0, 10, 20, 30, 40, 50 unidades
        self.action_space = spaces.Discrete(6)
        #Definir observaciones, el agente ve el dinero, precio y stock disponible
        self.observation_space = spaces.Box(
            low=0, #Para que no sean negativos
            high=np.iinfo(np.int32).max, #No hay limite superior
            shape=(4,), #Cantidad de variables a observar
            dtype=np.int32
        )


    def step(self, action):
        
        #Inicializar variables de control por defecto
        terminated = False
        truncated = False
        info = {}
        penalty = 0

        previus_money = self.money + (self.stock * self.price)

        # 1. LÓGICA DE LA ACCIÓN
        buy_unit = action * 10
        cost = buy_unit * self.price

        # Verificamos si tiene dinero suficiente (para no tener saldo negativo)
        if self.money >= cost:
            self.money -= cost
            self.stock += buy_unit
        else:
            penalty = 10
            info['compra_fallida'] = True
        demand = np.random.randint(0, 30)
        sold = min(self.stock, demand)
        earnings = sold * (self.price * 2)
        self.money += earnings      # Entra dinero
        self.stock -= sold          # Sale stock

        #2. OBSERVACIÓN
        if self.day < self.max_day:
            self.day += 1
        else:
            terminated = True # Dia final

        observation = np.array([
            self.money, 
            self.stock, 
            self.price, 
            self.day
        ], dtype=np.int32)

        #3. RECOMPENSA
        actual_money = self.money + (self.stock * self.price)
        reward = (actual_money - previus_money) - penalty

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None):
            super().reset(seed=seed)

            #Restaurar variables al estado inicial
            self.money = 100
            self.stock = 0
            self.day = 0
            self.price = 10

            #Construir la observación inicial
            observation = np.array([
                self.money, 
                self.stock, 
                self.price, 
                self.day
            ], dtype=np.int32)

            info = {}

            return observation, info


    def render(self):
        print(f"--- Día {self.day} ---")
        print(f"Dinero: {self.money:.2f} | Stock: {self.stock}")


env = Tienda(max_day=30)
obs, info = env.reset()
print("Estado Inicial:", obs)

for _ in range(100):
    # Tomar una acción aleatoria (0 a 5)
    action = env.action_space.sample()
    # Dar un paso
    obs, reward, terminated, truncated, info = env.step(action)
    # Usar tu nueva función render
    env.render()
    print(f"Acción: Comprar {action*10}u | Recompensa: {reward}")
    
    if terminated:
        break