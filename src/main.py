import gymnasium as gym
from gymnasium import spaces
import numpy as np

MAX_DEMANDA = 30

class Tienda(gym.Env):
    def __init__(self, max_day):
        super(Tienda, self).__init__()
        # Estado inicial
        self.max_day = max_day
        self.money = 100
        self.stock = 0
        self.price = 10
        self.day = 0
        #Acciones posibles, comprar de 0, 10, 20, 30, 40, 50 unidades
        self.action_space = spaces.Discrete(6)
        #Definir observaciones, el agente ve el dinero, precio y stock disponible
        self.observation_space = spaces.Box(
            low=np.array([0,0,10,0], dtype=np.int32), #Para que no sean negativos
            high=np.array([
            MAX_DEMANDA*30*10 + 100,          # Dinero máximo estimado
            1500,           # Stock máximo estimado
            10,            # Precio máximo
            self.max_day   # Día máximo
        ], dtype=np.int32), #No hay limite superior
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
        demand = np.random.randint(0, MAX_DEMANDA)
        sold = min(self.stock, demand)
        revenue = sold * (self.price * 2)
        self.money += revenue      # Entra dinero
        self.stock -= sold          # Sale stock
        missed_sales = demand - sold  # Ventas perdidas

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
        profit = revenue - cost
        opportunity_penalty = missed_sales * (self.price * 0.5)
        storage_penalty = self.stock * (self.price * 0.1)
        if info.get('compra_fallida'):
            penalty = 100
        else:
            penalty = 0
        reward = profit - opportunity_penalty - storage_penalty - penalty

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

