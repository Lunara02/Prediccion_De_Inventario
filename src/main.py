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
        self.inventario = []
        self.price = 10
        self.day = 0
        #Acciones posibles, comprar de 0, 10, 20, 30, 40, 50 unidades
        self.action_space = spaces.Discrete(6)
        #Definir observaciones, el agente ve el dinero, precio y stock disponible
        self.observation_space = spaces.Box(
            low=np.array([0,0,7,0], dtype=np.int32), #Para que no sean negativos
            high=np.array([
            6996,          # Dinero máximo estimado
            552,           # Stock máximo estimado
            13,            # Precio máximo
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

        # Eliminar los lotes vencidos
        Cantidad_lotes = len(self.inventario)
        lotes_vencidos = [l for l in self.inventario if l["dias"] <= 0]
        self.inventario = [l for l in self.inventario if l["dias"] > 0]



        # COMPRAR

        # Verificamos si tiene dinero suficiente (para no tener saldo negativo)
        if self.money >= cost:
            self.money -= cost
            self.stock += buy_unit
            nuevo_lote = {
                "cantidad" : buy_unit,
                "dias" : 5, # dias para que caduque
                "precio" : cost
            }
            self.inventario.append(nuevo_lote)
        else:
            penalty = 10
            info['compra_fallida'] = True

        # VENDER

        demand = np.random.randint(0, MAX_DEMANDA)
        sold = min(self.stock, demand)
        revenue = sold * (self.price * 2)
        self.money += revenue      # Entra dinero
        self.stock -= sold          # Sale stock
        missed_sales = demand - sold  # Ventas perdidas 

        cantidad_a_vender = sold
        for lote in self.inventario:
            if cantidad_a_vender <= 0:
                break
            disponible = lote["cantidad"]
            usado = min(disponible, cantidad_a_vender)
            lote["cantidad"] -= usado
            cantidad_a_vender -= usado

        # Eliminar los lotes que se vendieron completamente
        self.inventario = [lote for lote in self.inventario if lote["cantidad"] > 0]
        #2. OBSERVACIÓN
        if self.day < self.max_day:
            self.day += 1
            for lote in self.inventario:
                lote["dias"] = lote["dias"] - 1
        else:
            terminated = True # Dia final

        # Variamos el precio del producto, el agente podra aprender que le conviene comprar barato
        self.price += np.random.randint(-1,2)
        self.price = max(7, min(self.price, 13)) # Limitar precio entre 7 y 13

        observation = np.array([
            self.money, 
            self.stock, 
            self.price, 
            self.day
        ], dtype=np.int32)

        #3. RECOMPENSA
        profit = revenue - cost
        opportunity_penalty = missed_sales * (self.price * 0.5)
        if len(lotes_vencidos) > 0:
            lote_loss_penalty = lotes_vencidos[0]["precio"]
        else:
            lote_loss_penalty = 0
        storage_penalty = self.stock * (self.price * 0.1)
        if info.get('compra_fallida'):
            penalty = 100
        else:
            penalty = 0
        reward = (profit - opportunity_penalty - storage_penalty - penalty - lote_loss_penalty) / 1000

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