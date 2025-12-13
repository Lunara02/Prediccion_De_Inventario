# Instrucciones de Ejecución del Proyecto

## 1. Abrir la carpeta del proyecto
Ubique y abra la carpeta **`src`** del proyecto.

## 2. Abrir un terminal
Dentro de la carpeta **`src`**, abra un terminal (consola).

## 3. Instalar dependencias
Ejecute el siguiente comando para instalar las dependencias necesarias:

```bash
pip install -r requirements.txt
```

## 4. Abrir el archivo de algoritmos
Abra el archivo **`sarsa.py`** o **`q-learning.py`** dependiendo cual quiera ejecutar.

## 5. Ejecutar el programa
Al final del archivo encontrará la función `run`, la cual se ejecuta desde el `main`.

La función `run` recibe los siguientes parámetros:

- **Cantidad de meses**: número de episodios a ejecutar.
- **Entrenamiento (`is_training`)**:  
  - `True`: el agente entrena, actualiza y guarda la Q-table.  
  - `False`: el agente carga la Q-table previamente entrenada y solo evalúa.
- **Render (`render`)**:  
  - `True`: muestra el entorno por consola (muy rápido y poco útil para análisis).  
  - `False`: ejecución normal sin renderizado.

Ejemplo de ejecución:

```python
run(1000, is_training=True, render=False)
```
