# src/modeling/__init__.py

# Importa o Solver (HybridNeuralODE) de neural_ode.py
from .neural_ode import HybridNeuralODE

# Importa a Lógica da Derivada (ODEFunc) de ode_func.py
from .ode_func import ODEFunc

# Importa o Treinador
from .trainer import train

# Importa o VAE (se você estiver usando o backbone.py para o caso A)
from .backbone import VAE