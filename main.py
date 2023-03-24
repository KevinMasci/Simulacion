import random
import numpy as np
import matplotlib.pyplot as plt
from funciones import frecuencia_relativa, valor_promedio, valor_varianza

#Constantes
n = 37
elegido = 4
tiradas = 2500
l_tiradas = list(range(1, tiradas+1))
corrida = [random.randint(0,37) for i in range(tiradas)]
frec_esperada = 1/n
prom_esperado = np.mean(corrida)
varianza_esperada = np.var(corrida)

fr = frecuencia_relativa(corrida, elegido)
plt.plot(l_tiradas, fr, label = "Frecuencia relativa respecto a n")
plt.axhline(y = frec_esperada, color = 'r', linestyle = '-', label = "Frecuencia esperada")
plt.xlabel("Tiradas")
plt.ylabel("Frecuencia relativa")
plt.legend()
plt.show()

vp = valor_promedio(corrida)
plt.plot(l_tiradas, vp, label = "Valor promedio respecto a n")
plt.axhline(y = prom_esperado, color = 'r', linestyle = '-', label = "Valor promedio esperado")
plt.xlabel("Tiradas")
plt.ylabel("Valor promedio")
plt.legend()
plt.show()

vari = valor_varianza(corrida)
plt.plot(l_tiradas, vari, label = "Valor varianza respecto a n")
plt.axhline(y = varianza_esperada, color = 'r', linestyle = '-', label = "Valor varianza esperado")
plt.xlabel("Tiradas")
plt.ylabel("Valor varianza")
plt.legend()
plt.show()