import random
import numpy as np
import matplotlib.pyplot as plt
from funciones import frecuencia_relativa, valor_promedio, valor_varianza, valor_desvio, contar

#Constantes
n = 37
elegido = random.randint(0, 37)
tiradas = 2000
l_tiradas = list(range(1, tiradas+1))
corrida = [random.randint(0,37) for i in range(tiradas)]
frec_esperada = 1/n
prom_esperado = np.mean(corrida)
varianza_esperada = np.var(corrida)
desvio_esperado = np.std(corrida)

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

des = valor_desvio(corrida)
plt.plot(l_tiradas, des, label = "Valor desvio respecto a n")
plt.axhline(y = desvio_esperado, color = 'r', linestyle = '-', label = "Valor desvio esperado")
plt.xlabel("Tiradas")
plt.ylabel("Valor desvio")
plt.legend()
plt.show()

#plt.hist(corrida, bins = "auto")
cantidad = contar(corrida)
numeros = list(range(38))
plt.bar(numeros, cantidad)
plt.show()