import random

#Constantes
elegido = 4
corridas = 37
frec_esperada = 1/37

tiradas = [random.randint(0,37) for i in range(corridas)]
frec_relativa = tiradas.count(elegido)/corridas

print(tiradas)
print(frec_relativa)