import numpy as np

def frecuencia_relativa(corrida, elegido):
    frec_respecto_n = []
    for i, x in enumerate(corrida):   ## for para recorrer la lista y usar el indice de cada item ("i")
        ## cuento cuantas veces aparece el nro elegido en la lista "corrida" hasta la posicion i (frec abs) 
        ## y lo divido por i+1 (poblacion)
        frec_respecto_n.append(corrida[:i+1].count(elegido)/(i+1))  
    return frec_respecto_n

def valor_promedio(corrida):
    valor_prom_respecto_n = []
    for i, x in enumerate(corrida):
        valor_prom_respecto_n.append(np.mean(corrida[:i+1]))
    return valor_prom_respecto_n

def valor_varianza(corrida):
    valor_varianza_respecto_n = []
    for i, x in enumerate(corrida):
        valor_varianza_respecto_n.append(np.var(corrida[:i+1]))
    return valor_varianza_respecto_n

def valor_desvio(corrida):
    valor_desvio_respecto_n = []
    for i, x in enumerate(corrida):
        valor_desvio_respecto_n.append(np.std(corrida[:i+1]))
    return valor_desvio_respecto_n

def contar(corrida):
    cantidades = []
    for x in range(38):
        cantidades.append(corrida.count(x))
    return cantidades