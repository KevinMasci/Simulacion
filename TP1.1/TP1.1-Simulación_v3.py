import numpy as np
import matplotlib.pyplot as plt

#Constantes
n = 2000    #Número de tiradas de la ruleta
elegido = np.random.randint(0, 37) #Numero random a evaluar
corrida = [np.random.randint(0,37) for i in range(n)] 

ar=np.arange(37) #Arreglo con números del 0 al 36
frec_esperada = 1/37
prom_esperado = np.mean(ar)
varianza_esperada = np.var(ar)
desvio_esperado = np.std(ar)

#Funciones
def frecuencia_relativa(corrida, elegido): #Función para calcular la frecuencia relativa 
    frec_respecto_n = []
    for i, x in enumerate(corrida):                                 # for para recorrer la lista y usar el indice de cada item ("i") cuento cuantas veces aparece el nro elegido en 
        frec_respecto_n.append(corrida[:i+1].count(elegido)/(i+1))  # la lista "corrida" hasta la posicion i (frec abs) y lo divido por i+1 (muestra)
    return frec_respecto_n
 
def valor_promedio(corrida): #Función para calcular el promedio
    valor_prom_respecto_n = []
    for i, x in enumerate(corrida):
        valor_prom_respecto_n.append(np.mean(corrida[:i+1]))
    return valor_prom_respecto_n

def valor_varianza(corrida): #Función para calcular la varianza
    valor_varianza_respecto_n = []
    for i, x in enumerate(corrida):
        valor_varianza_respecto_n.append(np.var(corrida[:i+1]))
    return valor_varianza_respecto_n

def valor_desvio(corrida): #Función para calcular el desvío
    valor_desvio_respecto_n = []
    for i, x in enumerate(corrida):
        valor_desvio_respecto_n.append(np.std(corrida[:i+1]))
    return valor_desvio_respecto_n

def contar(corrida): #Función para calcular la frecuencia absoluta de cada número
    cantidades = []
    for x in range(37):
        cantidades.append(corrida.count(x))
    return cantidades


#Gráficos

plt.hist(corrida, bins=36, edgecolor= "black", linewidth=1)
plt.xlabel("valores")
plt.ylabel("frecuencia")
plt.savefig("Histograma.png")
plt.clf()

fr = frecuencia_relativa(corrida, elegido)
plt.plot(fr, color='red', label = 'frn (frecuencia relativa del número X con respecto a n)')
plt.axhline(y = frec_esperada, color = 'blue', linestyle = '-', label = 'fre (frecuencia relativa esperada de X)')
plt.xlabel('n (número de tiradas)')
plt.ylabel('fr (frecuencia relativa)')
plt.legend()
plt.savefig('Plot1.png')
plt.clf()

vp = valor_promedio(corrida)
plt.plot(vp, color='red', label = 'vpn (valor promedio de las tiradas con respecto a n)')
plt.axhline(y = prom_esperado, color = 'blue', linestyle = '-', label = 'vpe (valor promedio esperado)')
plt.xlabel('n (número de tiradas)')
plt.ylabel('vp (valor promedio de las tiradas)')
plt.legend()
plt.savefig('Plot2.png')
plt.clf()

vari = valor_varianza(corrida)
plt.plot(vari, color='red', label = 'vvn (valor de la varianza del número X con respecto a n)')
plt.axhline(y = varianza_esperada, color = 'blue', linestyle = '-', label = 'vve (valor de la varianza esperada)')
plt.xlabel('n (número de tiradas)')
plt.ylabel('vv (valor de la varianza)')
plt.legend()
plt.savefig('Plot3.png')
plt.clf()

des = valor_desvio(corrida)
plt.plot(des, color='red', label = 'vd (valor del desvío del número X con respecto a n)')
plt.axhline(y = desvio_esperado, color = 'blue', linestyle = '-', label = 'vde (valor del desvío esperado)')
plt.xlabel('n (número de tiradas)')
plt.ylabel('vd (valor del desvío)')
plt.legend()
plt.savefig('Plot4.png')
plt.clf()

#SEGUNDA PARTE

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()

#Listas para calcular los promedios de promedios (se van a ir sumando los valores de todas las corridas)
promedios_de_fr = [0]*n
promedios_de_prom = [0]*n
promedios_de_var = [0]*n
promedios_de_desv = [0]*n

for i in range(50):
    corrida2 = [np.random.randint(0,37) for j in range(n)]
    fr = frecuencia_relativa(corrida2, elegido)
    prom = valor_promedio(corrida2)
    var = valor_varianza(corrida2)
    desv = valor_desvio(corrida2)
    
    for x in range(n): #sumo los valores de la nueva corrida 
        promedios_de_fr[x] += fr[x]
        promedios_de_prom[x] += prom[x]
        promedios_de_var[x] += var[x]
        promedios_de_desv[x] += desv[x]

    ax1.plot(fr)
    ax2.plot(prom)
    ax3.plot(var)
    ax4.plot(desv)

#saco el promedio
promedios_de_fr = [x/50 for x in promedios_de_fr]
promedios_de_prom = [x/50 for x in promedios_de_prom]
promedios_de_var = [x/50 for x in promedios_de_var]
promedios_de_desv = [x/50 for x in promedios_de_desv]


ax1.axhline(y = frec_esperada, color = 'blue', linestyle = '-', label = 'fre (valor de la frecuencia relativa esperada)')
ax1.set_xlabel('n (número de tiradas)')
ax1.set_ylabel('fr (frecuencia relativa)')
ax1.legend()

ax2.axhline(y = prom_esperado, color = 'blue', linestyle = '-', label = 'vpe (valor del promedio esperado)')
ax2.set_xlabel('n (número de tiradas)')
ax2.set_ylabel('vp (valor promedio de las tiradas)')
ax2.legend()

ax3.axhline(y = varianza_esperada, color = 'blue', linestyle = '-', label = 'vve (valor de la varianza esperada)')
ax3.set_xlabel('n (número de tiradas)')
ax3.set_ylabel('vv (valor de la varianza)')
ax3.legend()

ax4.axhline(y = desvio_esperado, color = 'blue', linestyle = '-', label = 'dve (valor del desvío esperado)')
ax4.set_xlabel('n (número de tiradas)')
ax4.set_ylabel('vd (valor del desvío)')
ax4.legend()

fig1.savefig('Plot6.png')
fig2.savefig('Plot7.png')
fig3.savefig('Plot8.png')
fig4.savefig('Plot9.png')
plt.clf()

#Graficos de promedios
plt.plot(promedios_de_fr, color='red', label = 'promedios de frecuencias relativas')
plt.axhline(y = frec_esperada, color = 'blue', linestyle = '-', label = 'fre (frecuencia relativa esperada de X)')
plt.xlabel('n (número de tiradas)')
plt.ylabel('fr (frecuencia relativa)')
plt.legend()
plt.savefig('Plot10.png')
plt.clf()

plt.plot(promedios_de_prom, color='red', label = 'promedios de promedios')
plt.axhline(y = prom_esperado, color = 'blue', linestyle = '-', label = 'vpe (valor promedio esperado)')
plt.xlabel('n (número de tiradas)')
plt.ylabel('vp (valor promedio de las tiradas)')
plt.legend()
plt.savefig('Plot11.png')
plt.clf()

plt.plot(promedios_de_var, color='red', label = 'promedios de varianzas')
plt.axhline(y = varianza_esperada, color = 'blue', linestyle = '-', label = 'vve (valor de la varianza esperada)')
plt.xlabel('n (número de tiradas)')
plt.ylabel('vv (valor de la varianza)')
plt.legend()
plt.savefig('Plot12.png')
plt.clf()

plt.plot(promedios_de_desv, color='red', label = 'promedios de desvios')
plt.axhline(y = desvio_esperado, color = 'blue', linestyle = '-', label = 'vde (valor del desvío esperado)')
plt.xlabel('n (número de tiradas)')
plt.ylabel('vd (valor del desvío)')
plt.legend()
plt.savefig('Plot13.png')
plt.clf()