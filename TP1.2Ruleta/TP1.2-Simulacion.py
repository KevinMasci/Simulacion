import numpy as np
import matplotlib.pyplot as plt

rojos=[1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36]
negros=[2,4,6,8,10,11,13,15,17,20,22,24,26,28,29,31,33,35]

n = 1000  #Número de tiradas de la ruleta
corrida = [np.random.randint(0,37) for i in range(n)]
cant_corridas = 37

caja_acotada = 10000
caja_infinita = 100000000
apuesta_inicial = 10

def frecuencia_relativa(corrida):
    cont_perdidos = 1
    cont_ganados = 0
    cuando_gane = {tiradas: 0 for tiradas in range(1, n + 1)}
    for x in corrida:
        if x in rojos:
            cont_perdidos +=1
        else: 
            cont_ganados +=1
            cuando_gane[cont_perdidos] += 1
            cont_perdidos = 1
    for i in cuando_gane:
        cuando_gane[i] /= cont_ganados
    cuando_gane = {k: v for k, v in cuando_gane.items() if v > 0.0}
    return cuando_gane

def ruleta(corrida, tipo_caja, estrategia):
    aciertos = []
    for x in corrida:
        if x in rojos:
            aciertos.append(0)
        else: 
            aciertos.append(1)

    #Flujo de caja
    apuesta = apuesta_inicial
    #Martingala
    if estrategia == "martingala":
        if tipo_caja == "infinita":
            flujo_caja = []
            caja_actual = caja_infinita
            for x in aciertos:
                if x == 1:
                    caja_actual += apuesta
                    flujo_caja.append(caja_actual)
                    apuesta = apuesta_inicial
                else:
                    caja_actual -= apuesta
                    flujo_caja.append(caja_actual)
                    apuesta = apuesta*2
        else:
            flujo_caja = []
            caja_actual = caja_acotada
            for x in aciertos:
                if x == 1:
                    caja_actual += apuesta
                    flujo_caja.append(caja_actual)
                    apuesta = apuesta_inicial
                else:
                    caja_actual -= apuesta
                    flujo_caja.append(caja_actual)
                    apuesta = apuesta*2
                if caja_actual < apuesta: break
    #Dalembert
    elif estrategia == "dalembert":
        if tipo_caja == "infinita":
            flujo_caja = []
            caja_actual = caja_infinita
            for x in aciertos:
                if x == 1:
                    caja_actual += apuesta
                    flujo_caja.append(caja_actual)
                    if apuesta > 1:
                        apuesta = apuesta - 1
                else:
                    caja_actual -= apuesta
                    flujo_caja.append(caja_actual)
                    apuesta = apuesta + 1
        else:
            flujo_caja = []
            caja_actual = caja_acotada
            for x in aciertos:
                if x == 1:
                    caja_actual += apuesta
                    flujo_caja.append(caja_actual)
                    if apuesta > 1:
                        apuesta = apuesta - 1
                else:
                    caja_actual -= apuesta
                    flujo_caja.append(caja_actual)
                    apuesta = apuesta + 1
                if caja_actual < apuesta: break
                
    #Paroli
    else:
        victorias_consecutivas = 0
        objetivo_victorias = 3
        if tipo_caja == "infinita":
            flujo_caja = []
            caja_actual = caja_infinita
            for x in aciertos:
                if x == 1:
                    victorias_consecutivas += 1
                    caja_actual += apuesta
                    flujo_caja.append(caja_actual)
                    if victorias_consecutivas == objetivo_victorias:
                        apuesta = apuesta_inicial
                        victorias_consecutivas = 0
                    else:
                        apuesta *= 2
                else:
                    caja_actual -= apuesta
                    flujo_caja.append(caja_actual)
                    apuesta = apuesta_inicial
                    victorias_consecutivas = 0
        else:
            flujo_caja = []
            caja_actual = caja_acotada
            for x in aciertos:
                if x == 1:
                    victorias_consecutivas += 1
                    caja_actual += apuesta
                    flujo_caja.append(caja_actual)
                    if victorias_consecutivas == objetivo_victorias:
                        apuesta = apuesta_inicial
                        victorias_consecutivas = 0
                    else:
                        apuesta *= 2
                else:
                    caja_actual -= apuesta
                    flujo_caja.append(caja_actual)
                    apuesta = apuesta_inicial
                    victorias_consecutivas = 0
                if caja_actual < apuesta:
                    break
    return [flujo_caja]

#GRAFICAS INDIVIDUALES

#frecuencia
cuando_gane = frecuencia_relativa(corrida)
plt.bar(cuando_gane.keys(), cuando_gane.values(), color ='red', width = 0.4)
plt.xlabel("Nro de veces para ganar luego de haber perdido")
plt.ylabel("Frecuencia relativa")
plt.ticklabel_format(useOffset=False, style='plain')
plt.savefig('TP1.2Ruleta/frecuencia.png')
plt.clf()

#martingala acotada
flujo_caja = ruleta(corrida, "acotada", "martingala")[0]
plt.plot(flujo_caja, label= "Flujo de caja. Martingala acotada")
plt.axhline(y = caja_acotada, color = 'red', linestyle = '-')
plt.ylabel("Cantidad en caja")
plt.xlabel("Numero de tirada")
plt.savefig('TP1.2Ruleta/flujo_mart_acotada.png')
plt.clf()

#martingala infinita
flujo_caja = ruleta(corrida, "infinita", "martingala")[0]
plt.plot(flujo_caja, label= "Flujo de caja. Martingala infinita")
plt.axhline(y = caja_infinita, color = 'red', linestyle = '-')
plt.ylabel("Cantidad en caja")
plt.xlabel("Numero de tirada")
plt.savefig('TP1.2Ruleta/flujo_mart_infinita.png')
plt.clf()

#dalembert acotada
flujo_caja = ruleta(corrida, "acotada", "dalembert")[0]
plt.plot(flujo_caja, label= "Flujo de caja. Dalembert acotada")
plt.axhline(y = caja_acotada, color = 'red', linestyle = '-')
plt.ylabel("Cantidad en caja")
plt.xlabel("Numero de tirada")
plt.savefig('TP1.2Ruleta/flujo_dalem_acotada.png')
plt.clf()

#dalembert infinita
flujo_caja = ruleta(corrida, "infinita", "dalembert")[0]
plt.plot(flujo_caja, label= "Flujo de caja. Dalembert infinita")
plt.axhline(y = caja_infinita, color = 'red', linestyle = '-')
plt.ylabel("Cantidad en caja")
plt.xlabel("Numero de tirada")
plt.savefig('TP1.2Ruleta/flujo_dalem_infinita.png')
plt.clf()

#Paroli acotada
flujo_caja = ruleta(corrida, "acotada", "paroli")[0]
plt.plot(flujo_caja, label= "Flujo de caja. Paroli acotada")
plt.axhline(y = caja_acotada, color = 'red', linestyle = '-')
plt.ylabel("Cantidad en caja")
plt.xlabel("Numero de tirada")
plt.savefig('TP1.2Ruleta/flujo_paroli_acotada.png')
plt.clf()

#Paroli infinita
flujo_caja = ruleta(corrida, "infinita", "paroli")[0]
plt.plot(flujo_caja, label= "Flujo de caja. Paroli infinita")
plt.axhline(y = caja_infinita, color = 'red', linestyle = '-')
plt.ylabel("Cantidad en caja")
plt.xlabel("Numero de tirada")
plt.savefig('TP1.2Ruleta/flujo_paroli_infinita.png')
plt.clf()

#GRAFICAS MULTIPLES

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()
fig5, ax5 = plt.subplots()
fig6, ax6 = plt.subplots()



for i in range(cant_corridas):
    corrida2 = [np.random.randint(0,37) for j in range(n)]
    flujo_martingala_acotada = ruleta(corrida2, "acotada", "martingala")[0]
    flujo_martingala_infinita = ruleta(corrida2, "infinita", "martingala")[0]
    flujo_dalembert_acotada = ruleta(corrida2, "acotada", "dalembert")[0]
    flujo_dalembert_infinita = ruleta(corrida2, "infinita", "dalembert")[0]
    flujo_paroli_acotada = ruleta(corrida2, "acotada", "paroli")[0]
    flujo_paroli_infinita = ruleta(corrida2, "infinita", "paroli")[0]


    ax1.plot(flujo_martingala_acotada)
    ax2.plot(flujo_martingala_infinita)
    ax3.plot(flujo_dalembert_acotada)
    ax4.plot(flujo_dalembert_infinita)
    ax5.plot(flujo_paroli_acotada)
    ax6.plot(flujo_paroli_infinita)


ax1.axhline(y = caja_acotada, color = 'red', linestyle = '-')
ax1.set_xlabel('n (número de tiradas)')
ax1.set_ylabel('Flujo de caja')

ax2.axhline(y = caja_infinita, color = 'red', linestyle = '-')
ax2.set_xlabel('n (número de tiradas)')
ax2.set_ylabel('Flujo de caja')

ax3.axhline(y = caja_acotada, color = 'red', linestyle = '-')
ax3.set_xlabel('n (número de tiradas)')
ax3.set_ylabel('Flujo de caja')

ax4.axhline(y = caja_infinita, color = 'red', linestyle = '-')
ax4.set_xlabel('n (número de tiradas)')
ax4.set_ylabel('Flujo de caja')

ax5.axhline(y = caja_acotada, color = 'red', linestyle = '-')
ax5.set_xlabel('n (número de tiradas)')
ax5.set_ylabel('Flujo de caja')

ax6.axhline(y = caja_infinita, color = 'red', linestyle = '-')
ax6.set_xlabel('n (número de tiradas)')
ax6.set_ylabel('Flujo de caja')

fig1.savefig('TP1.2Ruleta/Mflujo_mart_acotada.png')
fig2.savefig('TP1.2Ruleta/Mflujo_mart_infinita.png')
fig3.savefig('TP1.2Ruleta/Mflujo_delamb_acotada.png')
fig4.savefig('TP1.2Ruleta/Mflujo_delam_infinita.png')
fig5.savefig('TP1.2Ruleta/Mflujo_paroli_acotada.png')
fig6.savefig('TP1.2Ruleta/Mflujo_paroli_infinita.png')
