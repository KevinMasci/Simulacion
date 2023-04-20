import numpy as np
import matplotlib.pyplot as plt


rojos=[1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36]
negros=[2,4,6,8,10,11,13,15,17,20,22,24,26,28,29,31,33,35]

n = 1000  #Número de tiradas de la ruleta
corrida = [np.random.randint(0,37) for i in range(n)]
cant_corridas = 10

caja_acotada = 50
caja_infinita = 10000
apuesta_inicial = 1

#Martingala

def ruleta(corrida, tipo_caja, estrategia):
    
    #Frecuencias relativas
    frec_relativa_aciertos = []
    aciertos = [] #1 si acierta, 0 si no acierta
    for x in corrida:
        if x in negros:
            aciertos.append(1)
        else: aciertos.append(0)
    for i, x in enumerate(aciertos):
        frec_relativa_aciertos.append(sum(aciertos[: i+1])/(i+1))
    
    #Flujo de caja
    apuesta = apuesta_inicial
    #Martingala
    if estrategia == "martingala":
        if tipo_caja == "infinita":
            flujo_caja = [caja_infinita]
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
            flujo_caja = [caja_infinita]
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
            flujo_caja = [caja_infinita]
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

    return [frec_relativa_aciertos, flujo_caja]

#GRAFICAS INDIVIDUALES

#martingala acotada
frec_rel, flujo_caja = ruleta(corrida, "acotada", "martingala")
plt.stem(frec_rel, markerfmt=" ", label="Frecuencia relativa respecto a n.")
plt.ylabel("Frecuencia relativa")
plt.savefig('TP1.2Ruleta/frec_rel.png')
plt.clf()

plt.plot(flujo_caja, label= "Flujo de caja. Martingala acotada")
plt.axhline(y = caja_acotada, color = 'blue', linestyle = '-')
plt.ylabel("Cantidad en caja")
plt.xlabel("Numero de tirada")
plt.savefig('TP1.2Ruleta/flujo_mart_acotada.png')
plt.clf()

#martingala infinita
frec_rel, flujo_caja = ruleta(corrida, "infinita", "martingala")
plt.plot(flujo_caja, label= "Flujo de caja. Martingala infinita")
plt.axhline(y = caja_infinita, color = 'blue', linestyle = '-')
plt.ylabel("Cantidad en caja")
plt.xlabel("Numero de tirada")
plt.savefig('TP1.2Ruleta/flujo_mart_infinita.png')
plt.clf()

#dalembert acotada
frec_rel, flujo_caja = ruleta(corrida, "acotada", "dalembert")
plt.plot(flujo_caja, label= "Flujo de caja. Dalembert acotada")
plt.axhline(y = caja_acotada, color = 'blue', linestyle = '-')
plt.ylabel("Cantidad en caja")
plt.xlabel("Numero de tirada")
plt.savefig('TP1.2Ruleta/flujo_dalem_acotada.png')
plt.clf()

#dalembert infinita
frec_rel, flujo_caja = ruleta(corrida, "infinita", "dalembert")
plt.plot(flujo_caja, label= "Flujo de caja. Dalembert infinita")
plt.axhline(y = caja_infinita, color = 'blue', linestyle = '-')
plt.ylabel("Cantidad en caja")
plt.xlabel("Numero de tirada")
plt.savefig('TP1.2Ruleta/flujo_dalem_infinita.png')
plt.clf()

#Paroli acotada
frec_rel, flujo_caja = ruleta(corrida, "acotada", "paroli")
plt.plot(flujo_caja, label= "Flujo de caja. Paroli acotada")
plt.axhline(y = caja_acotada, color = 'blue', linestyle = '-')
plt.ylabel("Cantidad en caja")
plt.xlabel("Numero de tirada")
plt.savefig('TP1.2Ruleta/flujo_paroli_acotada.png')
plt.clf()

#Paroli infinita
frec_rel, flujo_caja = ruleta(corrida, "infinita", "paroli")
plt.plot(flujo_caja, label= "Flujo de caja. Paroli infinita")
plt.axhline(y = caja_infinita, color = 'blue', linestyle = '-')
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

#Para el grafico de barras
prom_de_frecuencias = []

for i in range(cant_corridas):
    corrida2 = [np.random.randint(0,37) for j in range(n)]
    frec, flujo_martingala_acotada = ruleta(corrida2, "acotada", "martingala")
    flujo_martingala_infinita = ruleta(corrida2, "infinita", "martingala")[1]
    flujo_dalembert_acotada = ruleta(corrida2, "acotada", "dalembert")[1]
    flujo_dalembert_infinita = ruleta(corrida2, "infinita", "dalembert")[1]
    flujo_paroli_acotada = ruleta(corrida2, "acotada", "paroli")[1]
    flujo_paroli_infinita = ruleta(corrida2, "infinita", "paroli")[1]
    
    prom_frec = sum(frec)/n
    prom_de_frecuencias.append(prom_frec)

    ax1.plot(flujo_martingala_acotada)
    ax2.plot(flujo_martingala_infinita)
    ax3.plot(flujo_dalembert_acotada)
    ax4.plot(flujo_dalembert_infinita)
    ax5.plot(flujo_paroli_acotada)
    ax6.plot(flujo_paroli_infinita)
x = np.arange(len(prom_de_frecuencias))#para grafico de barras
print(sum(prom_de_frecuencias)/cant_corridas)

ax1.axhline(y = caja_acotada, color = 'blue', linestyle = '-')
ax1.set_xlabel('n (número de tiradas)')
ax1.set_ylabel('Flujo de caja')

ax2.axhline(y = caja_infinita, color = 'blue', linestyle = '-')
ax2.set_xlabel('n (número de tiradas)')
ax2.set_ylabel('Flujo de caja')

ax3.axhline(y = caja_acotada, color = 'blue', linestyle = '-')
ax3.set_xlabel('n (número de tiradas)')
ax3.set_ylabel('Flujo de caja')

ax4.axhline(y = caja_infinita, color = 'blue', linestyle = '-')
ax4.set_xlabel('n (número de tiradas)')
ax4.set_ylabel('Flujo de caja')

ax5.axhline(y = caja_acotada, color = 'blue', linestyle = '-')
ax5.set_xlabel('n (número de tiradas)')
ax5.set_ylabel('Flujo de caja')

ax6.axhline(y = caja_infinita, color = 'blue', linestyle = '-')
ax6.set_xlabel('n (número de tiradas)')
ax6.set_ylabel('Flujo de caja')

fig1.savefig('TP1.2Ruleta/Mflujo_mart_acotada.png')
fig2.savefig('TP1.2Ruleta/Mflujo_mart_infinita.png')
fig3.savefig('TP1.2Ruleta/Mflujo_delamb_acotada.png')
fig4.savefig('TP1.2Ruleta/Mflujo_delam_infinita.png')
fig5.savefig('TP1.2Ruleta/Mflujo_paroli_acotada.png')
fig6.savefig('TP1.2Ruleta/Mflujo_paroli_infinita.png')

plt.clf()
plt.bar(x, prom_de_frecuencias, color = "blue", width = 0.3)
plt.ylabel("Frecuencia relativa")
plt.xlabel("Corridas")
plt.title("Promedios de frecuencias relativas")
plt.savefig("TP1.2Ruleta/prmedio_frecuencias")