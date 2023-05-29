import numpy as np
import matplotlib.pyplot as plt

rojos=[1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36]
negros=[2,4,6,8,10,11,13,15,17,20,22,24,26,28,29,31,33,35]

n = 100  #Número de tiradas de la ruleta
corrida = [np.random.randint(0,37) for i in range(n)]
cant_corridas = 37

caja_acotada = 1000
caja_infinita = 8731
apuesta_inicial = 10

#Función que define la estrategia Martingala
def martingala(corrida, tipo_caja):
    apuesta=apuesta_inicial
    flujo_caja = []
    cont_perdidos = 1
    cont_ganados = 0
    cuando_gane = {tiradas: 0 for tiradas in range(1, n + 1)}
    if tipo_caja == "infinita":
        caja_actual = caja_infinita
    else:
        caja_actual = caja_acotada
    for x in corrida:
        if x in negros:
            caja_actual += apuesta
            flujo_caja.append(caja_actual)
            apuesta = apuesta_inicial
            cont_ganados += 1
            cuando_gane[cont_perdidos] += 1 
            cont_perdidos = 1
        else:
            caja_actual -= apuesta
            flujo_caja.append(caja_actual)
            apuesta = apuesta*2
            cont_perdidos += 1
        if caja_actual < apuesta or caja_actual <= 0: 
            break
    for i in cuando_gane:
        if cont_ganados > 0:
            cuando_gane[i] /= cont_ganados
        else: cuando_gane[i] = 0
    cuando_gane = {k: v for k, v in cuando_gane.items() if v > 0.0}
    return flujo_caja, cuando_gane

#Función que define la estrategia Dalembert
def dalembert(corrida,tipo_caja):
    apuesta=apuesta_inicial
    flujo_caja = []
    cont_perdidos = 1
    cont_ganados = 0
    cuando_gane = {tiradas: 0 for tiradas in range(1, n + 1)}
    if tipo_caja == "infinita":
        caja_actual = caja_infinita
    else:
        caja_actual = caja_acotada
    for x in corrida:
        if x in negros:
            caja_actual += apuesta
            flujo_caja.append(caja_actual)
            if apuesta > 1:
                apuesta = apuesta - 1
            cont_ganados += 1
            cuando_gane[cont_perdidos] += 1 
            cont_perdidos = 1
        else:
            caja_actual -= apuesta
            flujo_caja.append(caja_actual)
            apuesta = apuesta + 1
            cont_perdidos += 1
        if caja_actual < apuesta or caja_actual<=0: 
            break
    for i in cuando_gane:
        if cont_ganados > 0:
            cuando_gane[i] /= cont_ganados
        else: cuando_gane[i] = 0
    cuando_gane = {k: v for k, v in cuando_gane.items() if v > 0.0}
    return flujo_caja, cuando_gane

#Función que define la estrategia de Paroli
def paroli(corrida, tipo_caja):
    apuesta=apuesta_inicial
    flujo_caja = []
    cont_perdidos = 1
    cont_ganados = 0
    cuando_gane = {tiradas: 0 for tiradas in range(1, n + 1)}
    victorias_consecutivas = 0
    objetivo_victorias = 3
    if tipo_caja == "infinita":
        caja_actual = caja_infinita
    else:
        caja_actual = caja_acotada
    for x in corrida:
        if x in negros:
            victorias_consecutivas += 1
            caja_actual += apuesta
            flujo_caja.append(caja_actual)
            if victorias_consecutivas == objetivo_victorias:
                apuesta = apuesta_inicial
                victorias_consecutivas = 0
            else:
                apuesta *= 2
            cont_ganados += 1
            cuando_gane[cont_perdidos] += 1 
            cont_perdidos = 1
        else:
            caja_actual -= apuesta
            flujo_caja.append(caja_actual)
            apuesta = apuesta_inicial
            victorias_consecutivas = 0
            cont_perdidos += 1
        if caja_actual < apuesta or caja_actual<=0: 
            break
    for i in cuando_gane:
        if cont_ganados > 0:
            cuando_gane[i] /= cont_ganados
        else: cuando_gane[i] = 0
    cuando_gane = {k: v for k, v in cuando_gane.items() if v > 0.0}
    return flujo_caja, cuando_gane

#Martingala acotada
flujo_caja_mart_acotada, cuando_gane_mart_acotada = martingala(corrida,"acotada")

plt.title("Martingala con capital acotado")
plt.plot(flujo_caja_mart_acotada, label= "Flujo de caja. Martingala acotada")
plt.axhline(y = caja_acotada, color = 'red', linestyle = '-')
plt.ylabel("Cantidad en caja")
plt.xlabel("Numero de tirada")
plt.ticklabel_format(useOffset=False, style='plain')
plt.savefig('TP1.2Ruleta/flujo_mart_acotada.png')
plt.clf()

plt.bar(cuando_gane_mart_acotada.keys(), cuando_gane_mart_acotada.values(), color ='red', width = 0.4)
plt.title("Martingala con capital acotado")
plt.xlabel("Nro de veces para ganar luego de haber perdido")
plt.ylabel("Frecuencia relativa")
plt.ticklabel_format(useOffset=False, style='plain')
plt.savefig('TP1.2Ruleta/frec_mart_acotada.png')
plt.clf()

#Martingala infinita
flujo_caja_mart_infinita, cuando_gane_mart_infinita = martingala(corrida, "infinita")

plt.title("Martingala con capital infinito")
plt.plot(flujo_caja_mart_infinita, label= "Flujo de caja. Martingala infinita")
plt.axhline(y = caja_infinita, color = 'red', linestyle = '-')
plt.ylabel("Cantidad en caja")
plt.xlabel("Numero de tirada")
plt.ticklabel_format(useOffset=False, style='plain')
plt.savefig('TP1.2Ruleta/flujo_mart_infinita.png')
plt.clf()

plt.bar(cuando_gane_mart_infinita.keys(), cuando_gane_mart_infinita.values(), color ='red', width = 0.4)
plt.title("Martingala con capital infinito")
plt.xlabel("Nro de veces para ganar luego de haber perdido")
plt.ylabel("Frecuencia relativa")
plt.ticklabel_format(useOffset=False, style='plain')
plt.savefig('TP1.2Ruleta/frec_mart_infinita.png')
plt.clf()

#Dalembert acotada
flujo_caja_dal_acotada, cuando_gane_dal_acotada = dalembert(corrida, "acotada")

plt.title("Dalembert con capital acotado")
plt.plot(flujo_caja_dal_acotada, label= "Flujo de caja. Dalembert acotada")
plt.axhline(y = caja_acotada, color = 'red', linestyle = '-')
plt.ylabel("Cantidad en caja")
plt.xlabel("Numero de tirada")
plt.ticklabel_format(useOffset=False, style='plain')
plt.savefig('TP1.2Ruleta/flujo_dal_acotada.png')
plt.clf()

plt.bar(cuando_gane_dal_acotada.keys(), cuando_gane_dal_acotada.values(), color ='red', width = 0.4)
plt.title("Dalembert con capital acotado")
plt.xlabel("Nro de veces para ganar luego de haber perdido")
plt.ylabel("Frecuencia relativa")
plt.ticklabel_format(useOffset=False, style='plain')
plt.savefig('TP1.2Ruleta/frec_dal_acotada.png')
plt.clf()

#Dalembert infinita
flujo_caja_dal_infinita, cuando_gane_dal_infinita = dalembert(corrida, "infinita")

plt.title("Dalembert con capital infinito")
plt.plot(flujo_caja_dal_infinita, label= "Flujo de caja. Dalembert infinita")
plt.axhline(y = caja_infinita, color = 'red', linestyle = '-')
plt.ylabel("Cantidad en caja")
plt.xlabel("Numero de tirada")
plt.ticklabel_format(useOffset=False, style='plain')
plt.savefig('TP1.2Ruleta/flujo_dal_infinita.png')
plt.clf()

plt.bar(cuando_gane_dal_infinita.keys(), cuando_gane_dal_infinita.values(), color ='red', width = 0.4)
plt.title("Dalembert con capital infinito")
plt.xlabel("Nro de veces para ganar luego de haber perdido")
plt.ylabel("Frecuencia relativa")
plt.ticklabel_format(useOffset=False, style='plain')
plt.savefig('TP1.2Ruleta/frec_dal_infinita.png')
plt.clf()

#Paroli acotada
flujo_caja_par_acotada, cuando_gane_par_acotada = paroli(corrida, "acotada")

plt.title("Paroli con capital acotado")
plt.plot(flujo_caja_par_acotada, label= "Flujo de caja. Paroli acotada")
plt.axhline(y = caja_acotada, color = 'red', linestyle = '-')
plt.ylabel("Cantidad en caja")
plt.xlabel("Numero de tirada")
plt.ticklabel_format(useOffset=False, style='plain')
plt.savefig('TP1.2Ruleta/flujo_paroli_acotada.png')
plt.clf()

plt.bar(cuando_gane_par_acotada.keys(), cuando_gane_par_acotada.values(), color ='red', width = 0.4)
plt.title("Paroli con capital acotado")
plt.xlabel("Nro de veces para ganar luego de haber perdido")
plt.ylabel("Frecuencia relativa")
plt.ticklabel_format(useOffset=False, style='plain')
plt.savefig('TP1.2Ruleta/frec_paroli_acotada.png')
plt.clf()

#Paroli infinita
flujo_caja_par_infinita, cuando_gane_par_infinita = paroli(corrida, "infinita")

plt.title("Paroli con capital infinito")
plt.plot(flujo_caja_par_infinita, label= "Flujo de caja. Paroli infinita")
plt.axhline(y = caja_infinita, color = 'red', linestyle = '-')
plt.ylabel("Cantidad en caja")
plt.xlabel("Numero de tirada")
plt.ticklabel_format(useOffset=False, style='plain')
plt.savefig('TP1.2Ruleta/flujo_paroli_infinita.png')
plt.clf()

plt.bar(cuando_gane_par_infinita.keys(), cuando_gane_par_infinita.values(), color ='red', width = 0.4)
plt.title("Paroli con capital infinito")
plt.xlabel("Nro de veces para ganar luego de haber perdido")
plt.ylabel("Frecuencia relativa")
plt.ticklabel_format(useOffset=False, style='plain')
plt.savefig('TP1.2Ruleta/frec_paroli_infinita.png')
plt.clf()


#GRAFICAS MULTIPLES

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()
fig5, ax5 = plt.subplots()
fig6, ax6 = plt.subplots()
fig7, ax7 = plt.subplots()
fig8, ax8 = plt.subplots()
fig9, ax9 = plt.subplots()
fig10, ax10 = plt.subplots()
fig11, ax11 = plt.subplots()
fig12, ax12 = plt.subplots()

for i in range(15):
    corrida2 = [np.random.randint(0,37) for j in range(n)]
    flujo_martingala_acotada, cuando_gane_martingala_acotada= martingala(corrida2, "acotada")
    flujo_martingala_infinita, cuando_gane_martingala_infinita = martingala(corrida2, "infinita")
    flujo_dalembert_acotada, cuando_gane_dalembert_acotada = dalembert(corrida2, "acotada")
    flujo_dalembert_infinita, cuando_gane_dalembert_infinita = dalembert(corrida2, "infinita")
    flujo_paroli_acotada, cuando_gane_paroli_acotada = paroli(corrida2, "acotada")
    flujo_paroli_infinita, cuando_gane_paroli_infinita  = paroli(corrida2, "infinita")


    ax1.plot(flujo_martingala_acotada)
    ax2.bar(cuando_gane_martingala_acotada.keys(), cuando_gane_martingala_acotada.values())
    ax3.plot(flujo_martingala_infinita)
    ax4.bar(cuando_gane_martingala_infinita.keys(), cuando_gane_martingala_infinita.values())
    ax5.plot(flujo_dalembert_acotada)
    ax6.bar(cuando_gane_dalembert_acotada.keys(), cuando_gane_dalembert_acotada.values())
    ax7.plot(flujo_dalembert_infinita)
    ax8.bar(cuando_gane_dalembert_infinita.keys(), cuando_gane_dalembert_infinita.values())
    ax9.plot(flujo_paroli_acotada)
    ax10.bar(cuando_gane_paroli_acotada.keys(), cuando_gane_paroli_acotada.values())
    ax11.plot(flujo_paroli_infinita)
    ax12.bar(cuando_gane_paroli_infinita.keys(), cuando_gane_paroli_infinita.values())

#Martingala acotada
ax1.set_title('Flujo martingala acotada')
ax1.axhline(y = caja_acotada, color = 'red', linestyle = '-')
ax1.set_xlabel('n (número de tiradas)')
ax1.set_ylabel('Flujo de caja')


ax2.set_title("Martingala con capital acotado")
ax2.set_xlabel("Nro de veces para ganar luego de haber perdido")
ax2.set_ylabel("Frecuencia relativa")
ax2.ticklabel_format(useOffset=False, style='plain')

#Martingala infinita
ax3.set_title('Flujo martingala infinita')
ax3.axhline(y = caja_infinita, color = 'red', linestyle = '-')
ax3.set_xlabel('n (número de tiradas)')
ax3.set_ylabel('Flujo de caja')
ax3.ticklabel_format(useOffset=False, style='plain')


ax4.set_title("Martingala con capital infinito")
ax4.set_xlabel("Nro de veces para ganar luego de haber perdido")
ax4.set_ylabel("Frecuencia relativa")
ax4.ticklabel_format(useOffset=False, style='plain')


#Dalembert acotada
ax5.set_title('Flujo dalembert acotada')
ax5.axhline(y = caja_acotada, color = 'red', linestyle = '-')
ax5.set_xlabel('n (número de tiradas)')
ax5.set_ylabel('Flujo de caja')


ax6.set_title("Dalembert con capital acotado")
ax6.set_xlabel("Nro de veces para ganar luego de haber perdido")
ax6.set_ylabel("Frecuencia relativa")
ax6.ticklabel_format(useOffset=False, style='plain')


#Dalembert infinita
ax7.set_title('Flujo dalembert infinita')
ax7.axhline(y = caja_infinita, color = 'red', linestyle = '-')
ax7.set_xlabel('n (número de tiradas)')
ax7.set_ylabel('Flujo de caja')
ax7.ticklabel_format(useOffset=False, style='plain')


ax8.set_title("Dalembert con capital infinito")
ax8.set_xlabel("Nro de veces para ganar luego de haber perdido")
ax8.set_ylabel("Frecuencia relativa")
ax8.ticklabel_format(useOffset=False, style='plain')


#Paroli acotada
ax9.set_title('Flujo paroli acotada')
ax9.axhline(y = caja_acotada, color = 'red', linestyle = '-')
ax9.set_xlabel('n (número de tiradas)')
ax9.set_ylabel('Flujo de caja')


ax10.set_title("Paroli con capital acotado")
ax10.set_xlabel("Nro de veces para ganar luego de haber perdido")
ax10.set_ylabel("Frecuencia relativa")
ax10.ticklabel_format(useOffset=False, style='plain')


#Paroli infinita
ax11.set_title('Flujo paroli infinita')
ax11.axhline(y = caja_infinita, color = 'red', linestyle = '-')
ax11.set_xlabel('n (número de tiradas)')
ax11.set_ylabel('Flujo de caja')
ax11.ticklabel_format(useOffset=False, style='plain')


ax12.set_title("Paroli con capital infinito")
ax12.set_xlabel("Nro de veces para ganar luego de haber perdido")
ax12.set_ylabel("Frecuencia relativa")
ax12.ticklabel_format(useOffset=False, style='plain')




fig1.savefig('TP1.2Ruleta/Mflujo_mart_acotada.png')
fig2.savefig('TP1.2Ruleta/Mfrec_mart_acotada.png')
fig3.savefig('TP1.2Ruleta/Mflujo_mart_infinita.png')
fig4.savefig('TP1.2Ruleta/Mfrec_mart_infinita.png')
fig5.savefig('TP1.2Ruleta/Mflujo_delamb_acotada.png')
fig6.savefig('TP1.2Ruleta/Mfrec_delamb_acotada.png')
fig7.savefig('TP1.2Ruleta/Mflujo_delam_infinita.png')
fig8.savefig('TP1.2Ruleta/Mfrec_delam_infinita.png')
fig9.savefig('TP1.2Ruleta/Mflujo_paroli_acotada.png')
fig10.savefig('TP1.2Ruleta/Mfrec_paroli_acotada.png')
fig11.savefig('TP1.2Ruleta/Mflujo_paroli_infinita.png')
fig12.savefig('TP1.2Ruleta/Mfrec_paroli_infinita.png')
