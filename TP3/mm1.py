import random
from numpy import log as ln
import matplotlib.pyplot as plt
from colorama import Fore, Style
import numpy as np

corridas = 1

limite_Q = 200000 # Límite de longitud de la cola
BUSY = 1 # Servidor ocupado
IDLE = 0 # Servidor libre

tasa_prom_llegada = 1.25 # Tasa promedio de llegada
tasa_prom_servicio = 1.0 # Tasa promedio de servicio

area_estado_serv = 0.0 # Area bajo B(t)
area_q = 0.0 # Area bajo Q(t)
cant_clientes_cola = [] # Cantidad de clientes en cola en cada t
estado_servidor = IDLE # B(t)
nro_atenciones_req = 1000 # Número de atenciones requeridas antes de finalizar la simulación
nro_clientes_atendidos = 0 # Número de clientes que han sido atendidos
nro_clientes_q = 0 # Q(t): número actual de clientes en la cola.
nro_clientes_sistema = [] # Número de clientes en el sistema en cada t
nro_eventos_posibles = 0 # Número total de eventos posibles en el sistem
tiempo_llegada = [0.0] * (limite_Q + 1) # Tiempo de llegada
tiempo_prox_ev = [0.0] * 3 # Próximos tiempos_graf de eventos
tiempo_sim = 0.0 
tiempo_ult_ev  = 0.0 # Último tiempo de evento
tipo_prox_evento = None
total_dem = 0.0 # Suma de todas las demoras 
# denegaciones
total_arrivos = 0
denegaciones = {0: 0, 2: 0, 5: 0, 10: 0, 50: 0}
# Para graficas
tiempos_graf = []
estados_servidor_graf = []
nro_clientes_q_graf = []
prob_n_cli_corridas = []

#Promedios
prom_prom_demora_cola = 0
prom_prom_demora_sistema = 0
prom_prom_clientes_cola = 0
prom_prom_clientes_sistema = 0
prom_utilizacion_serv = 0

def expon(mean): # Genera un número aleatorio distirbuido exponencialmente 
    return -float(mean) * ln(random.random())

def calc_prob(lista):
    probs = {}
    
    for n in list(set(lista)):
        probs[n] = lista.count(n) / len(lista)
    
    return probs

def initialize():
    global tiempo_sim, estado_servidor, nro_clientes_q, ult_tiempo_ev, total_dem, nro_clientes_atendidos, area_q, area_estado_serv, tiempo_prox_ev, denegaciones
    global total_arrivos, tipo_prox_evento, cant_clientes_cola, nro_clientes_sistema, tiempo_llegada, tiempo_ult_ev, tiempos_graf, estados_servidor_graf, nro_clientes_q_graf
    
    tipo_prox_evento = None
    tiempo_sim = 0
    estado_servidor = IDLE
    nro_clientes_q = 0
    ult_tiempo_ev = 0.0
    nro_clientes_atendidos = 0
    total_dem = 0.0
    area_q = 0.0
    area_estado_serv = 0.0
    total_arrivos = 0
    denegaciones = {0: 0, 2: 0, 5: 0, 10: 0, 50: 0}
    tiempo_prox_ev[1] = tiempo_sim + expon(tasa_prom_llegada)
    tiempo_prox_ev[2] = 1.0e+30
    cant_clientes_cola = []
    nro_clientes_sistema = []
    tiempo_llegada = [0.0] * (limite_Q + 1)
    tiempo_ult_ev  = 0.0
    tiempos_graf = []
    estados_servidor_graf = []
    nro_clientes_q_graf = []
    
def timing():
    global tipo_prox_evento, tiempo_sim 
    
    min_time_next_event = 1.0e+29
    tipo_prox_evento = 0
    
    # Determino el tipo del proximo evento.
    for i in range(1, nro_eventos_posibles+1):
        if tiempo_prox_ev[i] < min_time_next_event:
            min_time_next_event = tiempo_prox_ev[i]
            tipo_prox_evento = i
    
    # Me fijo si la lista de eventos esta vacia.
    if tipo_prox_evento == 0:
        # Lista vacia, termino la simulacion.
        print(f"\nLista de eventos vacia en {tiempo_sim }")
        exit(1)
    
    # Lista no vacia, avanza el reloj de simulacion.
    tiempo_sim = min_time_next_event

def arrive():
    global nro_clientes_q, estado_servidor, nro_clientes_atendidos, total_dem, tiempo_sim, nro_clientes_sistema, cant_clientes_cola
    global denegaciones, estados_servidor_graf, nro_clientes_q_graf, total_arrivos
    
    # Calculo el proximo arrivo.
    tiempo_prox_ev[1] = tiempo_sim + expon(tasa_prom_llegada)
    
    nro_clientes_sistema.append(nro_clientes_sistema[-1] + 1)
    
    # Me fijo si existe condicion de desbordado (?).
    if nro_clientes_q >= limite_Q:
        # La cola esta desbordada, termino la simulacion
        print(f"\nDesbordamiento del arreglo array tiempo_llegada en el tiempo {tiempo_sim}")
        exit(2)
        
    # Me fijo si el servidor esta ocupado.
    if estado_servidor == BUSY:
        # Servidor ocupado, incremento el numero de clientes en cola.
        nro_clientes_q += 1
        cant_clientes_cola.append(nro_clientes_q)

        # Todavia hay espacio en la cola, guardo el tiempo de arrivo al final de tiempo_llegada
        tiempo_llegada[nro_clientes_q] = tiempo_sim
    else:
        # Servidor desocupado, el cliente tiene demora 0. Lo siguiente es solo para comprension del programa y no afecta la simulacion
        delay = 0.0
        total_dem += delay
        
        cant_clientes_cola.append(0)
        
        # Incremento el numero de clientes demorados y pongo el servidor ocupado.
        nro_clientes_atendidos += 1
        estado_servidor = BUSY
        
        # Calculo salida del cliente
        tiempo_prox_ev[2] = tiempo_sim + expon(tasa_prom_servicio)
        
    # Denegaciones
    total_arrivos += 1
    for cola in [0, 2, 5, 10, 50]:
        if nro_clientes_q > cola:
            denegaciones[cola] += 1

    # Para graficas
    tiempos_graf.append(tiempo_sim)
    estados_servidor_graf.append(estado_servidor)
    nro_clientes_q_graf.append(nro_clientes_q)
    
def depart():
    global nro_clientes_q, estado_servidor, tiempo_prox_ev, total_dem, nro_clientes_atendidos, tiempo_llegada, tiempo_sim, nro_clientes_sistema, cant_clientes_cola
    
    nro_clientes_sistema.append(nro_clientes_sistema[-1] - 1)
    
    # Me fijo si la cola esta vacia
    if nro_clientes_q == 0:
        # Cola vacia, pongo el servidor desocupado y no tengo en concideracion el evento de partida.
        estado_servidor = IDLE
        tiempo_prox_ev[2] = 1.0e+30

    else:
        # Cola no vacia, disminuyo el numero de clientes en cola.
        nro_clientes_q -= 1
        cant_clientes_cola.append(nro_clientes_q)
        
        # Calculo la demora del cliente que entra en servicio y actualizo el acum de demora total.
        delay = tiempo_sim  - tiempo_llegada[1]
        total_dem += delay
        
        # Incremento el numero de clientes demorados y calculo salida.
        nro_clientes_atendidos += 1
        tiempo_prox_ev[2] = tiempo_sim + expon(tasa_prom_servicio)
        
        # Muevo los clientes en cola 1 lugar adelante
        for i in range(1, nro_clientes_q+1):
            tiempo_llegada[i] = tiempo_llegada[i+1]
            
    # Para graficas
    tiempos_graf.append(tiempo_sim)
    estados_servidor_graf.append(estado_servidor)
    nro_clientes_q_graf.append(nro_clientes_q)
            
def report():
    global prom_prom_demora_cola, prom_prom_demora_sistema, prom_prom_clientes_cola, prom_prom_clientes_sistema, prom_utilizacion_serv
    # Calcula y escribe estimados de medidas de performance
    
    # Promedio de demora en cola
    prom_demora_cola = total_dem / nro_clientes_atendidos
    print(f"Promedio demora en cola: {prom_demora_cola:.3f} minutos")
    prom_prom_demora_cola += prom_demora_cola
    
    # Promedio de demora en el sistema
    prom_en_sistema = prom_demora_cola + (1/tasa_prom_servicio)
    print(f"Promedio demora en sistema: {prom_en_sistema:.3f} minutos")
    prom_prom_demora_sistema += prom_en_sistema
    
    # Promedio de numero de clientes en cola
    prom_nro_clientes_q = area_q / tiempo_sim
    print(f"Promedio clientes en cola: {prom_nro_clientes_q:.3f}")
    prom_prom_clientes_cola += prom_nro_clientes_q
    
    # Promedio de numero de clientes en sistema
    prom_nro_clientes_sistema = sum(nro_clientes_sistema) / len(nro_clientes_sistema)
    print(f"Promedio clientes en el sistema: {prom_nro_clientes_sistema:.3f}")
    prom_prom_clientes_sistema += prom_nro_clientes_sistema
    
    # Utilizacion del servidor
    utilizacion_servidor = area_estado_serv / tiempo_sim
    print(f"Utilizacion del servidor: {utilizacion_servidor:.3f}")
    prom_utilizacion_serv += utilizacion_servidor
    
    # Probabilidad de encontrar n clientes en cola
    probabilidades_n_q = calc_prob(cant_clientes_cola)
    prob_n_cli_corridas.append(probabilidades_n_q)
    #print("\nProbabilidad de encontrar n clientes en cola:")
    #print("Numero de clientes\tProbabilidad")
    #for n, p in probabilidades_n_q.items():
    #    print(f"\t{n}\t\t{p}")
        
    # Probabilidad de denegacion de servicio
    print("Tamaño de cola\tProbabilidad de denegación")
    for cola, denegacion in denegaciones.items():
        prob_denegacion = denegacion / total_arrivos
        print(f"\t{cola}\t\t{prob_denegacion}")

def update_time_avg_stats():
    global area_q, ult_tiempo_ev, tiempo_sim, nro_clientes_q, area_estado_serv
    
    # Actualiza acumuladores de area para contadores estadisticos.
    
    # Calculo tiempo desde el ultimo evento y actualizo el marcador de tiempo_ult_ev
    time_since_last_event = tiempo_sim - ult_tiempo_ev
    ult_tiempo_ev = tiempo_sim
    
    # Actualizo area bajo Q(t) (nro_clientes_q)
    area_q += nro_clientes_q * time_since_last_event
    
    # Actualizo el area bajo B(t) (estado_servidor)
    area_estado_serv += estado_servidor * time_since_last_event
            
def simulacion():
    global nro_eventos_posibles, tasa_prom_llegada, tasa_prom_servicio, nro_atenciones_req, nro_clientes_sistema
    
    nro_eventos_posibles = 2
    
    # Imprimo los parametros
    #print("Sistema de cola de un solo servidor\n")
    #print(f"Media de tiempo entre arrivos: {tasa_prom_llegada} minutos\n")
    #print(f"Media del tiempo de servicio: {tasa_prom_servicio} minutos\n")
    #print(f"Numero de clientes: {nro_atenciones_req}\n")
    
    # Llamo a rutina de inicializacion
    initialize()
    
    # Corro la simulacion hasta llegar al limite de Demorados
    nro_clientes_sistema = [0]
    
    while (int(nro_clientes_atendidos) < int(nro_atenciones_req)):
        # Determino el siguiente evento
        timing()
        # Update time-average statistical accumulators.
        update_time_avg_stats()
        # Invoco la rutina de eventos
        if tipo_prox_evento == 1:
            arrive()
        elif tipo_prox_evento == 2:
            depart()
    
    # Invoco el generador de reporte y termino la simulacion
    report()
    
def graficar():
    ## GRAFICAS
    # Grafica Probabilidades de denegacion de servicio
    tamanios_cola = list(denegaciones.keys())
    probabilidades_denegacion = [denegaciones[cola] / nro_atenciones_req for cola in tamanios_cola]
    plt.bar(tamanios_cola, probabilidades_denegacion)
    plt.xlabel('Tamaño de la cola')
    plt.ylabel('Probabilidad de denegación')
    plt.title('Probabilidad de denegación de servicio por tamaño de cola')
    plt.savefig("TP3/prob_denegacion")
    plt.clf()
    
    # Gráfica del estado del servidor en el tiempo
    plt.step(tiempos_graf, estados_servidor_graf, where='post')
    plt.title('Estado del servidor en el tiempo')
    plt.xlabel('Tiempo')
    plt.ylabel('Estado del servidor')
    plt.xlim(0, 1000)
    plt.savefig("TP3/estado_sv")
    plt.clf()
    
    # Gráfica de la cantidad de clientes en cola en el tiempo
    prom_nro_clientes_q_real = sum(nro_clientes_q_graf) / len(nro_clientes_q_graf)
    plt.step(tiempos_graf, nro_clientes_q_graf, where='post')
    plt.axhline(prom_nro_clientes_q_real, color='blue', linestyle='-', label=f'Promedio {prom_nro_clientes_q_real:.3f}')
    plt.title('Cantidad de clientes en cola en el tiempo')
    plt.xlabel('Tiempo')
    plt.ylabel('Clientes en cola')
    plt.xlim(0, 1000)
    plt.legend()
    plt.savefig("TP3/clientes_q")
    plt.clf()
    
    # Grafica del uso del servidor
    porcentaje_ocupado = (area_estado_serv / tiempo_sim) * 100
    porcentaje_ocio = 100 - porcentaje_ocupado
    porcentajes = [porcentaje_ocupado, porcentaje_ocio]
    etiquetas = ['Ocupado', 'Ocio']
    colores = ['lightcoral', 'lightskyblue']
    plt.pie(porcentajes, labels=etiquetas, colors=colores, autopct='%1.1f%%', startangle=90)
    plt.title('Porcentaje de uso del servidor')
    plt.axis('equal')
    plt.savefig("TP3/uso_sv")
    plt.clf()

def main():
    for c in range(10):
        print(Fore.BLUE + f"Corrida {c+1}" + Style.RESET_ALL)
        simulacion()
        print("\n")
        if c == 0: graficar()
    
    #Promedios de promedios
    print(Fore.YELLOW + "Promedios de promedios:" + Style.RESET_ALL)
    dem_cola = prom_prom_demora_cola / 10
    print(f"Demora en cola: {dem_cola}")
    dem_sistema = prom_prom_demora_sistema / 10
    print(f"Demora en sistema: {dem_sistema}")
    cli_cola = prom_prom_clientes_cola / 10
    print(f"Clientes en cola: {cli_cola}")
    cli_sistema = prom_prom_clientes_sistema / 10
    print(f"Clientes en sistema: {cli_sistema}")
    uso_servidor = prom_utilizacion_serv / 10
    print(f"Utilizacion del servidor: {uso_servidor}")
    
    if c == 0: graficar()
    
    # Grafica de probabilidades de encontrar n clientes multiples
    max_clients = max(max(corrida.keys()) for corrida in prob_n_cli_corridas)
    x = np.arange(max_clients + 1)
    bar_width = 0.2
    for i, corrida in enumerate(prob_n_cli_corridas):
        # Obtener las probabilidades en el orden correcto
        prob_list = [corrida.get(n, 0) for n in range(max_clients + 1)]
        # Desplazar cada barra para evitar superposición
        plt.bar(x + i * bar_width, prob_list, width=bar_width, label='Corrida {}'.format(i + 1))
    # Etiquetas de los ejes
    plt.xlabel('Número de clientes en cola')
    plt.ylabel('Probabilidad')
    plt.title('Probabilidad de encontrar n clientes en cola')
    plt.legend()
    plt.savefig('TP3/prob_n_clientes')
    
if __name__ == "__main__":
    main()