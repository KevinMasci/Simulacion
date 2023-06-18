import random
from numpy import log as ln
import matplotlib.pyplot as plt
import numpy as np

limite_Q = 100 # Límite de longitud de la cola
BUSY = 1 # Servidor ocupado
IDLE = 0 # Servidor libre

tipo_prox_evento = None
nro_clientes_atendidos = 0 # Número de clientes que han sido atendidos
cant_clientes_cola = [] # Cantidad de clientes en cola en cada t
nro_clientes_sistema = [] # Número de clientes en el sistema en cada t
nro_atenciones_req = 1000 # Número de atenciones requeridas antes de finalizar la simulación
nro_eventos_posibles = 0 # Número total de eventos posibles en el sistem
nro_clientes_q = 0 # Q(t): número actual de clientes en la cola.
estado_servidor = IDLE # B(t)
area_q = 0.0 # Area bajo Q(t)
area_estado_serv = 0.0 # Area bajo B(t)
tasa_prom_llegada = 1.0 # Tasa promedio de llegada
tasa_prom_servicio = 0.5 # Tasa promedio de servicio
tiempo_sim = 0.0 
tiempo_llegada = [0.0] * (limite_Q + 1) # Tiempo de llegada
tiempo_ult_ev  = 0.0 # Último tiempo de evento
tiempo_prox_ev = [0.0] * 3 # Próximos tiempos_graf de eventos
total_dem = 0.0 # Suma de todas las demoras 
# denegaciones
denegaciones = {0: 0, 2: 0, 5: 0, 10: 0, 50: 0}
# Para graficas
tiempos_graf = []
estados_servidor_graf = []
nro_clientes_q_graf = []


def expon(mean): # Genera un número aleatorio distirbuido exponencialmente 
    return -float(mean) * ln(random.random())

def calc_prob(lista):
    probs = {}
    
    for n in list(set(lista)):
        probs[n] = lista.count(n) / len(lista)
    
    return probs

def initialize():
    global tiempo_sim, estado_servidor, nro_clientes_q, ult_tiempo_ev, total_dem, nro_clientes_atendidos, area_q, area_estado_serv, tiempo_prox_ev
    
    # Reloj de simulacion
    tiempo_sim = 0
    
    # Variables de estado
    estado_servidor = IDLE
    nro_clientes_q = 0
    ult_tiempo_ev = 0.0
    
    # Contadores estadisticos
    nro_clientes_atendidos = 0
    total_dem = 0.0
    area_q = 0.0
    area_estado_serv = 0.0
    
    # Lista de eventos inicial
    tiempo_prox_ev[1] = tiempo_sim + expon(tasa_prom_llegada)
    tiempo_prox_ev[2] = 1.0e+30
    
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
    global denegaciones, estados_servidor_graf, nro_clientes_q_graf
    
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
    # Calcula y escribe estimados de medidas de performance
    
    # Promedio de demora en cola
    prom_demora_cola = total_dem / nro_clientes_atendidos
    print(f"\nPromedio de demora en cola: {prom_demora_cola:.3f} minutos")
    
    # Promedio de demora en el sistema
    prom_prom_en_sistema = prom_demora_cola / tasa_prom_servicio
    print(f"\nPromedio de demora en servicio: {prom_prom_en_sistema:.3f} minutos")
    
    # Promedio de numero de clientes en cola
    prom_nro_clientes_q = area_q / tiempo_sim
    print(f"\nPromedio de numero de clientes en cola: {prom_nro_clientes_q:.3f}")
    
    # Promedio de numero de clientes en sistema
    prom_nro_clientes_sistema = sum(nro_clientes_sistema) / len(nro_clientes_sistema)
    print(f"\nPromedio de numero de clientes en el sistema: {prom_nro_clientes_sistema:.3f}")
    
    # Utilizacion del servidor
    utilizacion_servidor = area_estado_serv / tiempo_sim
    print(f"\nUtilizacion del servidor: {utilizacion_servidor:.3f}")
    
    # Tiempo de fin de simulacion
    print(f"\nTiempo de fin de simulacion: {tiempo_sim:.3f} minutos")
    
    # Probabilidad de encontrar n clientes en cola
    probabilidades_n_q = calc_prob(cant_clientes_cola)
    print("\nProbabilidad de encontrar n clientes en cola:")
    print("Numero de clientes\tProbabilidad")
    for n, p in probabilidades_n_q.items():
        print(f"\t{n}\t\t{p}")
        
    # Probabilidad de denegacion de servicio
    print("Tamaño de cola\tProbabilidad de denegación")
    for cola, denegacion in denegaciones.items():
        prob_denegacion = denegacion / nro_atenciones_req
        print(f"\t{cola}\t\t{prob_denegacion}")
    
    # GRAFICAS
    # Graficas Probabilidades de encontrar N clientes en cola.
    n = list(probabilidades_n_q.keys())
    valores = list(probabilidades_n_q.values())
    plt.bar(n, valores)
    plt.xlabel('Cantidad de clientes')
    plt.ylabel('Probabilidad')
    plt.title('Probabilidad de encontrar N cliente en cola')
    plt.show()
    
    # Grafica Probabilidades de denegacion de servicio
    tamanios_cola = list(denegaciones.keys())
    probabilidades_denegacion = [denegaciones[cola] / nro_atenciones_req for cola in tamanios_cola]
    plt.bar(tamanios_cola, probabilidades_denegacion)
    plt.xlabel('Tamaño de la cola')
    plt.ylabel('Probabilidad de denegación')
    plt.title('Probabilidad de denegación de servicio por tamaño de cola')
    plt.show()
    
    # Gráfica del estado del servidor en el tiempo
    plt.step(tiempos_graf, estados_servidor_graf, where='post')
    plt.title('Estado del servidor en el tiempo')
    plt.xlabel('Tiempo')
    plt.ylabel('Estado del servidor')
    plt.xlim(0, 100)
    plt.show()
    
    # Gráfica de la cantidad de clientes en cola en el tiempo
    plt.step(tiempos_graf, nro_clientes_q_graf, where='post')
    plt.title('Cantidad de clientes en cola en el tiempo')
    plt.xlabel('Tiempo')
    plt.ylabel('Clientes en cola')
    plt.xlim(0, 100)
    plt.show()
    
    # Grafica del uso del servidor
    porcentaje_ocupado = (area_estado_serv / tiempo_sim) * 100
    porcentaje_ocio = 100 - porcentaje_ocupado
    porcentajes = [porcentaje_ocupado, porcentaje_ocio]
    etiquetas = ['Ocupado', 'Ocio']
    colores = ['lightcoral', 'lightskyblue']
    plt.pie(porcentajes, labels=etiquetas, colors=colores, autopct='%1.1f%%', startangle=90)
    plt.title('Porcentaje de uso del servidor')
    plt.axis('equal')
    plt.show()

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
            
def main():
    global nro_eventos_posibles, tasa_prom_llegada, tasa_prom_servicio, nro_atenciones_req, nro_clientes_sistema
    
    nro_eventos_posibles = 2
    
    # Imprimo los parametros
    print("Sistema de cola de un solo servidor\n")
    print(f"Media de tiempo entre arrivos: {tasa_prom_llegada} minutos\n")
    print(f"Media del tiempo de servicio: {tasa_prom_servicio} minutos\n")
    print(f"Numero de clientes: {nro_atenciones_req}\n")
    
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
    
if __name__ == "__main__":
    main()