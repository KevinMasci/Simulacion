import random
from numpy import log as ln
import matplotlib.pyplot as plt

corridas = 10

# Variables globales
cantidad = 0
nivel_max_inv = 0
nivel_ini_inv = 60
nivel_act_inv = 0
prox_ev = 0
num_ev = 0
num_meses = 120
num_tamanio_demanda = 4
num_politicas = 9
nivel_min_inv = 0
area_costo_mant = 0.0
area_costo_escasez = 0.0
costo_mantenimiento = 1.0
costo_incremental= 3.0
retraso_max = 1.0
t_prom_entre_dem = 0.10
retraso_min = 0.50
distrib_prob_dem = [0.0, 0.167, 0.5, 0.833, 1.0]
costo_config = 32.0
costo_escasez = 5.0
tiempo_sim = 0.0
t_ult_ev = 0.0
t_prox_ev = [0.0] * 5
costo_total = 0.0
politicas = [[20, 40], [20, 60], [20, 80], [20, 100], [40, 60], [40, 80], [40, 100], [60, 80], [60, 100]]

# Para graficas
nivel_inventario_graf = [nivel_ini_inv]
tiempos_graf = [0.0]
demanda_graf = [0.0]

prom_prom_costo_esc = [0.0] * len(politicas)
prom_prom_costo_mant = [0.0] * len(politicas)
prom_prom_costo_pedido = [0.0] * len(politicas)
prom_prom_costo_tot = [0.0] * len(politicas)

def expon(t_prom_entre_dem):
    return -t_prom_entre_dem * ln(random.random())

def random_integer(prob_distrib):
    u = random.random()  # Genera una variable aleatoria U(0,1)
    i = 1
    while u >= prob_distrib[i]:
        i += 1
    return i

def uniform(a, b):
    # Devuelve una variable aleatoria U(a,b).
    return a + random.random() * (b - a)

def simulacion():
    global num_ev, nivel_ini_inv, num_meses, num_politicas
    global num_tamanio_demanda, t_prom_entre_dem, costo_config, costo_incremental, costo_mantenimiento
    global costo_escasez, retraso_min, retraso_max, distrib_prob_dem , nivel_min_inv, nivel_max_inv, prox_ev

    # Especifico numero de eventos
    num_ev = 4

    # Escribe el encabezado del informe y los parámetros de entrada
    print("Sistema de Inventario de un solo producto\n")
    print(f"Nivel Inicial de Inventario {nivel_ini_inv} items\n")
    print(f"Número de Tamaños de Demanda {num_tamanio_demanda}\n")
    print("Función de Distribución de tamaños de demanda ")
    for i in range(1, num_tamanio_demanda + 1):
        print(str(distrib_prob_dem [i]))
    print("\n")
    print(f"Tiempo Medio entre Demanda {t_prom_entre_dem}\n")
    print(f"Rango de Demora de Entrega {retraso_min} a {retraso_max} meses\n")
    print(f"Duración de la Simulación {num_meses} meses\n")
    print(f"K = {costo_config} i = {costo_incremental} h = {costo_mantenimiento} pi = {costo_escasez}\n")
    print(f"Número de Políticas {num_politicas}\n\n")
    print(" \t\t Promedio \t Promeido \t Promedio \t Promedio")
    print(" Política \tcosto total\tcosto de pedido\tcosto de mantenimiento\tcosto de escasez")

    # Ejecuta la simulación variando la política de inventario
    for i in range(num_politicas):

        # Lee la política de inventario y inicializa la simulación
        nivel_min_inv, nivel_max_inv = politicas[i]
        initialize()

        # Ejecuta la simulación hasta que termine después de un evento de fin de simulación (tipo 3)
        while prox_ev != 3:
            # Determina el siguiente evento
            timing()
            # Actualiza los acumuladores estadísticos promedio de tiempo
            update_time_avg_stats()
            # Invoca la función de evento adecuada
            if prox_ev == 1:
                order_arrival()
            elif prox_ev == 2:
                demand()
            elif prox_ev == 4:
                evaluate()
            elif prox_ev == 3:
                report(i)
                graficar(i)

def initialize():
    global tiempo_sim, nivel_act_inv, t_ult_ev, costo_total, area_costo_mant, area_costo_escasez, t_prox_ev, prox_ev, tiempos_graf, nivel_inventario_graf, costos_acum_graf, tiempo_costos_graf
    global demanda_graf

    prox_ev = 0

    # Inicializa el reloj de la simulación
    tiempo_sim = 0.0

    # Inicializa las variables de estado
    nivel_act_inv = nivel_ini_inv
    t_ult_ev = 0.0

    # Inicializa los contadores estadísticos
    costo_total = 0.0
    area_costo_mant = 0.0
    area_costo_escasez = 0.0

    # Inicializa la lista de eventos. Dado que no hay pedidos pendientes, el evento de llegada de pedidos se elimina de la consideración.
    t_prox_ev[1] = 1.0e+30
    t_prox_ev[2] = tiempo_sim + expon(t_prom_entre_dem)
    t_prox_ev[3] = num_meses
    t_prox_ev[4] = 0.0
    
    tiempos_graf = [0.0]
    nivel_inventario_graf = [nivel_ini_inv]
    costos_acum_graf = [0.0]
    tiempo_costos_graf = [0.0]
    demanda_graf = [0.0]

def timing():
    global prox_ev, tiempo_sim

    min_t_prov_ev = 1.0e+29
    prox_ev = 0

    # Determino el tipo del proximo evento.
    for i in range(1, num_ev+1):
        if t_prox_ev[i] < min_t_prov_ev:
            min_t_prov_ev = t_prox_ev[i]
            prox_ev = i

    # Me fijo si la lista de eventos esta vacia.
    if prox_ev == 0:
        # Lista vacia, termino la simulacion.
        print(f"\nLista de eventos vacia en {tiempo_sim}")
        exit(1)

    # Lista no vacia, avanza el reloj de simulacion.
    tiempo_sim = min_t_prov_ev

def order_arrival():
    global nivel_act_inv, t_prox_ev, nivel_inventario_graf, tiempos_graf

    # Incrementa el nivel de inventario según la cantidad pedida
    nivel_act_inv += cantidad
    
    # Agrego el nuevo nivel de inventario al array para graficar
    nivel_inventario_graf.append(nivel_act_inv)
    
    # Agrego el tiempo del evento al array para graficar
    tiempos_graf.append(tiempo_sim)

    # Dado que no hay un pedido pendiente, elimina el evento de llegada de pedidos de la consideración
    t_prox_ev[1] = 1.0e+30

def demand():
    global nivel_act_inv, t_prox_ev, nivel_inventario_graf

    # Decrementa el nivel de inventario según el tamaño de la demanda generado
    tam_demanda = random_integer(distrib_prob_dem )
    nivel_act_inv -= tam_demanda
    
    # Agrego el tamaño de la demanda al array para graficar
    demanda_graf.append(tam_demanda)

    # Agrego el nuevo nivel de inventario al array para graficar
    nivel_inventario_graf.append(nivel_act_inv)
    
    # Agrega el tiempo del evento al array para graficar
    tiempos_graf.append(tiempo_sim)
    
    # Programa el tiempo de la próxima demanda
    t_prox_ev[2] = tiempo_sim + expon(t_prom_entre_dem)

def evaluate():
    global nivel_act_inv, cantidad, costo_total, tiempo_costos_graf

    # Verificar si el nivel de inventario es menor que "nivel_min_inv".
    if nivel_act_inv < nivel_min_inv:
        # El nivel de inventario es menor que "nivel_min_inv", por lo tanto, realizar un pedido por la cantidad correspondiente.
        cantidad = nivel_max_inv - nivel_act_inv
        costo_total += costo_config + costo_incremental * cantidad
        # Programar la llegada del pedido.
        t_prox_ev[1] = tiempo_sim + uniform(retraso_min, retraso_max)

    # Independientemente de la decisión de realizar un pedido, programar la próxima evaluación del inventario.
    t_prox_ev[4] = tiempo_sim + 1.0
    
    costos_acum_graf.append(costos_acum_graf[-1] + costo_total)
    tiempo_costos_graf.append(tiempo_sim)

def report(i):
    global prom_prom_costo_esc, prom_prom_costo_mant, prom_prom_costo_pedido, prom_prom_costo_tot
    # Calcula y escribe las estimaciones de las medidas de rendimiento deseadas.
    avg_holding_cost = costo_mantenimiento * area_costo_mant / num_meses
    avg_ordering_cost = costo_total / num_meses
    avg_shortage_cost = costo_escasez * area_costo_escasez / num_meses
    print(f"\n({nivel_min_inv:3d},{nivel_max_inv:3d}){avg_ordering_cost + avg_holding_cost + avg_shortage_cost:15.2f}{avg_ordering_cost:15.2f}{avg_holding_cost:15.2f}{avg_shortage_cost:15.2f}") 

    # Para corridas multiples
    prom_prom_costo_esc[i] += avg_shortage_cost
    prom_prom_costo_mant[i] += avg_holding_cost
    prom_prom_costo_pedido[i] += avg_ordering_cost
    prom_prom_costo_tot[i] += (avg_ordering_cost + avg_holding_cost + avg_shortage_cost)
    
    
def update_time_avg_stats():
    global t_ult_ev, tiempo_sim, area_costo_escasez, area_costo_mant, costos_acum_graf, costo_total_graf, costo_pedido_graf, costo_mantenimiento_graf, costo_escasez_graf, tiempo_graf

    # Calcula el tiempo transcurrido desde el último evento y actualiza el marcador de último evento.
    time_since_last_event = tiempo_sim - t_ult_ev
    t_ult_ev = tiempo_sim

    # Determina el estado del nivel de inventario durante el intervalo anterior.
    # Si el nivel de inventario durante el intervalo anterior fue negativo, actualiza area_costo_escasez.
    # Si fue positivo, actualiza area_costo_mant. Si fue cero, no se necesita actualización.
    if nivel_act_inv < 0:
        area_costo_escasez -= nivel_act_inv * time_since_last_event
    elif nivel_act_inv > 0:
        area_costo_mant += nivel_act_inv * time_since_last_event
        
def graficar(i):
    # Nivel de inventario en el tiempo
    prom = sum(nivel_inventario_graf) / len(nivel_inventario_graf)
    plt.step(tiempos_graf, nivel_inventario_graf, where="post")
    plt.axhline(nivel_max_inv, color='green', linestyle='--', label='Nivel maximo')
    plt.axhline(nivel_min_inv, color='green', linestyle='--', label='Nivel minimo')
    plt.axhline(prom, color='red', linestyle='-', label='Promedio')
    plt.xlabel('Tiempo')
    plt.ylabel('Nivel de inventario')
    plt.title('Nivel de inventario a lo largo de la simulación')
    plt.legend()
    plt.savefig(f"TP3/nivel_inv_pol{i+1}")
    plt.clf()

def main():
    for x in range(corridas):
        simulacion()
    print("\nPromedios de promedios")
    print(" Política \tcosto total\tcosto de pedido\tcosto de mantenimiento\tcosto de escasez")
    for x in range(num_politicas):
        prom_esc = prom_prom_costo_esc[x] / corridas
        prom_mant = prom_prom_costo_mant[x] / corridas
        prom_ped = prom_prom_costo_pedido[x] / corridas
        prom_tot = prom_prom_costo_tot[x] / corridas
        print(f"\n({politicas[x][0]:3d},{politicas[x][1]:3d}){prom_tot:15.2f}{prom_ped:15.2f}{prom_mant:15.2f}{prom_esc:15.2f}")

if __name__ == "__main__":
    main()