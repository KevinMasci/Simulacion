import random
from numpy import log as ln

# Variables globales
amount = 0
bigs = 0
initial_inv_level = 60
inv_level = 0
next_event_type = 0
num_events = 0
num_months = 120
num_values_demand = 4
num_policies = 9
smalls = 0
area_holding = 0.0
area_shortage = 0.0
holding_cost = 1.0
incremental_cost = 3.0
maxlag = 1.0
mean_interdemand = 0.10
minlag = 0.50
prob_distrib_demand = [0.0, 0.167, 0.5, 0.833, 1.0]
setup_cost = 32.0
shortage_cost = 5.0
sim_time = 0.0
time_last_event = 0.0
time_next_event = [0.0] * 5
total_ordering_cost = 0.0
policies = [[20, 40], [20, 60], [20, 80], [20, 100], [40, 60], [40, 80], [40, 100], [60, 80], [60, 100]]

def expon(mean_interdemand):
    return -mean_interdemand * ln(random.random())

def random_integer(prob_distrib):
    u = random.random()  # Genera una variable aleatoria U(0,1)
    i = 1
    while u >= prob_distrib[i]:
        i += 1
    return i

def uniform(a, b):
    # Devuelve una variable aleatoria U(a,b).
    return a + random.random() * (b - a)

def main():
    global num_events, initial_inv_level, num_months, num_policies
    global num_values_demand, mean_interdemand, setup_cost, incremental_cost, holding_cost
    global shortage_cost, minlag, maxlag, prob_distrib_demand, smalls, bigs, next_event_type
    
    # Especifico numero de eventos
    num_events = 4
        
    # Escribe el encabezado del informe y los parámetros de entrada
    print("Single-product inventory system\n")
    print(f"Initial inventory level {initial_inv_level} items\n")
    print(f"Number of demand sizes {num_values_demand}\n")
    print("Distribution function of demand sizes ")
    for i in range(1, num_values_demand + 1):
        print(str(prob_distrib_demand[i]))
    print("\n")
    print(f"Mean interdemand time {mean_interdemand}\n")
    print(f"Delivery lag range {minlag} to {maxlag} months\n")
    print(f"Length of the simulation {num_months} months\n")
    print(f"K = {setup_cost} i = {incremental_cost} h = {holding_cost} pi = {shortage_cost}\n")
    print(f"Number of policies {num_policies}\n\n")
    print(" \t\t Average \t Average \t Average \t Average")
    print(" Policy \ttotal cost\tordering cost\tholding cost\tshortage cost")

    # Ejecuta la simulación variando la política de inventario
    for i in range(num_policies):
        
        # Lee la política de inventario y inicializa la simulación
        smalls, bigs = policies[i]
        initialize()
        
        # Ejecuta la simulación hasta que termine después de un evento de fin de simulación (tipo 3)
        while next_event_type != 3:
            # Determina el siguiente evento
            timing()
            # Actualiza los acumuladores estadísticos promedio de tiempo
            update_time_avg_stats()
            # Invoca la función de evento adecuada
            if next_event_type == 1:
                order_arrival()
            elif next_event_type == 2:
                demand()
            elif next_event_type == 4:
                evaluate()
            elif next_event_type == 3:
                report()

def initialize():
    global sim_time, inv_level, time_last_event, total_ordering_cost, area_holding, area_shortage, time_next_event, next_event_type

    next_event_type = 0
    
    # Inicializa el reloj de la simulación
    sim_time = 0.0

    # Inicializa las variables de estado
    inv_level = initial_inv_level
    time_last_event = 0.0

    # Inicializa los contadores estadísticos
    total_ordering_cost = 0.0
    area_holding = 0.0
    area_shortage = 0.0

    # Inicializa la lista de eventos. Dado que no hay pedidos pendientes, el evento de llegada de pedidos se elimina de la consideración.
    time_next_event[1] = 1.0e+30
    time_next_event[2] = sim_time + expon(mean_interdemand)
    time_next_event[3] = num_months
    time_next_event[4] = 0.0

def timing():
    global next_event_type, sim_time
    
    min_time_next_event = 1.0e+29
    next_event_type = 0

    # Determino el tipo del proximo evento.
    for i in range(1, num_events+1):
        if time_next_event[i] < min_time_next_event:
            min_time_next_event = time_next_event[i]
            next_event_type = i

    # Me fijo si la lista de eventos esta vacia.
    if next_event_type == 0:
        # Lista vacia, termino la simulacion.
        print(f"\nLista de eventos vacia en {sim_time}")
        exit(1)

    # Lista no vacia, avanza el reloj de simulacion.
    sim_time = min_time_next_event

def order_arrival():
    global inv_level, time_next_event

    # Incrementa el nivel de inventario según la cantidad pedida
    inv_level += amount

    # Dado que no hay un pedido pendiente, elimina el evento de llegada de pedidos de la consideración
    time_next_event[1] = 1.0e+30

def demand():
    global inv_level, time_next_event

    # Decrementa el nivel de inventario según el tamaño de la demanda generado
    inv_level -= random_integer(prob_distrib_demand)

    # Programa el tiempo de la próxima demanda
    time_next_event[2] = sim_time + expon(mean_interdemand)

def evaluate():
    global inv_level, amount, total_ordering_cost

    # Verificar si el nivel de inventario es menor que "smalls".
    if inv_level < smalls:
        # El nivel de inventario es menor que "smalls", por lo tanto, realizar un pedido por la cantidad correspondiente.
        amount = bigs - inv_level
        total_ordering_cost += setup_cost + incremental_cost * amount
        # Programar la llegada del pedido.
        time_next_event[1] = sim_time + uniform(minlag, maxlag)

    # Independientemente de la decisión de realizar un pedido, programar la próxima evaluación del inventario.
    time_next_event[4] = sim_time + 1.0

def report():
    # Calcula y escribe las estimaciones de las medidas de rendimiento deseadas.
    avg_holding_cost = holding_cost * area_holding / num_months
    avg_ordering_cost = total_ordering_cost / num_months
    avg_shortage_cost = shortage_cost * area_shortage / num_months
    print(f"\n({smalls:3d},{bigs:3d}){avg_ordering_cost + avg_holding_cost + avg_shortage_cost:15.2f}{avg_ordering_cost:15.2f}{avg_holding_cost:15.2f}{avg_shortage_cost:15.2f}")

def update_time_avg_stats():
    global time_last_event, sim_time, area_shortage, area_holding
    
    # Calcula el tiempo transcurrido desde el último evento y actualiza el marcador de último evento.
    time_since_last_event = sim_time - time_last_event
    time_last_event = sim_time

    # Determina el estado del nivel de inventario durante el intervalo anterior.
    # Si el nivel de inventario durante el intervalo anterior fue negativo, actualiza area_shortage.
    # Si fue positivo, actualiza area_holding. Si fue cero, no se necesita actualización.
    if inv_level < 0:
        area_shortage -= inv_level * time_since_last_event
    elif inv_level > 0:
        area_holding += inv_level * time_since_last_event
        
if __name__ == "__main__":
    main()