import random
from numpy import log as ln

Q_LIMIT = 100
BUSY = 1
IDLE = 0

# Variables globales
next_event_type = None
num_custs_delayed = 0
num_delays_required = 0
num_events = 0
num_in_q = 0 # Q(t)
server_status = IDLE # B(t)
area_num_in_q = 0.0 # Area bajo Q(t)
area_server_status = 0.0 # Area bajo B(t)
mean_interarrival = 0.0
mean_service = 0.0
sim_time = 0.0
time_arrival = [0.0] * (Q_LIMIT + 1)
time_last_event  = 0.0
time_next_event = [0.0] * 3
total_of_delays = 0.0
infile = None
outfile = None

def expon(mean):
    return -float(mean) * ln(random.random())

def initialize():
    global sim_time, server_status, num_in_q, time_last_event, total_of_delays, num_custs_delayed, area_num_in_q, area_server_status, time_next_event
    
    # Reloj de simulacion
    sim_time = 0
    
    # Variables de estado
    server_status = IDLE
    num_in_q = 0
    time_last_event = 0.0
    
    # Contadores estadisticos
    num_custs_delayed = 0
    total_of_delays = 0.0
    area_num_in_q = 0.0
    area_server_status = 0.0
    
    # Lista de eventos inicial
    time_next_event[1] = sim_time + expon(mean_interarrival)
    time_next_event[2] = 1.0e+30
    
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

def arrive():
    global num_in_q, server_status, num_custs_delayed, total_of_delays, sim_time
    
    # Calculo el proximo arrivo.
    time_next_event[1] = sim_time + expon(mean_interarrival)
    
    # Me fijo si el servidor esta ocupado.
    if server_status == BUSY:
        # Servidor ocupado, incremento el numero de clientes en cola.
        num_in_q += 1
        
        # Me fijo si existe condicion de desbordado (?).
        if num_in_q > Q_LIMIT:
            # La cola esta desbordada, termino la simulacion
            print(f"\nDesbordamiento del arreglo array time_arrival en el tiempo {sim_time}")
            exit(2)

        # Todavia hay espacio en la cola, guardo el tiempo de arrivo al final de time_arrival
        time_arrival[num_in_q] = sim_time
    else:
        # Servidor desocupado, el cliente tiene demora 0. Lo siguiente es solo para comprension del programa y no afecta la simulacion
        delay = 0.0
        total_of_delays += delay
        
        # Incremento el numero de clientes demorados y pongo el servidor ocupado.
        num_custs_delayed += 1
        server_status = BUSY
        
        # Calculo salida del cliente
        time_next_event[2] = sim_time + expon(mean_service)
    
def depart():
    global num_in_q, server_status, time_next_event, total_of_delays, num_custs_delayed, time_arrival, sim_time
    
    # Me fijo si la cola esta vacia
    if num_in_q == 0:
        # Cola vacia, pongo el servidor desocupado y no tengo en concideracion el evento de partida.
        server_status = IDLE
        time_next_event[2] = 1.0e+30
    else:
        # Cola no vacia, disminuyo el numero de clientes en cola.
        num_in_q -= 1
        
        # Calculo la demora del cliente que entra en servicio y actualizo el acum de demora total.
        delay = sim_time - time_arrival[1]
        total_of_delays += delay
        
        # Incremento el numero de clientes demorados y calculo salida.
        num_custs_delayed += 1
        time_next_event[2] = sim_time + expon(mean_service)
        
        # Muevo los clientes en cola 1 lugar adelante
        for i in range(1, num_in_q+1):
            time_arrival[i] = time_arrival[i+1]
            
def report():
    # Calcula y escribe estimados de medidas de performance
    
    # Promedio de demora en cola
    average_delay_in_queue = total_of_delays / num_custs_delayed
    outfile.write(f"\nPromedio de demora en cola: {average_delay_in_queue:.3f} minutos")
    
    # Promedio de numero de clientes en cola
    average_num_in_queue = area_num_in_q / sim_time
    outfile.write(f"\nPromedio de numero de clientes en cola: {average_num_in_queue:.3f}")
    
    # Utilizacion del servidor
    server_utilization = area_server_status / sim_time
    outfile.write(f"\nUtilizacion del servidor: {server_utilization:.3f}")
    
    # Tiempo de fin de simulacion
    outfile.write(f"\nTiempo de fin de simulacion: {sim_time:.3f} minutos")
    
def update_time_avg_stats():
    global area_num_in_q, time_last_event, sim_time, num_in_q, area_server_status
    
    # Actualiza acumuladores de area para contadores estadisticos.
    
    # Calculo tiempo desde el ultimo evento y actualizo el marcador de time_last_event
    time_since_last_event = sim_time - time_last_event
    time_last_event = sim_time
    
    # Actualizo area bajo Q(t) (num_in_q)
    area_num_in_q += num_in_q * time_since_last_event
    
    # Actualizo el area bajo B(t) (server_status)
    area_server_status += server_status * time_since_last_event
            
def main():
    global infile, outfile, num_events, mean_interarrival, mean_service, num_delays_required
    # Abro archivos de lectura y escritura
    infile = open("TP3\mm1.in", "r")
    outfile = open("TP3\mm1.out", "w")
    
    num_events = 2
    
    # Leo parametros de ingreso
    mean_interarrival, mean_service, num_delays_required = infile.read().split()
    
    # Imprimo los parametros
    outfile.write("Sistema de cola de un solo servidor\n\n")
    outfile.write(f"Media de tiempo entre arrivos: {mean_interarrival} minutes\n\n")
    outfile.write(f"Media del tiempo de servicio: {mean_service} minutes\n\n")
    outfile.write(f"Numero de clientes: {num_delays_required}\n\n")
    
    # Llamo a rutina de inicializacion
    initialize()
    
    # Corro la simulacion hasta llegar al limite de Demorados
    while (int(num_custs_delayed) < int(num_delays_required)):
        # Determino el siguiente evento
        timing()
        # Update time-average statistical accumulators.
        update_time_avg_stats()
        # Invoco la rutina de eventos
        if next_event_type == 1:
            arrive()
        elif next_event_type == 2:
            depart()
    
    # Invoco el generador de reporte, cierro los archivos y termino la simulacion
    report()
    infile.close()
    outfile.close()
    
if __name__ == "__main__":
    main()