import random
import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import log as ln, exp
import scipy.stats as stats
from sklearn.mixture import GaussianMixture

#Constantes
a_uniforme = 0
b_uniforme = 1
lamb_exponencial = 2
k_gamma = 2
a_gamma = 1
ex_normal=0
stdx_normal=1
k_pascal = 5
q_pascal = 0.5
p_pascal = 0.5
n_binomial = 10
p_binomial = 0.5
tn_hipergeometrica = 50
ns_hipergeometrica = 20
p_hipergeometrica = 0.5
m_hipergeometrica = 25
lambda_poisson = 3
muestra_empirica = [0.273, 0.037, 0.195, 0.009, 0.124, 0.058, 0.062, 0.151, 0.047, 0.044]
size = 1000 #Tamaño de las muestras

#Transformada inversa
#Uniforme continua
def uniforme(a, b):
    r = random.random()
    num = a + (b-a) * r
    return num
    
#Exponencial
def exponencial(lamb):
    r = random.random()
    num = -ln(1-r)/lamb
    return num

#Normal
def normal(ex, stdx):
    sum = 0
    for i in range(12):
        r = random.random()
        sum += r
    x = stdx * (sum - 6.0) + ex
    return x

#Gamma
def inv_gamma(k, a):
    tr = 1
    for _ in range(k):
        r = random.random()
        tr *= r
    x = -ln(tr)/a    
    return x

#Pascal
def pascal(k, q):
    tr = 1
    qr = ln(q)
    for _ in range(k):
        r = random.random()
        tr *= r
    x = ln(tr)/qr
    return x

#Binomial
def binomial(n, p):
    x = 0
    for _ in range(1, n):
        r = random.random()
        if (r-p) <= 0:
            x += 1
    return x

#Hipergeométrica
def hipergeometrica(tn, ns, p):
    x = 0
    for i in range(1, ns + 1):
        r = random.random()
        if r - p <= 0:
            s = 1
            x += 1
        else:
            s = 0
        p = (tn * p - s) / (tn - 1)
        tn -= 1
    return x

#Poisson
def inv_poisson(p):
    x = 0
    b = exp(-p)
    tr = 1
    while (tr - b) >= 0:
        r = random.random()
        tr *= r
        x += 1
    return x


#Empirica discreta
def empirica_discreta(muestra):
    r = random.random()
    sum = 0
    x = 1
    for valor in muestra:
        sum += valor
        if r <= sum :
            break
        else:
            x += 1
    return x

#Gráficos Transformada Inversa
#Distribución Uniforme
datos_uniforme = [uniforme(a_uniforme, b_uniforme) for _ in range(size)] #Genera 1000 valores usando la función de la distribución Uniforme
plt.hist(datos_uniforme, bins=30, density=True)
plt.xlabel('Valores')
plt.ylabel('Densidad')
plt.title('Distribución Uniforme Continua-Transformada Inversa')
plt.show()

#Distribución Exponencial
datos_exponencial = [exponencial(lamb_exponencial) for _ in range(size)] #Genera 1000 valores usando la función de la distribución Exponencial
plt.hist(datos_exponencial, bins=30, density=True)
plt.xlabel('Valores')
plt.ylabel('Densidad')
plt.title('Distribución Exponencial-Transformada Inversa')
plt.show()

#Distribución Normal
datos_normal = [normal(ex_normal, stdx_normal) for _ in range(size)] #Genera 1000 valores usando la función de la distribución Normal
plt.hist(datos_normal, bins=30, density=True)
plt.xlabel('Valores')
plt.ylabel('Densidad')
plt.title('Distribución Normal-Transformada Inversa')
plt.show()

#Distribución Gamma
datos_gamma = [inv_gamma(k_gamma, a_gamma) for _ in range(size)] #Genera 1000 valores usando la función de la distribución Gamma
plt.hist(datos_gamma, bins=30, density=True)
plt.xlabel('Valores')
plt.ylabel('Densidad')
plt.title('Distribución Gamma-Transformada Inversa')
plt.show()

#Distribución Binomial
datos_binomial = [binomial(n_binomial, p_binomial) for _ in range(size)] #Genera 1000 valores usando la función de la distribución Binomial
valores, frecuencias = np.unique(datos_binomial, return_counts=True)
plt.bar(valores, frecuencias, width = 0.1)
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Distribución Binomial-Transformada Inversa')
plt.show()

#Distribución de Poisson
datos_poisson = [inv_poisson(lambda_poisson) for _ in range (size)] #Genera 1000 valores usando la función de la distribución Poisson
valores, frecuencias = np.unique(datos_poisson, return_counts=True)
plt.bar(valores, frecuencias, width=0.1)
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Distribución de Poisson-Transformada Inversa')
plt.show()

# Distribución de Pascal
datos_pascal = [int(pascal(k_pascal, q_pascal)) for _ in range(size)] #Genera 1000 valores usando la función de la distribución Pascal
valores, frecuencias = np.unique(datos_pascal, return_counts=True)
plt.bar(valores, frecuencias, width=0.1)
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Distribución Pascal-Transformada Inversa')
plt.show()

# Distribución Hipergeométrica
datos_hipergeometrica = [hipergeometrica(tn_hipergeometrica, ns_hipergeometrica, p_hipergeometrica) for _ in range(size)] #Genera 1000 valores usando la función de la distribución Hipergeometrica
valores, frecuencias = np.unique(datos_hipergeometrica, return_counts=True)
plt.bar(valores, frecuencias, width=0.1)
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Distribución Hipergeométrica-Transformada Inversa')
plt.show()

#Empírica Discreta
muestra = [empirica_discreta(muestra_empirica) for _ in range(size)] #Genera 1000 valores usando la función de la distribución Discreta
valores, frecuencias = np.unique(muestra, return_counts=True)
plt.bar(valores, frecuencias, width=0.1)
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Distribución Empírica Discreta-Transformada Inversa')
plt.show()

#Método del rechazo
#Distribución Uniforme
def uniforme_rechazo(a, b):
    while True:
        r1 = random.uniform(0, 1)  #Genera un valor aleatorio en el rango [0, 1]
        X = a + (b - a) * r1  #Aplica la transformación para obtener X en el rango [a, b]
        c = 1 / (b - a)  #Calcula la constante de rechazo
        r2 = random.uniform(0, 1) #Genera otro valor aleatorio para comparar con el criterio de aceptación
        #Comprueba el criterio de aceptación/rechazo
        if r2 <= c:
            return X  #Devuelve el valor aceptado

#Distribución Exponencial
def exponencial_rechazo(lambd):
    while True:
        r1 = random.uniform(0, 1)  #Genera un valor aleatorio r1 en el rango [0, 1]
        r2 = random.uniform(0, 1)  #Genera un valor aleatorio r2 en el rango [0, 1]
        X = -math.log(r1) / lambd  #Aplica la transformación para obtener X según la distribución exponencial
        c = lambd  #Calcula la constante de rechazo
        #Comprueba el criterio de aceptación/rechazo
        if r2 <= math.exp(-c * X):
            return X  #Devuelve el valor aceptado
          
#Distribución Gamma
def gamma_rechazo(k, a):
    while True:
        r1 = exponencial_rechazo(a)  #Genera un valor de prueba utilizando una distribución exponencial
        r2 = random.uniform(0, 1) #Genera una variable aleatoria r2 en el rango [0, 1]
        c = (a ** k) / math.gamma(k) #Calcula la constante de rechazo
        X = r1 ** (k - 1) * math.exp(-r1) / c #Calcula el valor candidato
        #Comprueba el criterio de aceptación/rechazo
        if r2 <= X:
            return r1

#Distribución Normal
def normal_rechazo(mu, sigma):
    while True:
        #Genera dos valores aleatorios uniformes en el rango (0, 1]
        r1 = random.uniform(0, 1)
        r2 = random.uniform(0, 1)
        #Aplica la transformación de Box-Muller para obtener un valor aleatorio normalmente distribuido
        Z = math.sqrt(-2 * math.log(r1)) * math.cos(2 * math.pi * r2)
        X = Z * sigma + mu #Calcula el valor candidato
        c = math.exp(-0.5 * ((X - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi)) #Calcula la constante de rechazo
        r = random.uniform(0, 1)  #Genera un valor aleatorio uniforme en el rango (0, 1]
        #Comprueba el criterio de aceptación/rechazo
        if r <= c:
            return X

#Distribución Binomial
def binomial_rechazo(n, p):
    #Calcula la constante de rechazo
    c = math.ceil(n*(1-p)/(p**2))
    while True:
        #Genera dos números aleatorios r1 y r2 distribuidos uniformemente en [0,1)
        r1 = random.uniform(0, 1)
        r2 = random.uniform(0, 1)
        #Calcula la función de probabilidad para g
        g = math.floor(r1 * c) + 1
        prob = ((1-p)**(n-g)) * (p**g) * (math.comb(n, g))
        #Rechaza el número generado si r2 > prob
        if r2 <= prob:
            return g

#Distribución de Poisson
def poisson_rechazo(lam):
    c = math.exp(lam) + 1  #Calcula la constante de rechazo c
    while True:
        x = random.choices(range(int(c+1)))  #Genera una variable aleatoria discreta con distribución uniforme
        x = x[0]
        fx = math.exp(-lam) * math.pow(lam, x) / math.factorial(x)  #Calcula la función de probabilidad de la distribución de Poisson
        r = random.uniform(0, 1)  #Calcula la probabilidad de aceptación
        #Comprueba si se cumple la condición de rechazo
        if r <= fx / (c + 1):
            return x

#Distribución de Pascal
def pascal_rechazo(k, p):
    c = (1 - p) ** k  #Calcula la constante de rechazo 
    while True:
        x = random.randint(k, k+1000)  #Genera una variable aleatoria discreta con distribución uniforme
        fx = math.comb(k + x - 1, x) * (p ** k) * ((1 - p) ** (x)) #Calcula la función de probabilidad de la distribución de Pascal
        r = random.uniform(0, 1)  #Calcula la probabilidad de aceptación
        #Comprueba si se cumple la condición de rechazo
        if r <= fx / c:
            return x

#Distribución Hipergeométrica
def hipergeometrica_rechazo(N, m, n):
    x = 0 #Contador del número de éxitos
    for _ in range(n):
        p = m / N #Calcula la probabilidad de éxito
        r = random.random() #Genera un número aleatorio r en el rango [0,1)
        #Si r es menor que p se considera éxito
        if r < p:  
            x += 1 #Se incrementa el contador de éxitos
            m -= 1 #Se reduce en 1 el número de éxitos restantes m
        N -= 1 #Se reduce en 1 el tamaño de la población total N
    return x 

#Distribución Empírica Discreta
def empirica_discreta_rechazo(muestra_empirica):
    n = len(muestra_empirica)  #Calcula la longitud de la muestra
    p = np.ones(n) / n  #Probabilidades uniformes para cada valor en la muestra
    c = np.max(p)  #Constante de rechazo
    while True:
        x = random.choices(muestra_empirica)[0]  #Selecciona un valor al azar de la muestra
        fx = p[muestra_empirica.index(x)]  # robabilidad de x en la muestra
        r = random.uniform(0, c)  #Genera una probabilidad de aceptación
        #Si se cumple la condición, se devuelve el valor, si no, se genera otro número
        if r <= fx:
            return x
        
#Gráficos del método de rechazo
#Distribución Uniforme
muestras_uniforme = [uniforme_rechazo(a_uniforme, b_uniforme) for _ in range(1000)]
plt.hist(muestras_uniforme, bins=30, density=True)
plt.xlabel('Valores')
plt.ylabel('Densidad')
plt.title('Distribución Uniforme Continua-Método de Rechazo')
plt.show()

#Distribución Exponencial
muestras_exponencial = [exponencial_rechazo(lamb_exponencial) for _ in range(1000)]
plt.hist(muestras_exponencial, bins=30, density=True)
plt.xlabel('Valores')
plt.ylabel('Densidad')
plt.title('Distribución Exponencial-Método de Rechazo')
plt.show()

#Distribución Gamma
muestras_gamma = [gamma_rechazo(k_gamma, a_gamma) for _ in range(1000)]
plt.hist(muestras_gamma, bins=30, density=True)
plt.xlabel('Valores')
plt.ylabel('Densidad')
plt.title('Distribución Gamma-Método de Rechazo')
plt.show()

#Distribución Normal
muestras_normal = [normal_rechazo(ex_normal, stdx_normal) for _ in range(1000)]
plt.hist(muestras_normal, bins=30, density=True)
plt.xlabel('Valores')
plt.ylabel('Densidad')
plt.title('Distribución Normal-Método de Rechazo')
plt.show()

#Distribución Binomial
muestras_binomial = [binomial_rechazo(10, 0.5) for _ in range(1000)]
valores_unicos = np.unique(muestras_binomial)
frecuencias = [muestras_binomial.count(valor) for valor in valores_unicos]
plt.bar(valores_unicos, frecuencias, width=0.1)
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Distribución Binomial-Método de Rechazo')
plt.show()
print(valores_unicos)
print(frecuencias)

#Distribución de Poisson
muestras_poisson = [poisson_rechazo(lambda_poisson) for _ in range(1000)]
valores_unicos = np.unique(muestras_poisson)
frecuencias = [muestras_poisson.count(valor) for valor in valores_unicos]
plt.bar(valores_unicos, frecuencias, width=0.1)
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Distribución de Poisson-Método de Rechazo')
plt.show()

#Distribución Pascal
muestras_pascal = [pascal_rechazo(k_pascal, p_pascal) for _ in range(1000)]
valores_unicos = np.unique(muestras_pascal)
frecuencias = [muestras_pascal.count(valor) for valor in valores_unicos]
plt.bar(valores_unicos, frecuencias, width=0.1)
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Distribución Pascal-Método de Rechazo')
plt.show()

#Distribución Hipergeométrica
muestras_hipergeometrica = [hipergeometrica_rechazo(tn_hipergeometrica, m_hipergeometrica, ns_hipergeometrica) for _ in range(1000)]
valores_unicos = np.unique(muestras_hipergeometrica)
frecuencias = [muestras_hipergeometrica.count(valor) for valor in valores_unicos]
plt.bar(valores_unicos, frecuencias, width=0.1)
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Distribución Hipergeométrica-Método de Rechazo')
plt.show()

#Distribución Empírica Discreta
muestras_empirica = [empirica_discreta_rechazo(muestra_empirica) for _ in range(1000)]
valores_unicos = np.unique(muestras_empirica)
frecuencias = [muestras_empirica.count(valor) for valor in valores_unicos]
plt.bar(valores_unicos, frecuencias, width=0.1)
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Distribución Empírica Discreta-Método de Rechazo')
plt.show()

#Test
#Generar muestras y realizar comparación para cada distribución
#Uniforme continua
data_uniforme = np.array([uniforme(a_uniforme, b_uniforme) for _ in range(size)])
media_uniforme = (a_uniforme+b_uniforme)/2
varianza_uniforme = ((b_uniforme-a_uniforme)**2)/12
print("Uniforme continua:")
print("Media teórica:", media_uniforme)
print("Media muestral:", np.mean(data_uniforme))
print("Varianza teórica:", varianza_uniforme)
print("Varianza muestral:", np.var(data_uniforme))
print()

#Exponencial
data_exponencial = np.array([exponencial(lamb_exponencial) for _ in range(size)])
media_exponencial = 1 / lamb_exponencial
varianza_exponencial = 1 / (lamb_exponencial ** 2)
print("Exponencial:")
print("Media teórica:", media_exponencial)
print("Media muestral:", np.mean(data_exponencial))
print("Varianza teórica:", varianza_exponencial)
print("Varianza muestral:", np.var(data_exponencial))
print()

#Gamma 
data_gamma_scipy = stats.gamma.rvs(k_gamma, scale=1/a_gamma, size=size)
media_gamma_scipy = k_gamma * a_gamma
varianza_gamma_scipy = k_gamma * (a_gamma ** 2)
print("Gamma:")
print("Media teórica:", media_gamma_scipy)
print("Media muestral:", np.mean(data_gamma_scipy))
print("Varianza teórica:", varianza_gamma_scipy)
print("Varianza muestral:", np.var(data_gamma_scipy))
print()

#Normal
data_normal = np.array([normal(ex_normal, stdx_normal) for _ in range(size)])
media_normal = ex_normal
varianza_normal = stdx_normal**2
print("Normal:")
print("Media teórica:", media_normal)
print("Media muestral:", np.mean(data_normal))
print("Varianza teórica:", varianza_normal)
print("Varianza muestral:", np.var(data_normal))
print()

#Pascal
data_pascal = np.array([pascal(k_pascal, q_pascal) for _ in range(size)])
media_pascal = k_pascal * (1 / q_pascal)
varianza_pascal = k_pascal * (1 - q_pascal) / (q_pascal ** 2)
print("Pascal:")
print("Media teórica:", media_pascal)
print("Media muestral:", np.mean(data_pascal))
print("Varianza teórica:", varianza_pascal)
print("Varianza muestral:", np.var(data_pascal))
print()

#Binomial
data_binomial = np.array([binomial(n_binomial, p_binomial) for _ in range(size)])
media_binomial = n_binomial * p_binomial
varianza_binomial = n_binomial * p_binomial * (1 - p_binomial)
print("Binomial:")
print("Media teórica:", media_binomial)
print("Media muestral:", np.mean(data_binomial))
print("Varianza teórica:", varianza_binomial)
print("Varianza muestral:", np.var(data_binomial))
print()

#Hipergeométrica
data_hipergeometrica = np.array([hipergeometrica(tn_hipergeometrica, ns_hipergeometrica, p_hipergeometrica) for _ in range(size)])
media_hipergeometrica = ns_hipergeometrica * (p_hipergeometrica / tn_hipergeometrica)
varianza_hipergeometrica = (ns_hipergeometrica * p_hipergeometrica * (tn_hipergeometrica - ns_hipergeometrica) * (tn_hipergeometrica - p_hipergeometrica)) / (tn_hipergeometrica ** 2 * (tn_hipergeometrica - 1))
print("Hipergeométrica:")
print("Media teórica:", media_hipergeometrica)
print("Media muestral:", np.mean(data_hipergeometrica))
print("Varianza teórica:", varianza_hipergeometrica)
print("Varianza muestral:", np.var(data_hipergeometrica))
print()

#Poisson
data_poisson = np.random.poisson(lambda_poisson, size)
media_muestral = np.mean(data_poisson)
varianza_muestral = np.var(data_poisson)
media_teorica = lambda_poisson
varianza_teorica = lambda_poisson
print("Poisson:")
print("Media teórica:", media_teorica)
print("Media muestral:", media_muestral)
print("Varianza teórica:", varianza_teorica)
print("Varianza muestral:", varianza_muestral)
print()

#Empírica discreta
data_empirica_discreta = np.array([empirica_discreta(muestra_empirica) for _ in range(size)])
media_empirica_discreta = np.mean(muestra_empirica)
varianza_empirica_discreta = np.var(muestra_empirica)
print("Empírica discreta:")
print("Media teórica:", media_empirica_discreta)
print("Media muestral:", np.mean(data_empirica_discreta))
print("Varianza teórica:", varianza_empirica_discreta)
print("Varianza muestral:", np.var(data_empirica_discreta))
print()