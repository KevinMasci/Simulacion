import random
import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import log as ln, exp
import scipy.stats as stats
from sklearn.mixture import GaussianMixture

#Constantes
muestra_empirica = [0.273, 0.037, 0.195, 0.009, 0.124, 0.058, 0.062, 0.151, 0.047, 0.044]

# Transformada inversa
# Uniforme continua
def uniforme(a, b):
    r = random.random()
    num = a + (b-a) * r
    return num
    
# Exponencial
def exponencial(lamb):
    r = random.random()
    num = -ln(1-r)/lamb
    return num

# Normal
def normal(ex, stdx):
    sum = 0
    for i in range(12):
        r = random.random()
        sum += r
    x = stdx * (sum - 6.0) + ex
    return x

# Gamma
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

# Distribución Uniforme
datos_uniforme = [uniforme(0, 1) for _ in range(1000)]
plt.hist(datos_uniforme, bins=30, density=True)
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Distribución Uniforme Continua')
plt.show()

# Distribución Exponencial
datos_exponencial = [exponencial(2) for _ in range(1000)]
plt.hist(datos_exponencial, bins=30, density=True)
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Distribución Exponencial')
plt.show()

# Distribución Normal
datos_exponencial = [normal(0, 1) for _ in range(1000)]
plt.hist(datos_exponencial, bins=30, density=True)
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Distribución Normal')
plt.show()

#Distribución Gamma
datos_gamma = [inv_gamma(2, 1) for _ in range(1000)]
plt.hist(datos_gamma, bins=30, density=True)
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Distribución Gamma')
plt.show()

# Distribución Binomial
datos_binomial = [binomial(10, 0.5) for _ in range(1000)]
valores, frecuencias = np.unique(datos_binomial, return_counts=True)
plt.bar(valores, frecuencias, width = 0.5)
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Distribución Binomial')
plt.show()

# Distribución de Poisson
datos_poisson = [inv_poisson(3) for _ in range (1000)]
valores, frecuencias = np.unique(datos_poisson, return_counts=True)
plt.bar(valores, frecuencias, width=0.5)
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Distribución de Poisson')
plt.show()

# Distribución de Pascal
datos_pascal = [int(pascal(5, 0.3)) for _ in range(1000)]
valores, frecuencias = np.unique(datos_pascal, return_counts=True)
plt.bar(valores, frecuencias, width=0.5)
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Distribución Pascal')
plt.show()

# Distribución Hipergeométrica
datos_hipergeometrica = [hipergeometrica(50, 20, 0.5) for _ in range(1000)]
valores, frecuencias = np.unique(datos_hipergeometrica, return_counts=True)
plt.bar(valores, frecuencias, width=0.5)
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Distribución Hipergeométrica')
plt.show()

#Empírica Discreta
muestra = [empirica_discreta(muestra_empirica) for _ in range(1000)]
valores, frecuencias = np.unique(muestra, return_counts=True)
plt.bar(valores, frecuencias, width=0.5)
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Distribución Empírica Discreta')
plt.show()

# Metodo del rechazo
#Continuas
def uniforme_rechazo(a, b):
    while True:
        r1 = random.uniform(0, 1)  # Generar un valor aleatorio U en el rango [0, 1]
        X = a + (b - a) * r1  # Aplicar la transformación para obtener X en el rango [a, b]
        c = 1 / (b - a)  # Calcular la constante de rechazo
        r2 = random.uniform(0, 1) # Generar otro valor aleatorio para comparar con el criterio de aceptación
        # Comprobar el criterio de aceptación/rechazo
        if r2 <= c:
            return X  # Devolver el valor aceptado

def exponencial_rechazo(lambd):
    while True:
        r1 = random.uniform(0, 1)  # Generar un valor aleatorio r1 en el rango [0, 1]
        r2 = random.uniform(0, 1)  # Generar un valor aleatorio r2 en el rango [0, 1]
        X = -math.log(r1) / lambd  # Aplicar la transformación para obtener X según la distribución exponencial
        c = lambd  # Calcular la constante de rechazo
        # Comprobar el criterio de aceptación/rechazo
        if r2 <= math.exp(-c * X):
            return X  # Devolver el valor aceptado
          
def gamma_rechazo(alpha, beta):
    while True:
        r1 = exponencial_rechazo(beta)   # Generar un valor de prueba utilizando una distribución exponencial
        r2 = random.uniform(0, 1) # Generar una variable aleatoria U en el rango [0, 1]
        c = (beta ** alpha) / math.gamma(alpha) # Calcular la constante de rechazo
        X = r1 ** (alpha - 1) * math.exp(-r1) / c # Calcular el valor candidato
        # Comprobar el criterio de aceptación/rechazo
        if r2 <= X:
            return r1

def normal_rechazo(mu, sigma):
    while True:
        # Generar dos valores aleatorios uniformes en el rango (0, 1]
        r1 = random.uniform(0, 1)
        r2 = random.uniform(0, 1)
        # Aplicar la transformación de Box-Muller para obtener un valor aleatorio normalmente distribuido
        Z = math.sqrt(-2 * math.log(r1)) * math.cos(2 * math.pi * r2)
        X = Z * sigma + mu # Calcular el valor candidato
        c = math.exp(-0.5 * ((X - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi)) # Calcular la constante de rechazo
        r = random.uniform(0, 1)  # Generar un valor aleatorio uniforme en el rango (0, 1]
        # Comprobar el criterio de aceptación/rechazo
        if r <= c:
            return X

#Discretas
def binomial_rechazo(n, p):
    # Se define una constante c tal que c >= n*(1-p)/p**2
    c = math.ceil(n*(1-p)/(p**2))
    while True:
        # Se generan dos números aleatorios r1 y r2 distribuidos uniformemente en [0,1)
        r1 = random.uniform(0, 1)
        r2 = random.uniform(0, 1)
        # Se calcula la función de probabilidad para g
        g = math.floor(r1 * c) + 1
        prob = (1-p)**(n-g) * p**g * math.comb(n, g)
        # Se rechaza el número generado si V > prob
        if r2 <= prob:
            return g

def poisson_rechazo(lam):
    c = math.exp(lam) + 1  # Calcula la constante de rechazo c
    while True:
        x = random.choices(range(int(c+1)))  # Genera una variable aleatoria discreta con distribución uniforme
        x = x[0]
        fx = math.exp(-lam) * math.pow(lam, x) / math.factorial(x)  # Calcula la función de probabilidad de la distribución de Poisson
        r = random.uniform(0, 1)  # Calcula la probabilidad de aceptación
        # Comprueba si se cumple la condición de rechazo
        if r <= fx / (c + 1):
            return x

def pascal_rechazo(k, p):
    c = int(p / (1 - p)) ** k  # Calcula la constante de rechazo c como entero
    while True:
        x = random.randint(0, k)  # Genera una variable aleatoria discreta con distribución uniforme
        fx = math.comb(x + k - 1, k - 1) * p ** k * (1 - p) ** x  # Calcula la función de probabilidad de la distribución de Pascal
        r = random.uniform(0, 1)  # Calcula la probabilidad de aceptación
        # Comprueba si se cumple la condición de rechazo
        if r <= fx / c:
            return x

def hipergeometrica_rechazo(N, m, n):
    x = 0
    for _ in range(n):
        p = m / N
        r = random.random()
        if r < p:
            x += 1
            m -= 1
        N -= 1
    return x

    
def empirica_discreta_rechazo(muestra_empirica):
    n = len(muestra_empirica)
    p = np.ones(n) / n  # Probabilidades uniformes para cada valor en la muestra
    c = np.max(p)  # Constante de rechazo, máximo de las probabilidades
    while True:
        x = random.choices(muestra_empirica)[0]  # Generar un valor aleatorio de la muestra
        fx = p[muestra_empirica.index(x)]  # Probabilidad de x en la muestra
        r = random.uniform(0, c)  # Generar una probabilidad de aceptación
        if r <= fx:
            return x
        
#Gráficos del método de rechazo
# Distribución Uniforme
a = 0
b = 1
muestras_uniforme = [uniforme_rechazo(a, b) for _ in range(1000)]
plt.hist(muestras_uniforme, bins=30, density=True)
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Distribución Uniforme')
plt.grid(True)
plt.show()

# Distribución Exponencial
lambd = 2
muestras_exponencial = [exponencial_rechazo(lambd) for _ in range(1000)]
plt.hist(muestras_exponencial, bins=30, density=True)
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Distribución Exponencial')
plt.grid(True)
plt.show()

# Distribución Gamma
alpha = 2
beta = 1
muestras_gamma = [gamma_rechazo(alpha, beta) for _ in range(1000)]
plt.hist(muestras_gamma, bins=30, density=True)
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Distribución Gamma')
plt.grid(True)
plt.show()

# Distribución Normal
mu = 0
sigma = 1
muestras_normal = [normal_rechazo(mu, sigma) for _ in range(1000)]
plt.hist(muestras_normal, bins=30, density=True)
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Distribución Normal')
plt.grid(True)
plt.show()

#Gráficos 
# Distribución Binomial
n = 10
p = 0.5
muestras_binomial = [binomial_rechazo(n, p) for _ in range(1000)]
valores_unicos = np.unique(muestras_binomial)
frecuencias = [muestras_binomial.count(valor) for valor in valores_unicos]
plt.bar(valores_unicos, frecuencias)
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Distribución Binomial')
plt.grid(True)
plt.show()

# Distribución de Poisson
lam = 3
muestras_poisson = [poisson_rechazo(lam) for _ in range(1000)]
valores_unicos = np.unique(muestras_poisson)
frecuencias = [muestras_poisson.count(valor) for valor in valores_unicos]
plt.bar(valores_unicos, frecuencias)
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Distribución de Poisson')
plt.grid(True)
plt.show()

# Distribución Pascal
k = 5
p = 0.5
muestras_pascal = [pascal_rechazo(k, p) for _ in range(1000)]
valores_unicos = np.unique(muestras_pascal)
frecuencias = [muestras_pascal.count(valor) for valor in valores_unicos]
plt.bar(valores_unicos, frecuencias, width=0.2)
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Distribución Pascal')
plt.grid(True)
plt.show()

# Distribución Hipergeométrica
N = 1000
m = 500
n = 200
muestras_hipergeometrica = [hipergeometrica_rechazo(N, m, n) for _ in range(1000)]
valores_unicos = np.unique(muestras_hipergeometrica)
frecuencias = [muestras_hipergeometrica.count(valor) for valor in valores_unicos]
plt.bar(valores_unicos, frecuencias)
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Distribución Hipergeométrica')
plt.grid(True)
plt.show()

# Distribución Empírica Discreta
muestra_empirica = [1, 3, 5, 7, 9]
muestras_empirica = [empirica_discreta_rechazo(muestra_empirica) for _ in range(1000)]
valores_unicos = np.unique(muestras_empirica)
frecuencias = [muestras_empirica.count(valor) for valor in valores_unicos]
plt.bar(valores_unicos, frecuencias)
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Distribución Empírica Discreta')
plt.grid(True)
plt.show()

# Test
# Parámetros para las distribuciones
a_uniforme = 0
b_uniforme = 1
lamb_exponencial = 2
k_gamma = 2
a_gamma = 1.5
k_pascal = 5
q_pascal = 0.3
n_binomial = 10
p_binomial = 0.4
tn_hipergeometrica = 20
ns_hipergeometrica = 7
p_hipergeometrica = 0.4
lambda_poisson = 3
size = 1000
muestra_empirica_discreta = [1, 2, 3, 4, 5]

# Tamaño de las muestras
size = 1000

# Generar muestras y realizar comparación para cada distribución

# Uniforme continua
data_uniforme = np.array([uniforme(a_uniforme, b_uniforme) for _ in range(size)])
media_uniforme = (a_uniforme + b_uniforme) / 2
varianza_uniforme = ((b_uniforme - a_uniforme) ** 2) / 12

print("Uniforme continua:")
print("Media teórica:", media_uniforme)
print("Media muestral:", np.mean(data_uniforme))
print("Varianza teórica:", varianza_uniforme)
print("Varianza muestral:", np.var(data_uniforme))
print()

# Exponencial
data_exponencial = np.array([exponencial(lamb_exponencial) for _ in range(size)])
media_exponencial = 1 / lamb_exponencial
varianza_exponencial = 1 / (lamb_exponencial ** 2)

print("Exponencial:")
print("Media teórica:", media_exponencial)
print("Media muestral:", np.mean(data_exponencial))
print("Varianza teórica:", varianza_exponencial)
print("Varianza muestral:", np.var(data_exponencial))
print()

# Gamma (scipy)
data_gamma_scipy = stats.gamma.rvs(k_gamma, scale=1/a_gamma, size=size)
media_gamma_scipy = k_gamma * a_gamma
varianza_gamma_scipy = k_gamma * (a_gamma ** 2)

print("Gamma (scipy):")
print("Media teórica:", media_gamma_scipy)
print("Media muestral:", np.mean(data_gamma_scipy))
print("Varianza teórica:", varianza_gamma_scipy)
print("Varianza muestral:", np.var(data_gamma_scipy))
print()

# Pascal
data_pascal = np.array([pascal(k_pascal, q_pascal) for _ in range(size)])
media_pascal = k_pascal * (1 / q_pascal)
varianza_pascal = k_pascal * (1 - q_pascal) / (q_pascal ** 2)

print("Pascal:")
print("Media teórica:", media_pascal)
print("Media muestral:", np.mean(data_pascal))
print("Varianza teórica:", varianza_pascal)
print("Varianza muestral:", np.var(data_pascal))
print()

# Binomial
data_binomial = np.array([binomial(n_binomial, p_binomial) for _ in range(size)])
media_binomial = n_binomial * p_binomial
varianza_binomial = n_binomial * p_binomial * (1 - p_binomial)

print("Binomial:")
print("Media teórica:", media_binomial)
print("Media muestral:", np.mean(data_binomial))
print("Varianza teórica:", varianza_binomial)
print("Varianza muestral:", np.var(data_binomial))
print()

# Hipergeométrica
data_hipergeometrica = np.array([hipergeometrica(tn_hipergeometrica, ns_hipergeometrica, p_hipergeometrica) for _ in range(size)])
media_hipergeometrica = ns_hipergeometrica * (p_hipergeometrica / tn_hipergeometrica)
varianza_hipergeometrica = (ns_hipergeometrica * p_hipergeometrica * (tn_hipergeometrica - ns_hipergeometrica) * (tn_hipergeometrica - p_hipergeometrica)) / (tn_hipergeometrica ** 2 * (tn_hipergeometrica - 1))

print("Hipergeométrica:")
print("Media teórica:", media_hipergeometrica)
print("Media muestral:", np.mean(data_hipergeometrica))
print("Varianza teórica:", varianza_hipergeometrica)
print("Varianza muestral:", np.var(data_hipergeometrica))
print()

# Poisson
data_poisson = np.random.poisson(lambda_poisson, size)
media_muestral = np.mean(data_poisson)
varianza_muestral = np.var(data_poisson)
media_teorica = lambda_poisson
varianza_teorica = lambda_poisson

print("Distribución de Poisson:")
print("Media teórica:", media_teorica)
print("Media muestral:", media_muestral)
print("Varianza teórica:", varianza_teorica)
print("Varianza muestral:", varianza_muestral)
print()

# Empírica discreta
data_empirica_discreta = np.array([empirica_discreta(muestra_empirica_discreta) for _ in range(size)])
media_empirica_discreta = np.mean(muestra_empirica_discreta)
varianza_empirica_discreta = np.var(muestra_empirica_discreta)

print("Empírica discreta:")
print("Media teórica:", media_empirica_discreta)
print("Media muestral:", np.mean(data_empirica_discreta))
print("Varianza teórica:", varianza_empirica_discreta)
print("Varianza muestral:", np.var(data_empirica_discreta))
print()

# Normal (scikit-learn)
gmm = GaussianMixture(n_components=1, covariance_type='full')
gmm.fit(data_exponencial.reshape(-1, 1))
data_normal_sklearn = gmm.sample(size)[0].flatten()
media_normal_sklearn = np.mean(data_exponencial)
varianza_normal_sklearn = np.var(data_exponencial)

print("Normal (scikit-learn):")
print("Media teórica:", media_normal_sklearn)
print("Media muestral:", np.mean(data_normal_sklearn))
print("Varianza teórica:", varianza_normal_sklearn)
print("Varianza muestral:", np.var(data_normal_sklearn))