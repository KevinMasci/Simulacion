import math
import numpy as np
import random
from PIL import Image
from scipy.stats import chi2

n = 250000
seed = 9731
a = 1103515245
c = 12345
m = 2**31

# -----------------Generadores-----------------

#Generador GCL
def gcl(a, c, m, seed, n):
    nums= []
    num_actual = seed
    for i in range(n):
        num_nuevo = num_actual / m
        nums.append(num_nuevo)
        num_actual = (a * num_actual + c) % m
    return nums

#Generador randu
def randu(seed, n):
    nums = []
    for i in range(n):
        seed = (65539 * seed) % (2**31)
        nums.append(seed / (2**31))
    return nums

#Generador mid-square
def mid_square(seed, n):
    nums = []
    for i in range(n):
        seed = seed ** 2
        seed = str(seed)
        if len(seed) < 8:
            seed = '0' * (8 - len(seed)) + seed
        seed = seed[2:6]
        seed = int(seed)
        nums.append(seed/10**4)
    return nums

# --------------------Tests----------------------

def monobit_test(bits):
    num_ones = sum(bits)
    num_zeros = len(bits) - num_ones
    S = (num_ones - num_zeros) / math.sqrt(len(bits))
    p_value = math.erfc(abs(S) / math.sqrt(2))
    return p_value

def block_test(bits, block_size):
    num_blocks = len(bits) // block_size
    proportions = [sum(bits[i*block_size:(i+1)*block_size]) / block_size for i in range(num_blocks)]
    expected_proportion = 0.5
    chi2 = block_size * sum([(p - expected_proportion)**2 for p in proportions])
    critical_value = 3.841  # Valor crítico para un nivel de significancia de 0.05 (2 grados de libertad)
    return chi2 <= critical_value

def runs_test(bits):
    runs = [bits[0]]
    for i in range(1, len(bits)):
        if bits[i] != bits[i-1]:
            runs.append(bits[i])
    num_runs = len(runs)
    expected_runs = (2 * len(bits) - 1) / 3
    if abs(num_runs - expected_runs) > expected_runs:
        is_random = False
    else:
        is_random = True
    return is_random

def chisq_test(numbers):
    n = len(numbers)
    obs_freq, _ = np.histogram(numbers, bins=10, range=(0,1))
    exp_freq = np.repeat(n/10, 10)
    chi_sq = np.sum((obs_freq - exp_freq)**2 / exp_freq)
    p_value = 1 - chi2.cdf(x=chi_sq, df=9)
    return p_value

# ----------------Generadores de bits----------------

def generate_bits_gcl(seed, n):
    nums = gcl(a, c, m, seed, n)
    bits = [1 if num >= 0.5 else 0 for num in nums]
    return bits

def generate_bits_randu(seed, n):
    rand_nums = randu(seed, n)
    bits = []
    bits = [1 if num >= 0.5 else 0 for num in rand_nums]
    return bits

def generate_bits_mid_square(seed, n):
    nums = mid_square(seed, n)
    bits = []
    bits = [1 if num >= 0.5 else 0 for num in nums]
    return bits

def generate_bits_pyrandom(nums):
    bits = []
    bits = [1 if num >= 0.5 else 0 for num in nums]
    return bits

# ---------------------Testeo----------------------

numeros = [random.uniform(0, 1) for _ in range(250000)]
bits_gcl = generate_bits_gcl(seed, n)
bits_randu = generate_bits_randu(seed,n)
bits_mid_square = generate_bits_mid_square(seed,n)
bits_pyrandom = generate_bits_pyrandom(numeros)

#GCL
#Monobit
resultado_monobit = monobit_test(bits_gcl)
print(resultado_monobit)

#Block Test
resultado_block_test = block_test(bits_gcl,10)
print(resultado_block_test)

#Runs test
resultado_runs_test = runs_test(bits_gcl)
print(resultado_runs_test)

#Chi square test
resultado_chisq_test = chisq_test(gcl(a, c, m, seed, n))
print(resultado_chisq_test)

#Randu
#Monobit
resultado_monobit = monobit_test(bits_randu)
print(resultado_monobit)

#Block Test
resultado_block_test = block_test(bits_randu,10)
print(resultado_block_test)

#Runs test
resultado_runs_test = runs_test(bits_randu)
print(resultado_runs_test)

#Chi square test
resultado_chisq_test = chisq_test(randu(seed, n))
print(resultado_chisq_test)

#Mid-square
#Monobit
resultado_monobit = monobit_test(bits_mid_square)
print(resultado_monobit)

#Block Test
resultado_block_test = block_test(bits_mid_square,10)
print(resultado_block_test)

#Runs test
resultado_runs_test = runs_test(bits_mid_square)
print(resultado_runs_test)

#Chi square test
resultado_chisq_test = chisq_test(mid_square(seed, n))
print(resultado_chisq_test)

# Random Python
#Monobit
resultado_monobit = monobit_test(bits_pyrandom)
print(resultado_monobit)

#Block Test
resultado_block_test = block_test(bits_pyrandom,10)
print(resultado_block_test)

#Runs test
resultado_runs_test = runs_test(bits_pyrandom)
print(resultado_runs_test)

#Chi square test
resultado_chisq_test = chisq_test(numeros)
print(resultado_chisq_test)

# ----------------Graficos---------------

#GCL
width = 256
height = 256
values = gcl(a, c, m, seed, width * height)
img = Image.new('1', (width, height))
pixels = img.load()
for y in range(height):
    for x in range(width):
        pixel= values[y * width + x]
        img.putpixel((x, y), int(pixel * 2))
img.show()

#Randu
width = 256
height = 256
values = randu(seed, width * height)
img = Image.new('1', (width, height))
for y in range(height):
    for x in range(width):
        pixel = values[y * width + x]
        img.putpixel((x, y), int(pixel * 2))
img.show()

#Mid-square
values = mid_square(seed, n)
img = Image.new('1', (width, height))
pixels = img.load()
for y in range(height):
    for x in range(width):
        pixel_value = values[y * width + x]
        pixel = int(pixel_value * 2)
        pixels[x, y] = pixel
img.show()

#Random.uniform(Python)
width = 256
height = 256
img = Image.new('1', (width, height))
pixels = img.load()
for y in range(height):
    for x in range(width):
        pixel_value = random.uniform(0, 1)
        pixel = int(pixel_value * 2)
        pixels[x, y] = pixel
img.show()