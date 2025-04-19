#program_1 pokazuje różnice czasu obliczeń

import numpy as np
import timeit

a = list(range(1000))
aa = np.array(a)

def dodawanie_lista (a, value):
    for i in range(len(a)):
        a[i] += value
    return a

def dodawanie_array (aa, value):
    return aa + value

czas_lista = timeit.timeit(lambda: dodawanie_lista (a, 10), number=100)
czas_array = timeit.timeit(lambda: dodawanie_array (aa, 10), number=100)



print("Czas dobliczeń (listy):", czas_lista)
print("Czas obliczeń numpy:", czas_array)
