## Funciones

import numpy as np
import scipy.linalg as la


def filaPivote(U, col):  ##buscar la fila con el elemento más grande
    n = U.shape[0]

    #Guardo valor máximo y fila (arrancando desde la columna en la que estoy)
    filaMax = col
    valorMax = abs(U[col, col])

    for fil in range(col + 1, n):
        if abs(U[fil, col]) > valorMax:
            valorMax = abs(U[fil, col])
            filaMax = fil

    return filaMax


def pivotear(P,L,U, col):
    filaMax = filaPivote(U,col) ##busco la fila con el emeento máximo

    if filaMax != col:   #Si hay que pivotear
        U[[col, filaMax]] = U[[filaMax, col]]   # Intercambiamos filas en U
        L[[col, filaMax], :col] = L[[filaMax, col], :col] #Lo mismo en L (para sólo hasta la fila actual)
        P[[col, filaMax]] = P[[filaMax, col]] # y en P

    return P,L,U


def calcularLU(A):
    n = A.shape[0]
    U = A.astype(float)
    L = np.eye(n)
    P = np.eye(n)

    for col in range(n-1):
        P,L,U = pivotear(P,L,U, col)

        if np.isclose(U[col, col], 0):
            raise ValueError("No es invertible, su pivote es 0")

        for fil in range(col+1, n):
            factor = U[fil][col] / U[col][col]
            L[fil][col] = factor
            U[fil] -= U[col] * factor

        #if np.isclose(U[n-1, n-1], 0):
        #    raise ValueError("No es invertible, su último pivote es 0") #faltabaa veriticar si el ultimo pivote es 0

    return P,L,U


def inversaLU(P,L,U):
  #en vez de I, directamente pongo P acá  (me corrije las permutaciones)
  # Otra manera sería Poner Y = solve_triangular(L, I, lower=True)
  # luego                   Inv = solve_triangular(U, Y, lower=False)
  # y al final              Inv = Inv @ P
  # Esa manera es más clara, pero realiza una multiplicación de más.

    Y = la.solve_triangular(L, P, lower=True)
    Inv = la.solve_triangular(U, Y, lower=False)
    return Inv


# Calcula p con el modelo de región simple
def calcularLeontief(A, d):
    n = A.shape[0]
    I = np.eye(n)
    P,L,U = calcularLU(I - A)
    MatrizLeont = inversaLU(P,L,U)
    return MatrizLeont @ d


# Calcula p_r con el modelo de dos regiones
def calcularLeontief_dos_paises_LU(A_rr, A_ss, A_rs, A_sr, d_r):
    I_r = np.eye(A_rr.shape[0])
    I_s = np.eye(A_ss.shape[0])

    # Calcular (I - A_ss) y su inversa usando descomposición LU
    P_ss, L_ss, U_ss = calcularLU(I_s - A_ss)
    inv_A_ss = inversaLU(P_ss, L_ss, U_ss)

    # Calcular el término interregional A_rs @ inv(I_s - A_ss) @ A_sr
    termino = A_rs @ inv_A_ss @ A_sr

    # Calcular la inversa de (I - A_rr - A_rs @ inv(I_s - A_ss) @ A_sr) usando descomposición LU
    P_total, L_total, U_total = calcularLU(I_r - A_rr - termino)
    inv_total = inversaLU(P_total, L_total, U_total)

    # Calcular el cambio en la producción
    res_p_r = inv_total @ d_r

    return res_p_r

"""
def metodoPotencia(A, veces):

    tabla = np.empty(veces)
    x = np.random.rand(A.shape[1])
    norma = la.norm(x,2)
    x = x/norma

    for i in range(veces):
        x = A @ x
        norma = la.norm(x,2)
        x = x/norma

        autoval = ((np.transpose(x) @ A) @ x) / (np.transpose(x) @ x)
        tabla[i] = autoval

    return autoval, x, tabla
"""

def metodoPotencia(A, epsilon):
    x = np.random.rand(A.shape[1])
    norma = la.norm(x, 2)
    x = x / norma
    autoval_anterior = 0
    tabla = []

    for _ in range(100000):
        x_nueva = A @ x
        norma = la.norm(x_nueva, 2)
        x_nueva = x_nueva / norma

        autoval = ((np.transpose(x_nueva) @ A) @ x_nueva) / (np.transpose(x_nueva) @ x_nueva)
        tabla.append(autoval)

        ##Agrego la parada

        if la.norm(x_nueva - x, 2) <= (1 - epsilon):
            break

        x = x_nueva

    return autoval, x, np.array(tabla)

def sumaGeom (A, veces):
    I = np.eye(np.shape(A)[0])
    suma = I
    tabla = np.empty(veces+1)
    tabla[0]= la.norm(I, 2)
    for i in range(1, veces+1):
        suma = suma + np.linalg.matrix_power(A, i)
        tabla[i] = la.norm(suma, 2)

    return suma, tabla

def coeficientesTécnicos(z, p):
  n = len(z)
  A = np.zeros((n,n))
  for i in range (0,n):
    A[:,i] = z[:,i]/p[i]
  return A

def error(A, inv_ImenosA, n):
    I = np.eye(A.shape[0])  # Matriz identidad
    suma = I  # Inicializamos la suma con la identidad
    errores = np.empty(n+1)
    errores[0] = la.norm(I - inv_ImenosA, 2)  # Error inicial solo con la identidad

    # Calcular la suma acumulada y el error en cada paso hasta n
    for i in range(1, n + 1):
        suma += np.linalg.matrix_power(A, i)  # Agregar la potencia acumulada de A^i
        error = la.norm(suma - inv_ImenosA, 2)  # Calcular el error respecto a la inversa teórica
        errores[i]= error

    return errores