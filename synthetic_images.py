from re import I
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def get_matrix_data1():
    # Se define una imagen con la letra T
    show=True
    str_target = 't'
    ro_x, ro_y = -2.5, 2 # Posicion inicial (x,y)m
    d_x, d_y = 0.25,0.25#0.125,0.125 #0.25, 0.25 # Paso en los ejes (x,y)m
    N_r, N_c = 21,21#41,41 #21, 21 # Numero de filas y columnas de la matriz de targets
    Data = np.array([[0]*7+[1]*7+[0]*7]*14+[[1]*N_c]*7)#np.array([[0]*13+[1]*15+[0]*13]*28+[[1]*N_c]*13) #np.array([[0]*4+[1]*3+[0]*4]*7+[[1]*N_c]*4)
    I_t = Data.reshape(1,N_r*N_c)[0] # Vector de intensidades

    rt_x=ro_x+d_x*np.arange(N_c) # Coordenada x de los targets(empezando en ro_x)
    rt_y=ro_y+d_y*np.arange(N_r) # Coordenada y de los targets(empezando en ro_y)
    rt=np.array([(x,y) for y in rt_y for x in rt_x]) #  Vector de coordenadas (x,y)

    # Grafica de la imagen a simular
    if show:
        fig, ax = plt.subplots()
        ax.imshow(Data,aspect='auto',origin='lower',extent=[rt_x.min(), rt_x.max(), rt_y.min(),rt_y.max()])
        ax.set(xlabel='Eje x(m)',ylabel='Eje y(m)', title='Imagen para la simulación')

    return I_t, rt, rt_x, rt_y, N_r, N_c, str_target

def get_matrix_data3():
    # Se define una imagen con la letra ROJ
    str_target = 'roj'
    show=True
    ro_x, ro_y = -5, 3 # Posicion inicial (x,y)m
    d_x, d_y = 0.25,0.25#0.125,0.125 #0.25, 0.25 # Paso en los ejes (x,y)m
    N_r, N_c = 16,41

    # Formando las letras x separado
    # Letra R
    #letraR = np.array([[1]*3+[0]*6+[1]*2]*2+[[1]*3+[0]*4+[1]*2+[0]*2]*2+[[1]*3+[0]*2+[1]*2+[0]*4]*2+[[1]*5+[0]*6]*2+[[1]*11]*8)
    letraR = np.array([[1]*3+[0]*6+[1]*2]*2+[[1]*3+[0]*4+[1]*2+[0]*2]*2+[[1]*3+[0]*2+[1]*2+[0]*4]*2+[[1]*5+[0]*6]*2+[[1]*11]*3+[[1]*3+[0]*5+[1]*3]*2+[[1]*11]*3)
    # Letra 0
    #letraO = np.ones((16,11))
    letraO = np.array([[1]*11]*3+[[1]*3+[0]*5+[1]*3]*10+[[1]*11]*3)
    # Letra J
    letraJ = np.array([[1]*9+[0]*2]*3+[[1]*3+[0]*3+[1]*3+[0]*2]*2+[[0]*6+[1]*3+[0]*2]*8+[[1]*11]*3)
    # Joining all the letters
    Data = np.zeros((16,41))#np.zeros((16,41))
    Data[:,2:13] = letraR
    Data[:,15:26] = letraO
    Data[:,28:39] = letraJ

    I_t = Data.reshape(1,N_r*N_c)[0] # Vector de intensidades

    rt_x=ro_x+d_x*np.arange(N_c) # Coordenada x de los targets(empezando en ro_x)
    rt_y=ro_y+d_y*np.arange(N_r) # Coordenada y de los targets(empezando en ro_y)
    rt=np.array([(x,y) for y in rt_y for x in rt_x]) #  Vector de coordenadas (x,y)

    # Grafica de la imagen a simular
    if show:
        fig, ax = plt.subplots(figsize=(12,6))
        im=ax.imshow(Data,aspect='auto',origin='lower',extent=[rt_x.min(), rt_x.max(), rt_y.min(),rt_y.max()])
        ax.set(xlabel='Eje x(m)',ylabel='Eje y(m)', title='Imagen para la simulación')

    return I_t, rt, rt_x, rt_y, N_r, N_c, str_target

def get_matrix_data5():
    # Se define una imagen con la letra UNI
    str_target = 'uni'
    show = True
    ro_x, ro_y = -5, 3 # Posicion inicial (x,y)m
    d_x, d_y = 0.25,0.25#0.125,0.125 #0.25, 0.25 # Paso en los ejes (x,y)m
    N_r, N_c = 16,41

    # Formando las letras x separado
    # Letra U
    letraU = np.array([[1]*11]*3+[[1]*3+[0]*5+[1]*3]*13)
    # Letra N
    letraN = np.array([[1]*16]*3+[[0]*10+[1]*4+[0]*2]*1+[[0]*8+[1]*4+[0]*4]*1+[[0]*6+[1]*4+[0]*6]*1+[[0]*4+[1]*4+[0]*8]*1+[[0]*2+[1]*4+[0]*10]*1+[[1]*16]*3).T
    # Letra I
    letraI = np.array([[1]*11]*3+[[0]*4+[1]*3+[0]*4]*10+[[1]*11]*3)

    # Joining all the letters
    Data = np.zeros((16,41))
    Data[:,2:13] = letraU
    Data[:,15:26] = letraN
    Data[:,28:39] = letraI

    I_t = Data.reshape(1,N_r*N_c)[0] # Vector de intensidades

    rt_x=ro_x+d_x*np.arange(N_c) # Coordenada x de los targets(empezando en ro_x)
    rt_y=ro_y+d_y*np.arange(N_r) # Coordenada y de los targets(empezando en ro_y)
    rt=np.array([(x,y) for y in rt_y for x in rt_x]) #  Vector de coordenadas (x,y)

    # Grafica de la imagen a simular
    if show:
        fig, ax = plt.subplots()
        im=ax.imshow(Data,aspect='auto',origin='lower',extent=[rt_x.min(), rt_x.max(), rt_y.min(),rt_y.max()])
        ax.set(xlabel='Eje x(m)',ylabel='Eje y(m)', title='Imagen para la simulación')

    return I_t, rt, rt_x, rt_y, N_r, N_c, str_target
