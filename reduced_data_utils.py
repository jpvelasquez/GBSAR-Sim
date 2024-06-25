def distance_nk(r_n,x_k): # vector de coordenadas de la imagen, punto del riel "k"
    d=((r_n[0]-x_k)**2+(r_n[1])**2)**0.5
    return d

def get_parameters2(dset):
    # Definición y lectura de parámetros
    c = 0.299792458 # Velocidad de la luz (x 1e9 m/s)
    fi = dset['fi'].values[0]*1e-9 # Frecuencia inferior(GHz)
    fs = dset['ff'].values[0]*1e-9 # Frecuencia superior(GHz)
    Nf = dset['nfre'].values[0] # Numero de frecuencias
    xi = float(dset['xi'].values[0]) # Posicion inicial(m)
    xf = float(dset['xf'].values[0]) # Posicion final(m)
    Np = dset['npos'].values[0] # Numero de posiciones
    dx = float(dset['dx'].values[0]) # Paso del riel(m)
    b_angle = dset['beam_angle'].values[0]/2 # Angulo del haz(°)

    fc = (fi+fs)/2 # Frecuencia Central(GHz)
    BW = fs-fi # Ancho de banda(GHz)
    Ls = xf-xi # Longitud del riel (m)
    Ro = 0 # Constante debido al delay de los cables(m)

    # Definicion de las dimensiones de la imagen ("dh" y "dw" usados solo en BP)
    w = 20#50#100#700 # Ancho de la imagen(m)
    h = 50#100#200#850 # Altura de la imagen(m)
    hi = 50#100 # Posicion inicial en el eje del rango
    dw = 0.5 # Paso en el eje del ancho(m)
    dh = 0.5 # Paso en el eje de la altura(m)
    print('Initial and final positions: ', xi,xf)
    N_c = w/dw
    N_r = h/dh
    print('Dimensions of the image: ',N_r,N_c)
    prm={
        'c':c,
        'fc':fc,
        'BW':BW,
        'Ls':Ls,
        'Ro':Ro,
        'theta':b_angle,
        'Np':Np,
        'Nf':Nf,
        'dx':dx,
        'w':w,
        'h':h,
        'hi':hi,
        'dw':dw,
        'dh':dh,
        'xi':xi,
        'xf':xf
    }
    return prm

