import numpy as np
#import DLIPsim as dlip
import sarPrm as sp
import matplotlib.pyplot as plt
import numpy.linalg as LA
import drawFigures as dF # Library created to draw FFT functions
#import SARdata as Sd # Library created to get theoric data(Phase Historic)
import timeit
from sklearn.metrics import mean_squared_error
import math
#import spgl1 as spgl1
#from sklearn.linear_model import orthogonal_mp
from PIL import Image
import h5py as hp 
import BP_real_main as FDBP_data


show=True
dset_name = 'dset_1.hdf5'
f = hp.File(dset_name,'r')
global dset
dset = f['sar_dataset']
prm = sp.get_parameters2(dset)
#global c,fc,BW,Nf,Ls,Np,Ro,theta,Lx,Ly,dx,dy,yi,dp2

c,fc,BW,Nf = prm['c'],prm['fc'],prm['BW'],prm['Nf']
Ls,Np,Ro,theta = prm['Ls'],prm['Np'],prm['Ro'],prm['theta']
Lx,Ly,dx,dy,yi = prm['w'],prm['h'],prm['dw'],prm['dh'],prm['hi'] # Dimensiones de la imagen
time = dset.attrs['datetime']
xi,xf = prm['xi'], prm['xf']
dp2 = prm['dx']
print(prm)
N_c = int(Lx/dx)
N_r = int(Ly/dy)
riel_p = np.linspace(xi,xf,N_c)
#print('Dimensions of the image: ',N_r,N_c)
#print('Largo y ancho de la imagen:', Lx, Ly)

per = 30/100
Np_rand = int(per*Np)
n = Np_rand  # for 2 random indices
ind = np.random.choice(Np, n, replace=False)
index = np.sort(ind)
#print(Np_rand,Np,Nf)

def distance_nk(r_n, x_k): # punto "n", punto del riel "k"
    d=((r_n[0]-x_k)**2+(r_n[1])**2)**0.5
    return d

def get_SAR_real_data():
    """ Obtiene el histórico de fase ya sea simulado o real"""
    # Cálculo de parámetros
    dp = Ls/(Np-1) # Paso del riel(m)
    df = BW/(Nf-1) # Paso en frecuencia del BW
    fi = fc-BW/2 # Frecuencia inferior(GHz)
    fs = fc+BW/2 # Frecuencia superior(GHz)

    # Rango máximo
    R_max=Nf*c/(2*BW)
    # Paso del riel máximo
    dp_max=c/(fc*4*np.sin(theta*np.pi/180)) # Theta en grados sexagesimales

    # Cálculo de las resoluciones
    rr_r = c/(2*BW) # Resolución en rango
    rr_a = c/(2*Ls*fc) # Resolución en azimuth

    #-----------------VERIFICACIÓN DE CONDICIONES------------------------
    print("------------------------------------------------------")
    print("--------------INFORMACIÓN IMPORTANTE------------------")
    print("------------------------------------------------------")
    print("- Resolución en rango(m) : ", rr_r)
    print("- Resolución en azimuth(rad): ", rr_a)
    print("------------------------------------------------------")
    print("- Rango máximo permitido(m): ", R_max)
    print("------------------------------------------------------")
    print("______¿Se cumplen las siguientes condiciones?_________")
    print("Rango máximo del target <= rango máximo?: ", R_max<=R_max) # Ponerle un try-except
    print("Paso del riel <= paso máximo?: ", dp<=dp_max) # Evita el aliasing en el eje de azimuth
    print("Theta: ", theta)
    print("------------------------------------------------------")

    #----------------OBTENCIÓN DEL HISTÓRICO DE FASE----------------------
    Sr_f = np.array(list(dset))
    print("Sr_f shape: ", Sr_f.shape)

    #-----------------GRÁFICA DEL HISTÓRICO DE FASE-----------------------
    dF.plotImage(Sr_f, x_min=fi, x_max=fs, y_min=-Ls/2, y_max=Ls/2,xlabel_name='Frecuency(GHz)',
                 ylabel_name='Riel Position(m)', title_name='Histórico de fase',unit_bar='dBu', origin_n='upper')

    return {'Sr_f':Sr_f, 'df':df, 'fi':fi, 'fs':fs, 'rr_r':rr_r}

data = get_SAR_real_data()

#start_time = timeit.default_timer()
Ski = data['Sr_f'].copy()

fi = data['fi']
fs = data['fs']

    #------------- PRIMERA ETAPA: Planteamiento del problema -------------
    # Get following matrix and vectors:
    #              [Es] = [A][X] ___________________
    #   Raw_data <--|      |--> SAR_System_Matrix  |--> Reflectivity
    #---------------------------------------------------------------------

    # Creation of vector S1
#S1 = np.reshape(Ski.T,(len(Ski)*len(Ski[0]),1)) # Convert Raw Data matrix into vector

riel_p = np.linspace(-Ls/2, Ls/2, Np) # Vector de posiciones
f = np.linspace(fi, fs, Nf) # Vector de frecuencias
ks = 2*np.pi*f/c # Range domain wavenumber
    # d_ks = ks[1]-ks[0] # Step

    # Definition of coordinates, including last value(Lx/2 and Ly)
x_c=np.arange(-Lx/2,Lx/2+dx,dx)
y_c=np.arange(0,Ly+dy,dy)
r_c=np.array([(i,j) for j in y_c for i in x_c])
print(x_c.shape)

riel_p_rand = riel_p[index]
Ski_rand = Ski[index]

start_time = timeit.default_timer()
S1_rand = np.reshape(Ski_rand.T,(len(Ski_rand)*len(Ski_rand[0]),1))
A_rand = np.zeros((Np_rand*Nf,len(r_c)), dtype=complex)
riel_p_rand = riel_p[index]
for n in range(len(r_c)):
    m = 0
    for s in range(len(f)):
        for l in range(len(riel_p_rand)):
            A_rand[m,n] = np.exp(-2j*ks[s]*np.abs(distance_nk(r_c[n],riel_p_rand[l])))#/(4*np.pi*distance_nk(r_c[n],riel_p[l]))**2
            m+=1
#start_time = timeit.default_timer()
Ap_rand = LA.pinv(A_rand) # Calculate pseudoinverse using SVD
S2_rand = Ap_rand.dot(S1_rand) # Array
S3_rand = np.reshape(S2_rand,(len(y_c),len(x_c)))
print("S3 y S2 random: ", S3_rand.shape, S2_rand.shape)
print("Tiempo del procesamiento(IP): ",timeit.default_timer() - start_time," s")
print("Tiempo de la inversión: ",timeit.default_timer() - start_time," s")
Im_rand = S3_rand.copy()
round_per = np.round(per*100)
matSize = A_rand.__sizeof__()
print('Matrix size: ', matSize)

fontsize=16
fig, (ax, ax2) = plt.subplots(ncols=2,figsize=(12,6),
                  gridspec_kw={"width_ratios":[1,1]})#, 0.05]})
fig.subplots_adjust(wspace=0.3)
#im  = ax.imshow(np.random.rand(11,8), vmin=0, vmax=1)
#im2 = ax2.imshow(np.random.rand(11,8), vmin=0, vmax=1)
clrs1 = ax.pcolormesh(x_c, y_c, 20*np.log10(np.abs(Im_rand)),cmap='jet')
clrs2 = ax2.pcolormesh(x_c, y_c, np.angle(Im_rand),cmap='jet')
ax.set_ylabel("Rango (m) label")
ax.set_xlabel('Azimut (m)',fontsize=fontsize)
ax.xaxis.set_tick_params(labelsize=fontsize)
ax.yaxis.set_tick_params(labelsize=fontsize)
ax.set_title('Magnitud',fontsize=fontsize+4)
cb1 = plt.colorbar(clrs1, ax=ax)
ax2.set_title('Fase',fontsize=fontsize+4)
ax2.set_xlabel('Azimut (m)',fontsize=fontsize)
ax2.xaxis.set_tick_params(labelsize=fontsize)
ax2.yaxis.set_tick_params(labelsize=fontsize)
cb2 = plt.colorbar(clrs2, ax=ax2)
#fig.colorbar(clrs2, cax=cax)
fig.suptitle('Magnitud - DLIP undersampled (%d %%) - %s' % (round_per, time) ,fontsize=fontsize)
plt.show()
