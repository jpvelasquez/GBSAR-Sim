import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy.linalg as LA
import timeit
from sklearn.metrics import mean_squared_error
import math
import spgl1 as spgl1
from sklearn.linear_model import orthogonal_mp
import pywt
import h5py
from data_matrices import *
from sys import getsizeof
#import DLIPsim as dlip
import inspect
from pprint import pprint
import warnings
from cs_utils import *
from decimal import Decimal
import psutil
import gc
warnings.filterwarnings('ignore')
gc.set_threshold(0)
######################################################################
rugosity = 0.1
target='elliptical'
SNR = 1e-2 # 0dB
#1e4#40 #dB
SNR_dB = int(10*np.log10(SNR))
print(SNR_dB)
c = 0.3 #0.299792458 # Velocidad de la luz (x 1e9 m/s)
fc = 15 # Frecuencia Central(GHz)
BW = 0.6 # Ancho de banda(GHz)
Ls = 1.2 # 4 # 0.6 # Longitud del riel (m)
Ro = 0 # Constante debido al delay de los cables(m)
theta = 90 # Angulo azimuth de vision de la imagen final(grados sexagesimales E [0-90])
###############################################################################
I_t, R_t, x_c,y_c, N_r, N_c, w, h,dw, dh, str_target = SelectTarget(target,rugosity)
# Hallando el Np a partir de los pasos
dp=c/(4.1*fc*np.sin(theta*np.pi/180)) # paso del riel para un angulo de vision de 180°
Np=int(Ls/dp)+1 # Numero de pasos del riel
if Np%2!=0:
    Np+=1   # Para que el numero de pasos sea par
# Hallando el Nf en funcion a la distancia máxima deseada
r_r=c/(2*BW) # resolucion en rango
Nf=150#int(h/r_r) #+1 #Numero de frecuencias
################################################################################
print(I_t.shape, x_c.shape, y_c.shape)
I_sim = I_t.reshape(N_r,N_c)
################################################################################
Lx,Ly,dx,dy = w,h,dw,dh # Dimensiones de la imagen
x_min, x_max= [-(Lx+dx)/2, (Lx+dx)/2]
y_min, y_max= [0-dy/2,Ly+dy/2]
extent = [x_min,x_max,y_min,y_max]
prm={'c':c,'fc':fc,'BW':BW,'Ls':Ls,'Ro':Ro,'theta':theta,'Np':Np,'Nf':Nf,
       'w':w,'h':h,'dw':dw,'dh':dh}
################################################################################
fi=fc-BW/2 # Frecuencia inferior(GHz)
fs=fc+BW/2 # Frecuencia superior(GHz)
# Cálculo de las resoluciones
rr_r=c/(2*BW) # Resolución en rango
rr_a=(c/(2*Ls*fc))*R_t.T[1].max() # Resolución en azimuth
#-----------------VERIFICACIÓN DE CONDICIONES------------------------
# Rango máximo
R_max=Nf*c/(2*BW)
# Paso del riel máximo
dx_max=c/(fc*4*np.sin(theta*np.pi/180)) # Theta en grados sexagesimales
print("------------------------------------------------------")
print("--------------INFORMACIÓN IMPORTANTE------------------")
print("------------------------------------------------------")
print("- Resolución en rango(m) : ", rr_r)
print("- Resolución en azimuth(m): ", rr_a)
print("------------------------------------------------------")
print("- Rango máximo permitido(m): ", R_max)
print("------------------------------------------------------")
print("______¿Se cumplen las siguientes condiciones?_________")
print("Rango máximo del target <= rango máximo?: ", R_t.T[1].max()<=R_max) # Ponerle un try-except
print("Paso del riel <= paso máximo?: ", dp<=dx_max) # Evita el aliasing en el eje de azimuth
print("------------------------------------------------------")
########################################################################
#----------------OBTENCIÓN DE la historia DE FASE----------------------
#start_time = timeit.default_timer()
Ski = get_phaseH(prm,I_t,R_t)
#print("Tiempo del procesamiento(IP): ",timeit.default_timer() - start_time," s")
data = {'Ski':Ski, 'dp':dp, 'fi':fi, 'fs':fs, 'R_max':R_max}
module_sim = abs(I_t).reshape(N_r,N_c)
phase_sim = np.zeros(I_t.shape).reshape(N_r,N_c)
#####################################################################################################
fontsize=18
#rectangle = Rectangle(( x_c[8],  y_c[8]), 30, 30,
#                      alpha=1, facecolor='black',fill=None,linewidth=4)
#rectangle2 = Rectangle(( x_c[8],  y_c[8]), 30, 30,
#                      alpha=1, facecolor='black',fill=None,linewidth=4)
'''
fig, ax = plt.subplots(1,2,figsize=(12,6), layout='constrained')#,sharey=True)
clrs1 = ax[0].pcolormesh(x_c, y_c,np.abs(I_sim) ,cmap='jet')
box=ax[0].get_position()
cb1=plt.colorbar(clrs1,ax=ax[0])#,location='left'
ax[0].set_title('Magnitude ' ,fontsize=fontsize)
cb1.set_label(r'dB', fontsize=fontsize)
cb1.ax.tick_params(labelsize=fontsize-3)
ax[0].xaxis.set_tick_params(labelsize=fontsize)
ax[0].yaxis.set_tick_params(labelsize=fontsize)
#ax[0].add_patch(rectangle)
clrs2 = ax[1].pcolormesh(x_c, y_c,np.angle(I_sim) ,cmap='jet')
box=ax[1].get_position()
cb2=plt.colorbar(clrs2,ax=ax[1])
ax[1].set_title('Phase ' ,fontsize=fontsize)
cb2.set_label(r'dB', fontsize=fontsize)
cb2.ax.tick_params(labelsize=fontsize-3)
ax[1].xaxis.set_tick_params(labelsize=fontsize)
ax[1].yaxis.set_tick_params(labelsize=fontsize)
#ax[1].add_patch(rectangle2)
fig.suptitle('Simulated target (rugosity = %.2f)' % rugosity,fontsize=fontsize)
plt.savefig('simulated-target-with-displacement-region-marked.png' ,bbox_inches='tight')
plt.close()
'''
###############################################################################################
fi = data['fi']
fs = data['fs']
riel_p = np.linspace(-Ls/2, Ls/2, Np) # Vector de posiciones
f = np.linspace(fi, fs, Nf) # Vector de frecuencias
ks = 2*np.pi*f/c # Range domain wavenumber
r_c=np.asarray([(i,j) for j in y_c for i in x_c],dtype="object")
###############################################################################################
D2 = np.linalg.norm(Ski)**2/(Ski.shape[0]*Ski.shape[1])
n0 = np.sqrt(D2/SNR)
print(SNR)
print(n0)
print(Ski.shape)
fils,cols = Ski.shape
#por = 40
#por = np.linspace(5,100,20)
#per=por/100
porcentajes = np.linspace(5,100,20)
percentages = porcentajes/100
R_t = 0
numRealizations = 100
mode = 'periodization'#['zero', 'constant', 'symmetric', 'periodic', 'smooth', 'periodization', 'reflect', 'antisymmetric', 'antireflect']
wavelet = 'db2'
lambd = c/fc
#######################################################################################################
sigma_list = []
for j,per in enumerate(percentages):
    print("************************************")
    print("Percentage: %2d %%" % porcentajes[j])
    xf_list = []
    x_temp = np.zeros((len(y_c),len(x_c)),dtype=complex)
    Im_list = []
    for i in range(numRealizations):
        print('Realization number',i)
        noise_re = np.random.rand(fils,cols)
        noise_im = np.random.rand(fils,cols)
        noise = n0*(noise_re+1j*noise_im)/np.sqrt(2)
        Ski_noisy = Ski + noise
        A_rand, S1_rand = UnderSample(r_c,riel_p,R_t,Nf,Np,per,Ski_noisy,f)
        b_rand = S1_rand.flatten()
        N = A_rand.shape[1]#b_rand.shape[0]
        Psi = WaveletMatrix(wavelet,N)
        #ar = Psi.T@Psi
        #ar[ar<1e-15]=0
        sigma   = n0
        N = A_rand.shape[1]
        Psi = WaveletMatrix(wavelet,N)
        A_i = np.imag(np.dot(A_rand,Psi))
        A_r = np.real(np.dot(A_rand,Psi))
        b_i = np.imag(b_rand)
        b_r = np.real(b_rand)
        Aug = np.block([[A_r, -A_i],[A_i, A_r]])
        baug = np.concatenate((b_r,b_i),axis=0)
        xaug,raug,gaug,info = spgl1.spg_bpdn(Aug, baug.flatten(), sigma)
        x = xaug[0:N] + 1j*xaug[N:]
        x_mp = Psi@x
        x_f = np.reshape(x_mp,(len(y_c),len(x_c)))
        xf_list.append(x_f)
        if i>=1:
            Im_prod = np.multiply(x_temp,np.conjugate(x_f))[8,8]   
            Im_list.append(Im_prod)
            del Im_prod
        x_temp = x_f
        del A_rand, b_rand, S1_rand, Psi, A_i, A_r, b_r, b_i
        del Aug, baug,xaug,raug,gaug,info,x, x_mp, x_f
        del Ski_noisy, noise_re, noise_im, noise
        print(psutil.cpu_percent())
        gc.collect()
    ##############################
    del x_temp
    Im_list = np.array(Im_list)
    phase_list = np.angle(Im_list)
    sigma_rad = np.std(phase_list)
    sigma_deg = (180/np.pi)*sigma_rad
    sigma_delta_r = sigma_rad*(lambd/4*np.pi)
    delta_r = np.array(phase_list)*(lambd/4*np.pi)
    str_std = '%.2E' % Decimal(sigma_delta_r)
    print(str_std)
    #######################################
    fontsize=24
    fig, ax = plt.subplots(2,1,figsize=(18,9), layout='constrained',sharex=True)
    ax[0].plot(phase_list)
    ax[0].xaxis.set_tick_params(labelsize=fontsize)
    ax[0].yaxis.set_tick_params(labelsize=fontsize)
    ax[0].set_ylabel(r'$\Delta \phi$ (rad)',fontsize=fontsize)
    ax[0].grid()
    ax[1].plot(delta_r*1000,label=r'$\sigma_r$ = %s mm' % str_std)
    ax[1].xaxis.set_tick_params(labelsize=fontsize)
    ax[1].yaxis.set_tick_params(labelsize=fontsize)
    ax[1].set_ylabel(r'$\Delta r$ (mm)',fontsize=fontsize)
    ax[1].set_xlabel(r'Realization number',fontsize=fontsize)
    ax[1].grid()
    ax[1].legend(fontsize=fontsize,loc='best')
    fig.suptitle(r'Phase/displacement ($\Delta \phi$/ $\Delta r$) vs realization number (SNR = %d dB, data taken = %d%%)' % (SNR_dB,porcentajes[j]) ,fontsize=fontsize)
    plt.savefig('displacement-variance-SNR=%d-dB-percentage=%d.png' % (SNR_dB,porcentajes[j]),bbox_inches='tight')
    plt.close(fig)
    sigma_list.append(sigma_delta_r)
    del Im_list, phase_list
#########################################################################################################################################
#########################################################################################################################################
str_name_hdf5 = 'realizationNum=%d-sigma-vs-percentage-undersampling-pos-db2-haar-sym2-arrays-snr=%d-db.hdf5' % (numRealizations,SNR_dB)
h5f = h5py.File(str_name_hdf5,'w')      
h5f.create_dataset('percentage', data=percentages)
h5f.create_dataset('sigma_delta_r', data=sigma_list)
h5f.close()
#########################################################################################################################################
