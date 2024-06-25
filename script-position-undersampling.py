import h5py
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import timeit
from sklearn.metrics import mean_squared_error
import math
import spgl1 as spgl1
from sklearn.linear_model import orthogonal_mp
import pywt
from data_matrices import *
from sys import getsizeof
#import DLIPsim as dlip
import inspect
from pprint import pprint
import warnings
from cs_utils import *
warnings.filterwarnings('ignore')
import psutil
import gc
gc.set_threshold(0)

rugosity = 0.1
target='elliptical'
SNR = 1e-1
#1e-2 # -20 dB
#1e1 # 10 dB
#1e3 # 30 dB 
#1e4 #40 dB
#1e2 # 20 dB 
#1e3 # 30 dB
#1e2 # 20 dB
#1e1 # 10 dB
#1 # 0 dB
#1e-1 # -10 dB
#1e4#40 #dB
pers = np.linspace(5,100,20)/100
#pers = np.linspace(5,100,2)/100

SNR_db = int(10*np.log10(SNR))
print(SNR_db)
str_name_hdf5 = 'undersampling-position-db2-haar-sym2-arrays-snr=%d-db.hdf5' % SNR_db
print('*************************************************')
print('Hdf5 file name: ', str_name_hdf5)
print('*************************************************')

c = 0.3 #0.299792458 # Velocidad de la luz (x 1e9 m/s)
fc = 15 # Frecuencia Central(GHz)
BW = 0.6 # Ancho de banda(GHz)
Ls = 1.2 # 4 # 0.6 # Longitud del riel (m)
Ro = 0 # Constante debido al delay de los cables(m)
theta = 90 # Angulo azimuth de vision de la imagen final(grados sexagesimales E [0-90])

I_t, R_t, x_c,y_c, N_r, N_c, w, h,dw, dh, str_target = SelectTarget(target,rugosity)
# Hallando el Np a partir de los pasos
dp=c/(4.1*fc*np.sin(theta*np.pi/180)) # paso del riel para un angulo de vision de 180°
Np=int(Ls/dp)+1 # Numero de pasos del riel
if Np%2!=0:
    Np+=1   # Para que el numero de pasos sea par
# Hallando el Nf en funcion a la distancia máxima deseada
r_r=c/(2*BW) # resolucion en rango
Nf=150#int(h/r_r) #+1 #Numero de frecuencias
#print(Nf,Np,N_r*N_c)
#print(dw, dh, N_r, N_c)
#print(I_t.shape)

por = 40
per=por/100
#err = 0.1
#a = np.linspace(1,10,10)
#Np = 10
Nf_rand = int(per*Nf)
n = Nf_rand  # for 2 random indices
ind = np.random.choice(Nf, n, replace=False)
index = np.sort(ind)

vmin, vmax = -100, -20 #dB
Lx,Ly,dx,dy = w,h,dw,dh # Dimensiones de la imagen
x_min, x_max= [-(Lx+dx)/2, (Lx+dx)/2]
y_min, y_max= [0-dy/2,Ly+dy/2]
extent = [x_min,x_max,y_min,y_max]
print(Lx,Ly,dx,dy)
print(Lx/dx+1, Ly/dy+1)
print(extent)
prm={'c':c,'fc':fc,'BW':BW,'Ls':Ls,'Ro':Ro,'theta':theta,'Np':Np,'Nf':Nf,
       'w':w,'h':h,'dw':dw,'dh':dh}

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

#----------------OBTENCIÓN DE la historia DE FASE----------------------
start_time = timeit.default_timer()
Ski = get_phaseH(prm,I_t,R_t)
print("Tiempo del procesamiento(IP): ",timeit.default_timer() - start_time," s")
data = {'Ski':Ski, 'dp':dp, 'fi':fi, 'fs':fs, 'R_max':R_max}
#module_sim = 20*np.log10(abs(I_t)).reshape(N_r,N_c)#abs(I_t).reshape(N_r,N_c)#
phase_sim = np.zeros(I_t.shape).reshape(N_r,N_c)
#print(extent)
I_sim = I_t.reshape(N_r,N_c)
module_sim = abs(I_sim)
#print(module_sim.shape)
print('Phase history: ', Ski.shape)
D2 = np.linalg.norm(Ski)**2/(Ski.shape[0]*Ski.shape[1])
#print(D2)
n0 = np.sqrt(D2/SNR)
#print(n0)
#print(Ski.shape)
fils,cols = Ski.shape
#noise = np.random.normal(scale=1,size=(2,2))
noise_re = np.random.rand(fils,cols)
noise_im = np.random.rand(fils,cols)
noise = n0*(noise_re+1j*noise_im)
sigma = n0
######################################
Ski_noisy = Ski + noise
######################################
#Ski = data['Ski'].copy()
fi = data['fi']
fs = data['fs']
# Creation of vector S1
S1 = np.reshape(Ski_noisy.T,(len(Ski)*len(Ski[0]),1)) # Convert Raw Data matrix into vector
riel_p = np.linspace(-Ls/2, Ls/2, Np) # Vector de posiciones
f = np.linspace(fi, fs, Nf) # Vector de frecuencias
ks = 2*np.pi*f/c # Range domain wavenumber
r_c=np.asarray([(i,j) for j in y_c for i in x_c], dtype='object')

A_rand, b_rand =  UnderSample(r_c,riel_p,R_t,Nf,Np,per,Ski_noisy,f)
#print(A_rand.shape, b_rand.shape)
N = A_rand.shape[1]#b_rand.shape[0]
wavelet = 'db2'
mode = 'periodization'#['zero', 'constant', 'symmetric', 'periodic', 'smooth', 'periodization', 'reflect', 'antisymmetric', 'antireflect']
Psi = WaveletMatrix(wavelet,N)
ar = Psi.T@Psi
#print("Is it diagonal? ", np.array_equal(ar, np.diag(np.diag(ar))))
#identity = np.array(np.identity((ar.shape[0])),dtype=float)
#print('Psi@Psi.T is the identity: ')
#ar[ar<1e-15]=0
#print(ar==identity)

#print("Is it diagonal? ", np.array_equal(ar, np.diag(np.diag(ar))))
#fils, cols = ar.shape

#ar_round = np.round(ar) 
per_100 = 1
A_100, b_100 = UnderSample_frequency(r_c,riel_p,R_t,Nf,Np,per_100,Ski,f)
Ap_100 = LA.pinv(A_100) 
S2_100 = Ap_100.dot(b_100) 
R_100 = np.reshape(S2_100,(len(y_c),len(x_c)))

#############################################################################################3


wavelets = ['db2','haar','sym2']

#['db2','db4','haar','coif2','sym2']
#['db2', 'coif2']#,'coif2']
#['db2','db4','haar','coif2','sym2']

list_err_mag = []
list_err_phase = []
list_psnr = []
list_err_mag_spgl1 = []
list_err_phase_spgl1 = []
list_psnr_spgl1 = []

list_err_mag_std = []
list_err_phase_std = []
list_psnr_std = []
list_err_mag_spgl1_std = []
list_err_phase_spgl1_std = []
list_psnr_spgl1_std = []

cut_off = 0.5
#print(Ski.shape)
fils,cols = Ski.shape
numRealizations = 10#1#10
#noise = np.random.normal(scale=1,size=(2,2))
#noise_re = np.random.rand(fils,cols)
#noise_im = np.random.rand(fils,cols)
#noise = n0*(noise_re+1j*noise_im)
D2 = np.linalg.norm(Ski)**2/(Ski.shape[0]*Ski.shape[1])
print(D2)
n0 = np.sqrt(D2/SNR)#*2**(-1)
sigma = n0
#sigma = np.sqrt(n0**2*(Ski_noisy.shape[0]*Ski_noisy.shape[1]))
print(n0)
print(Ski.shape)
fils,cols = Ski.shape
for wv in wavelets:
    print("*************************************************")
    print('Wavelet: ',wv)
    mag_err_temp = []
    phase_err_temp = []
    psnr_temp = []
    spgl1_mag_err_temp = []
    spgl1_phase_err_temp = []
    spgl1_psnr_temp = []
    for i in range(numRealizations):
        print("Realization number: ", i)
        noise = n0*(np.random.rand(fils,cols)+1j*np.random.rand(fils,cols))/np.sqrt(2)
        Ski_noisy = Ski + noise
        sigma = n0
        print("x_c: ", x_c)
        mag_err, phase_err,psnr_vector,spgl1_mag_err, spgl1_phase_err,spgl1_psnr_vector = GetErrorsBoth(x_c,y_c,r_c,riel_p,R_t,Np,per,Ski_noisy,wv,f,sigma,R_100,I_sim,pers)
        #spgl1_mag_err, spgl1_phase_err,spgl1_psnr_vector =  GetErrorsSPGL1(x_c,y_c,r_c,riel_p,R_t,Nf,Np,per,Ski_noisy,wv,f,sigma,R_100, I_sim, cut_off, str_target,pers)
        #mag_err, phase_err,psnr_vector =  GetErrors(x_c,y_c,r_c,riel_p,R_t,Nf,Np,per,Ski_noisy,wv,f,sigma**2,R_100,I_sim,cut_off, str_target,pers)
#        spgl1_mag_err, spgl1_phase_err,spgl1_psnr_vector =  GetErrorsSPGL1(x_c,y_c,r_c,riel_p,R_t,Nf,Np,per,Ski_noisy,wv,f,sigma,R_100, I_sim, cut_off, str_target)
        mag_err_temp.append(mag_err)
        phase_err_temp.append(phase_err)
        psnr_temp.append(psnr_vector)
        spgl1_mag_err_temp.append(spgl1_mag_err)
        spgl1_phase_err_temp.append(spgl1_phase_err)
        spgl1_psnr_temp.append(spgl1_psnr_vector)
        print("Phase errors inside the loop: ",phase_err)
        print("CPU usage (%):", psutil.cpu_percent())
        del Ski_noisy, noise
        gc.collect()
    mag_err_avg = np.nanmean(mag_err_temp,axis=0)
    phase_err_avg = np.nanmean(phase_err_temp,axis=0)
    psnr_vector_avg = np.nanmean(psnr_temp,axis=0)
    spgl1_mag_err_avg = np.nanmean(spgl1_mag_err_temp,axis=0)
    spgl1_phase_err_avg = np.nanmean(spgl1_phase_err_temp,axis=0)
    spgl1_psnr_vector_avg = np.nanmean(spgl1_psnr_temp,axis=0)
    ###Faltan agregar vectores de std
    mag_err_std = np.nanstd(mag_err_temp,axis=0)
    phase_err_std = np.nanstd(phase_err_temp,axis=0)
    psnr_vector_std = np.nanstd(psnr_temp,axis=0)
    spgl1_mag_err_std = np.nanstd(spgl1_mag_err_temp,axis=0)
    spgl1_phase_err_std = np.nanstd(spgl1_phase_err_temp,axis=0)
    spgl1_psnr_vector_std = np.nanstd(spgl1_psnr_temp,axis=0)
    #######################################################
    print("Average for magnitude, phase error and psnr (OMP): ", mag_err_avg, phase_err_avg, psnr_vector_avg)
    print("Std for magnitude, phase error and psnr (OMP): ", mag_err_std, phase_err_std, psnr_vector_std)
    
    print("Average for magnitude, phase error and psnr (BPDN): ", spgl1_mag_err_avg, spgl1_phase_err_avg, spgl1_psnr_vector_avg)
    print("Std for magnitude, phase error and psnr (BPDN): ", spgl1_mag_err_std, spgl1_phase_err_std, spgl1_psnr_vector_std)
    
    list_err_mag.append(mag_err_avg)
    list_err_phase.append(phase_err_avg)
    list_psnr.append(psnr_vector_avg)
    list_err_mag_spgl1.append(spgl1_mag_err_avg)
    list_err_phase_spgl1.append(spgl1_phase_err_avg)
    list_psnr_spgl1.append(spgl1_psnr_vector_avg)
    
    list_err_mag_std.append(mag_err_std)
    list_err_phase_std.append(phase_err_std)
    list_psnr_std.append(psnr_vector_std)
    list_err_mag_spgl1_std.append(spgl1_mag_err_std)
    list_err_phase_spgl1_std.append(spgl1_phase_err_std)
    list_psnr_spgl1_std.append(spgl1_psnr_vector_std)

fontsize=20
fig, ax = plt.subplots(figsize=(12,6))
#pers = np.linspace(5,100,20)/100
#pers = np.linspace(5,100,2)/100
for i in range(len(wavelets)):
    #if i!=2:
    #    continue
    ax.errorbar(10*np.log10(pers),list_err_mag[i],yerr=list_err_mag_std[i],label=wavelets[i],marker='o')
    #ax.set_xlabel('%% of data',fontsize=fontsize)
    ax.set_xlabel(r'$10log_{10}(M/N)$',fontsize=fontsize)
    ax.set_ylabel('RMSE',fontsize=fontsize)
    ax.set_title('RMSE - Magnitude (OMP) - Target %s (SNR=%d dB)' % (str_target.upper(),SNR_db),fontsize=fontsize)
    ax.legend(fontsize=fontsize)
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)
plt.savefig('us-position-target-%s-rmse-magnitude-omp-different-basis-with-errors-snr=%d.png' % (str_target,SNR_db),bbox_inches='tight')    
plt.close(fig)
fig, ax = plt.subplots(figsize=(12,6))
for i in range(len(wavelets)):
    ax.errorbar(np.log10(pers),list_err_phase[i],yerr=list_err_phase_std[i],label=wavelets[i],marker='o')
    #ax.set_xlabel('%% of data',fontsize=fontsize)
    ax.set_xlabel(r'$10log_{10}(M/N)$',fontsize=fontsize)
    ax.set_ylabel('RMSE',fontsize=fontsize)
    ax.set_title('RMSE - Phase (OMP) - Target %s (SNR=%d dB)' % (str_target.upper(),SNR_db),fontsize=fontsize)
    ax.legend(fontsize=fontsize)
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)
plt.savefig('us-position-target-%s-rmse-phase-omp-different-basis-with-errors-snr=%d.png' % (str_target.upper(),SNR_db),bbox_inches='tight')    
plt.close(fig)
fig, ax = plt.subplots(figsize=(12,6))
for i in range(len(wavelets)):
    #if i==2:
    #    continue
    ax.errorbar(np.log10(pers),list_psnr[i],yerr=list_psnr_std[i],label=wavelets[i],marker='o')
    #ax.set_xlabel('%% of data',fontsize=fontsize)
    ax.set_xlabel(r'$10log_{10}(M/N)$',fontsize=fontsize)
    ax.set_ylabel('PSNR (dB)',fontsize=fontsize)
    ax.set_title('PSNR (OMP) - Target %s (SNR=%d dB)' % (str_target.upper(),SNR_db),fontsize=fontsize)
    ax.legend(fontsize=fontsize)
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)
#ax.plot(x_fit, y_fit,label='linear fit',lw=4,marker='v')
ax.legend(fontsize=fontsize)
plt.savefig('us-position-target-%s-psnr-phase-omp-different-basis-with-errors-snr=%d.png' % (str_target,SNR_db),bbox_inches='tight')    
plt.close(fig)

fontsize=20
fig, ax = plt.subplots(figsize=(12,6))
for i in range(len(wavelets)):
    ax.errorbar(np.log10(pers),list_err_mag_spgl1[i],yerr=list_err_mag_spgl1_std[i],label=wavelets[i],marker='o')
    #ax.set_xlabel('%% of data',fontsize=fontsize)
    ax.set_xlabel(r'$log_{10}(M/N)$',fontsize=fontsize)
    ax.set_ylabel('RMSE',fontsize=fontsize)
    ax.set_title('RMSE - Magnitude (BPDN) - Target %s (SNR=%d dB)' % (str_target.upper(),SNR_db),fontsize=fontsize)
    ax.legend(fontsize=fontsize)
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)
plt.savefig('us-position-target-%s-rmse-magnitude-spgl1-different-basis-with-errors-snr=%d.png' % (str_target,SNR_db),bbox_inches='tight')    
plt.close(fig)

fig, ax = plt.subplots(figsize=(12,6))
for i in range(len(wavelets)):
    ax.errorbar(np.log10(pers),list_err_phase_spgl1[i],yerr=list_err_phase_spgl1_std[i],label=wavelets[i],marker='o')
    #ax.set_xlabel('%% of data',fontsize=fontsize)
    ax.set_xlabel(r'$log_{10}(M/N)$',fontsize=fontsize)
    ax.set_ylabel('RMSE',fontsize=fontsize)
    ax.set_title('RMSE - Phase (BPDN) - Target %s (SNR=%d dB)' % (str_target.upper(), SNR_db),fontsize=fontsize)
    ax.legend(fontsize=fontsize,loc='best')
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)
plt.savefig('us-position-target-%s-rmse-phase-spgl1-different-basis-with-errors-snr=%d.png' % (str_target,SNR_db),bbox_inches='tight')    
plt.close(fig)
fig, ax = plt.subplots(figsize=(12,6))
for i in range(len(wavelets)):
    #if i==2:
    #    continue
    ax.errorbar(np.log10(pers),list_psnr_spgl1[i],yerr=list_psnr_spgl1_std[i],label=wavelets[i],marker='o')
    #ax.set_xlabel('%% of data',fontsize=fontsize)
    ax.set_xlabel(r'$log_{10}(M/N)$',fontsize=fontsize)
    ax.set_ylabel('PSNR',fontsize=fontsize)
    ax.set_title('PSNR (BPDN) - Target %s (SNR=%d dB)' % (str_target.upper(), SNR_db),fontsize=fontsize)
    #ax.set_title('PSNR (BPDN)',fontsize=fontsize)
    ax.legend(fontsize=fontsize)
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)
plt.savefig('us-position-target-%s-psnr-phase-spgl1-different-basis-with-errors-snr=%d.png'%(str_target,SNR_db),bbox_inches='tight')    
plt.close(fig)

#############################################################################
### Saving arrays##########################################3
#['db2','haar','sym2']
#h5f = h5py.File('freq-db2-haar-sym2-arrays-snr=%d-db.hdf5' % SNR_db, 'w')
print(str_name_hdf5)
h5f = h5py.File(str_name_hdf5,'w')
h5f.create_dataset('percentage', data=pers)
for i in range(len(wavelets)):
    str_module_omp = 'rmse-omp-module-%s' % wavelets[i]
    str_error_module_omp = 'rmse-omp-error-module-%s' % wavelets[i]
    str_phase_omp = 'rmse-omp-phase-%s' % wavelets[i]
    str_error_phase_omp = 'rmse-omp-error-phase-%s' % wavelets[i]
    str_psnr_omp = 'psnr-omp-%s' % wavelets[i]
    str_error_psnr_omp = 'error-psnr-omp-%s' % wavelets[i]
    #list_err_mag[i],yerr=list_err_mag_std[i],label=wavelets[i]
    h5f.create_dataset(str_module_omp, data=list_err_mag[i])
    h5f.create_dataset(str_error_module_omp, data=list_err_mag_std[i])
    #list_err_phase[i],yerr=list_err_phase_std[i],label=wavelets[i]
    h5f.create_dataset(str_phase_omp, data=list_err_phase[i])
    h5f.create_dataset(str_error_phase_omp, data=list_err_phase_std[i])
    #list_psnr[i],yerr=list_psnr_std[i],label=wavelets[i]
    h5f.create_dataset(str_psnr_omp, data=list_psnr[i])
    h5f.create_dataset(str_error_psnr_omp, data=list_psnr_std[i])	
    str_module_spgl1 = 'rmse-spgl1-module-%s' % wavelets[i]
    str_error_module_spgl1 = 'rmse-spgl1-error-module-%s' % wavelets[i]
    str_phase_spgl1 = 'rmse-slpgl1-phase-%s' % wavelets[i]
    str_error_phase_spgl1 = 'rmse-spgl1-error-phase-%s' % wavelets[i]
    str_psnr_spgl1 = 'psnr-spgl1-%s' % wavelets[i]
    str_error_psnr_spgl1 = 'error-psnr-spgl1-%s' % wavelets[i]
    #list_err_mag_spgl1[i],yerr=list_err_mag_spgl1_std[i],label=wavelets[i]
    h5f.create_dataset(str_module_spgl1, data=list_err_mag_spgl1[i])
    h5f.create_dataset(str_error_module_spgl1, data=list_err_mag_spgl1_std[i])
    #list_err_phase_spgl1[i],yerr=list_err_phase_spgl1_std[i],label=wavelets[i]
    h5f.create_dataset(str_phase_spgl1, data=list_err_phase_spgl1[i])
    h5f.create_dataset(str_error_phase_spgl1, data=list_err_phase_spgl1_std[i])
    #list_psnr_spgl1[i],yerr=list_psnr_spgl1_std[i],label=wavelets[i]
    h5f.create_dataset(str_psnr_spgl1, data=list_psnr_spgl1[i])
    h5f.create_dataset(str_error_psnr_spgl1, data=list_psnr_spgl1_std[i])
h5f.close()

##############################################################################
