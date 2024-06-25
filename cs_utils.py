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
import inspect
from pprint import pprint
import warnings
warnings.filterwarnings('ignore')
import psutil

##############################################################
def get_process_info():
    processes = []
    for process in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info', 'create_time']):
        processes.append({'pid': process.info['pid'],'name': process.info['name'],'cpu_percent': process.info['cpu_percent'],'memory_info': process.info['memory_info'].rss,'create_time': process.info['create_time']})
        print("Process info: ", process.info['pid'], process.info['name'],  process.info['cpu_percent'],process.info['memory_info'].rss, process.info['create_time'])
    return processes
###############################################################
def GetMaskedArray(phase_40_bpdn, mag_40_bpdn,cut_off):
    cut = 10*np.log10(cut_off)
    mask = mag_40_bpdn>cut
    arr1 = np.ma.masked_array(phase_40_bpdn)
    arr1[mask] = np.ma.masked
    return arr1
###############################################################
def GetMaskedArray_v2(phase_40_bpdn, mag_40_bpdn,cut_off):
    mask = mag_40_bpdn>=cut_off
    arr1 = np.ma.masked_array(phase_40_bpdn)
    arr1[mask] = np.ma.masked
    return arr1
#    return phase_40_bpdn[mask]
##############################################################
def UnderSample_v1(r_c,riel_p,R_t,Nf,Np,per,Ski,f):
    Np_rand = int(per*Np)
    n = Np_rand  # for 2 random indices
    ind = np.random.choice(Np, n, replace=False)
    index = np.sort(ind)
    #print(Np_rand,Np,Nf)
    A_rand = np.zeros((Np_rand*Nf,len(r_c)), dtype=complex)
    riel_p_rand = riel_p[index]
    Ski_rand_aux = Ski[index].flatten()
    c = 0.3
    ks = 2*np.pi*f/c #
    #print('Dimensiones de I_t: ', I_t.shape)
    for n in range(len(r_c)):
        m = 0
        for s in range(len(f)):
            for l in range(len(riel_p_rand)):
                A_rand[m,n] = np.exp(-2j*ks[s]*np.abs(distance_nk(r_c[n],riel_p_rand[l])))
                m+=1
    Ski_r = Ski_rand_aux#np.array([sum(I_t[i]*np.exp(-1j*4*np.pi*fi*distance_nk(R_t[i],xi)/c) for i in range(len(I_t))) for xi in riel_p_rand for fi in f]) # Create a vector with value for each fi y ri
    Ski_rand = np.reshape(Ski_r,(Np_rand,Nf)) # Reshape the last vector Sr_f
    S1_rand = np.reshape(Ski_rand.T,(len(Ski_rand)*len(Ski_rand[0]),1))

    return A_rand, S1_rand, index

##############################################################
def UnderSample(r_c,riel_p,R_t,Nf,Np,per,Ski,f):
    Np_rand = int(per*Np)
    n = Np_rand  # for 2 random indices
    ind = np.random.choice(Np, n, replace=False)
    index = np.sort(ind)
    #print(Np_rand,Np,Nf)
    A_rand = np.zeros((Np_rand*Nf,len(r_c)), dtype=complex)
    riel_p_rand = riel_p[index]
    Ski_rand_aux = Ski[index].flatten()
    c = 0.3
    ks = 2*np.pi*f/c #
    #print('Dimensiones de I_t: ', I_t.shape)
    for n in range(len(r_c)):
        m = 0
        for s in range(len(f)):
            for l in range(len(riel_p_rand)):
                A_rand[m,n] = np.exp(-2j*ks[s]*np.abs(distance_nk(r_c[n],riel_p_rand[l])))
                m+=1
    Ski_r = Ski_rand_aux#np.array([sum(I_t[i]*np.exp(-1j*4*np.pi*fi*distance_nk(R_t[i],xi)/c) for i in range(len(I_t))) for xi in riel_p_rand for fi in f]) # Create a vector with value for each fi y ri
    Ski_rand = np.reshape(Ski_r,(Np_rand,Nf)) # Reshape the last vector Sr_f
    S1_rand = np.reshape(Ski_rand.T,(len(Ski_rand)*len(Ski_rand[0]),1))
    del Ski_rand
    return A_rand, S1_rand
################################################################
def UnderSample_frequency(r_c,riel_p,R_t,Nf,Np,per,Ski,f):
    Nf_rand = int(per*Nf)
    #n = Nf_rand  # for 2 random indices
    ind = np.random.choice(Nf, Nf_rand, replace=False)
    index = np.sort(ind)
    A_rand = np.zeros((Np*Nf_rand,len(r_c)), dtype=complex)
    f_rand = f[index]
    Ski_rand_aux = Ski[:,index].flatten()
    c = 0.3
    ks = 2*np.pi*f_rand/c #
    for n in range(len(r_c)):
        m = 0
        for s in range(len(f_rand)):
            for l in range(len(riel_p)):
                A_rand[m,n] = np.exp(-2j*ks[s]*np.abs(distance_nk(r_c[n],riel_p[l])))
                m+=1
    #Ski_r = Ski_rand_aux
    Ski_rand = np.reshape(Ski_rand_aux,(Np,Nf_rand)) # Reshape the last vector Sr_f
    S1_rand = np.reshape(Ski_rand.T,(len(Ski_rand)*len(Ski_rand[0]),1))

    return A_rand, S1_rand
################################################################
def UnderSample_mixed(r_c,riel_p,R_t,Nf,Np,per,Ski,f):
    per_aux = per**(1/2)
    Nf_rand = int(per_aux*Nf)
    Np_rand = int(per_aux*Np)
    ind_f = np.random.choice(Nf, Nf_rand, replace=False)
    ind_p = np.random.choice(Np, Np_rand, replace=False)
    index_f = np.sort(ind_f)
    index_p = np.sort(ind_p) 
    
    A_rand = np.zeros((Np_rand*Nf_rand,len(r_c)), dtype=complex)
    f_rand = f[index_f]
    riel_p_rand = riel_p[index_p]
    Ski_rand_aux0 = Ski[index_p,:]
    Ski_rand_aux = Ski_rand_aux0[:,index_f]
    c=0.3
    ks = 2*np.pi*f_rand/c 
    for n in range(len(r_c)):
        m = 0
        for s in range(len(f_rand)):
            for l in range(len(riel_p_rand)):
                A_rand[m,n] = np.exp(-2j*ks[s]*np.abs(distance_nk(r_c[n],riel_p_rand[l])))
                m+=1
    Ski_r = Ski_rand_aux.copy()
    Ski_rand = np.reshape(Ski_r,(Np_rand,Nf_rand)) # Reshape the last vector Sr_f
    S1_rand = np.reshape(Ski_rand.T,(len(Ski_rand)*len(Ski_rand[0]),1))
    del Ski_rand_aux0, Ski_rand_aux
    return A_rand, S1_rand
################################################################
def get_phaseH(prm, I_t, rt): # Parámetros, Vector de intensidades, vector de posiciones
    # Data
    c,fc,BW,Nf,Ls,Np = prm['c'],prm['fc'],prm['BW'],prm['Nf'],prm['Ls'],prm['Np']
    fi=fc-BW/2 # Frecuencia inferior(GHz)
    fs=fc+BW/2 # Frecuencia superior(GHz)
    
    Lista_f = np.linspace(fi, fs, Nf) #  Vector de frecuencias(GHz)
    Lista_pos = np.linspace(-Ls/2, Ls/2, Np) # Vector de posiciones del riel(m)
    print(Lista_f.shape, Lista_pos.shape)
    #-------------------SCATTERED SIGNAL---------------------#

    Sr_f = np.array([sum(I_t[i]*np.exp(-1j*4*np.pi*fi*distance_nk(rt[i],xi)/c)
        for i in range(len(I_t))) for xi in Lista_pos for fi in Lista_f]) # Create a vector with value for each fi y ri
    Sr_f = np.reshape(Sr_f,(Np,Nf)) # Reshape the last vector Sr_f

    return Sr_f
################################################################
#    noise = np.random.normal(scale=err,size=ellipse.shape)
#    noisy_ellipse = ellipse *(1+noise)
def get_phaseH_with_noise(prm, I_t, rt, noiseAmplitude): # Parámetros, Vector de intensidades, vector de posiciones
    # Data
    c,fc,BW,Nf,Ls,Np = prm['c'],prm['fc'],prm['BW'],prm['Nf'],prm['Ls'],prm['Np']
    fi=fc-BW/2 # Frecuencia inferior(GHz)
    fs=fc+BW/2 # Frecuencia superior(GHz)

    Lista_f = np.linspace(fi, fs, Nf) #  Vector de frecuencias(GHz)
    Lista_pos = np.linspace(-Ls/2, Ls/2, Np) # Vector de posiciones del riel(m)
    print("Phase history")
    print(Lista_f.shape, Lista_pos.shape)
    print("Size of arrays: ", Nf, Np)
    print("I_t, rt: ", I_t.shape, rt.shape)
    #-------------------SCATTERED SIGNAL---------------------#

    Sr_f = np.array([sum(I_t[i]*np.exp(-1j*4*np.pi*fi*distance_nk(rt[i],xi)/c)
        for i in range(len(I_t))) for xi in Lista_pos for fi in Lista_f]) # Create a vector with value for each fi y ri
    noise = np.random.normal(scale=noiseAmplitude,size=Sr_f.shape)
    Sr_f_noisy = Sr_f*(1+noise)
    Sr_f_noisy = np.reshape(Sr_f_noisy,(Np,Nf)) # Reshape the last vector Sr_f
    #noise = np.random.normal(scale=noiseAmplitude,size=Sr_f.shape)
    #noisy_ellipse = ellipse *(1+noise)

    return Sr_f_noisy

################################################################
# Distance vector between target and riel_k position in matrix target
def distance_nk(r_n, x_k): # punto "n", punto del riel "k"
    d=((r_n[0]-x_k)**2+(r_n[1])**2)**0.5
    return d
################################################################
def WaveletMatrix(wavelet,N):
    #n2 = np.log2(N)
    H = np.zeros((N,N))
    for n in range(N):
        v1 = np.zeros((N))
        v1[n] = 1
        vaux = np.block(pywt.wavedec(v1,wavelet,mode='periodization'))
        diff = vaux.shape[0]-v1.shape[0]
        #print("diff, vaux :", diff, vaux.shape)
        if diff==0:
            v2 = vaux
        else:
            v2 = vaux[0:-diff]
        #print("####################################")
        #print("Dentro de la función: ", v1.shape, v2.shape)
        #print("####################################")
        H[:,n] = v2
    return H.T
################################################################
def GetSPGL1Solution(A_rand, b_rand, sigma,wavelet,x_c,y_c):
    #sigma=1e-8#
    N = A_rand.shape[1]
    Psi = WaveletMatrix(wavelet,N)
    A_i = np.imag(np.dot(A_rand,Psi))
    A_r = np.real(np.dot(A_rand,Psi))
    b_i = np.imag(b_rand)
    b_r = np.real(b_rand)
    Aug = np.block([[A_r, -A_i],[A_i, A_r]])
    baug = np.concatenate((b_r,b_i),axis=0)
    print("BPDN: ", Aug.shape, baug.shape)
    xaug,raug,gaug,info = spgl1.spg_bpdn(Aug, baug.flatten(), sigma)
    x = xaug[0:N] + 1j*xaug[N:]
    x_mp = Psi@x
    x_f = np.reshape(x_mp,(len(y_c),len(x_c)))
    del A_i, A_r, b_i, b_r, Aug, baug,Psi
    del xaug,raug,gaug,info
    del x_mp, x
    return x_f
################################################################
def GetOMPSolution(A_rand, b_rand, sigma,wavelet, x_c, y_c):
    N = A_rand.shape[1]
    Psi = WaveletMatrix(wavelet,N)
    A_i = np.imag(np.dot(A_rand,Psi))
    A_r = np.real(np.dot(A_rand,Psi))
    b_i = np.imag(b_rand)
    b_r = np.real(b_rand)
    Aug = np.block([[A_r, -A_i],[A_i, A_r]])
    baug = np.concatenate((b_r,b_i),axis=0)
    print("OMP: ", Aug.shape, baug.shape)
    c=orthogonal_mp(Aug,baug,tol=sigma)
    
    c_mp = c[0:N] + 1j*c[N:]
    x_mp = Psi@c_mp
    x_mp = np.reshape(x_mp,(len(y_c),len(x_c)))
    del A_i, A_r, b_i, b_r, Aug, baug,Psi
    del c,c_mp

    return x_mp
################################################################
def PSNR(R_99,I):
    mse = np.mean(abs(R_99 - I) ** 2)
    max_pixel = np.max(np.abs(I))
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr
################################################################
def PlotReco(x_c, y_c,x_40_omp, sigma2, per,str_alg, target, wavelet):
    fig, ax = plt.subplots(1,2,figsize=(12,6), layout='constrained')
    fontsize=18

#    clrs1 = ax[0].pcolormesh(x_c, y_c,20*np.log10(np.abs(x_40_omp)) ,cmap='jet')
    clrs1 = ax[0].pcolormesh(x_c, y_c,np.abs(x_40_omp) ,cmap='jet')
    box=ax[0].get_position()
    cb1=plt.colorbar(clrs1,ax=ax[0])#,cax=cbarax)
    ax[0].set_title('Magnitude ' ,fontsize=fontsize)
    cb1.set_label(r'dB', fontsize=fontsize)
    cb1.ax.tick_params(labelsize=fontsize-3)
    ax[0].xaxis.set_tick_params(labelsize=fontsize)
    ax[0].yaxis.set_tick_params(labelsize=fontsize)
#'''
    clrs2 = ax[1].pcolormesh(x_c, y_c,np.angle(x_40_omp) ,cmap='jet')
    box=ax[1].get_position()
    cb2=plt.colorbar(clrs2,ax=ax[1])#,cax=cbarax)
    ax[1].set_title('Phase ' ,fontsize=fontsize)
    cb2.set_label(r'rad', fontsize=fontsize)
    cb2.ax.tick_params(labelsize=fontsize-3)
    ax[1].xaxis.set_tick_params(labelsize=fontsize)
    ax[1].yaxis.set_tick_params(labelsize=fontsize)
    fig.suptitle(r'OMP - %d %% de datos - $\sigma$ = %f ' % (int(round(per*100)),sigma2)  ,fontsize=fontsize)
    plt.savefig('Reconstrucciones/reco-%s-%s-%d-porciento-%s.png' % (str_alg,target,round(per*100), wavelet),bbox_inches='tight')
    #fig.close
    plt.close(fig)
###############################################################
def GetErrorsBoth(x_c,y_c,r_c,riel_p,R_t,Np,per,Ski,wavelet,f,sigma,R_100,I_sim,pers):
    mode = 'periodization'
    mag_sim = np.abs(I_sim)
    phase_sim = np.angle(I_sim)
    mag_err_roj = []
    phase_err_roj = []
    psnr_vector = []
    mag_err_roj_spgl1 = []
    phase_err_roj_spgl1 = []
    psnr_vector_spgl1 = []
    Nf = len(f)
    str_alg = 'OMP'
    for per in pers:
        print('Percentage: ', per)
        A_rand, b_rand =  UnderSample(r_c,riel_p,R_t,Nf,Np,per,Ski,f)#,x_c, y_c)
        N = A_rand.shape[0]
        Im_rand = GetOMPSolution(A_rand, b_rand, sigma,wavelet,x_c,y_c)
        Im_rand_spgl1 = GetSPGL1Solution(A_rand, b_rand, sigma,wavelet,x_c,y_c)
        mag_rand = np.abs(Im_rand)#19*np.log10(np.abs(Im_rand))
        mag_rand_spgl1 = np.abs(Im_rand_spgl1)
        mag_sim = np.abs(I_sim)
        where_ind = np.where(mag_sim>=0.3)
        ########### Aplicar máscara sobre imagen simulada.
        phase_rand = np.angle(Im_rand)
        error = math.sqrt(mean_squared_error(mag_sim[where_ind],mag_rand[where_ind]))
        phase_error = math.sqrt(mean_squared_error(phase_sim[where_ind],phase_rand[where_ind]))
        #######################################################################################
        phase_rand_spgl1 = np.angle(Im_rand_spgl1)
        error_spgl1 = math.sqrt(mean_squared_error(mag_sim[where_ind],mag_rand_spgl1[where_ind]))
        phase_error_spgl1 = math.sqrt(mean_squared_error(phase_sim[where_ind],phase_rand_spgl1[where_ind]))
        psnr = PSNR(R_100,Im_rand)
        mag_err_roj.append(error)
        phase_err_roj.append(phase_error)
        psnr_vector.append(psnr)
        #################################33
        psnr_spgl1 = PSNR(R_100,Im_rand_spgl1)
        mag_err_roj_spgl1.append(error_spgl1)
        phase_err_roj_spgl1.append(phase_error_spgl1)
        psnr_vector_spgl1.append(psnr_spgl1)
        #####################################
        del A_rand, b_rand, Im_rand, Im_rand_spgl1
    return mag_err_roj, phase_err_roj,psnr_vector,mag_err_roj_spgl1, phase_err_roj_spgl1,psnr_vector_spgl1

#  mag_err_roj, phase_err_roj,psnr_vector,mag_err_roj_spgl1, phase_err_roj_spgl1,psnr_vector_spgl1 = GetErrorsBoth(x_c,y_c,r_c,riel_p,R_t,Nf,Np,per,Ski,wavelet,f,sigma,R_100,I_sim,cut_off, str_target,pers)
#############################################################
#GetErrorsBoth(x_c,y_c,r_c,riel_p,R_t,Nf,Np,per,Ski,wavelet,f,sigma,R_100,I_sim,cut_off, str_target,pers)
def GetErrors(x_c,y_c,r_c,riel_p,R_t,Nf,Np,per,Ski,wavelet,f,sigma,R_100,I_sim,cut_off, str_target,pers):
    mode = 'periodization'
    #['zero', 'constant', 'symmetric', 'periodic', 'smooth', 'periodization', 'reflect', 'antisymmetric', 'antireflect']
    mag_sim = np.abs(I_sim)
    phase_sim = np.angle(I_sim)
    mag_err_roj = []
    phase_err_roj = []
    psnr_vector = []
    #ks = 2*np.pi*f/c #
    str_alg = 'OMP'
    for per in pers:
        #UnderSample(r_c,riel_p,R_t,Nf,Np,per,Ski,f)
        print('Percentage: ', per)
        A_rand, b_rand =  UnderSample(r_c,riel_p,R_t,Nf,Np,per,Ski,f)#,x_c, y_c)
        N = A_rand.shape[0]
        Im_rand = GetOMPSolution(A_rand, b_rand, sigma,wavelet,x_c,y_c)
        mag_rand = np.abs(Im_rand)#19*np.log10(np.abs(Im_rand))
        #mask = mag_rand>=0.1
        mask1 = mag_rand<0.3
        where_ind = np.where(mag_rand>=0.3)
        ########### Aplicar máscara sobre imagen simulada.
        phase_rand = np.angle(Im_rand)
        #phase_rand =  GetMaskedArray(phase_rand, mag_rand,cut_off)
        error = math.sqrt(mean_squared_error(mag_sim,mag_rand))
        phase_error = math.sqrt(mean_squared_error(phase_sim[where_ind],phase_rand[where_ind]))
        #phase_error = math.sqrt(mean_squared_error(phase_sim[mask],phase_rand[mask]))

        psnr = PSNR(R_100,Im_rand)
        mag_err_roj.append(error)
        phase_err_roj.append(phase_error)
        psnr_vector.append(psnr)
        PlotReco(x_c, y_c,Im_rand, sigma, per,str_alg, str_target, wavelet)
        del A_rand, b_rand, x_c, y_c, Im_rand

    return mag_err_roj, phase_err_roj,psnr_vector
##############################################################################################
def GetErrorsFreq(x_c,y_c,r_c,riel_p,R_t,Nf,Np,per,Ski,wavelet,f,sigma,R_100,I_sim,cut_off, str_target,pers):
    mode = 'periodization'
    #['zero', 'constant', 'symmetric', 'periodic', 'smooth', 'periodization', 'reflect', 'antisymmetric', 'antireflect']
 #   sigma=0e-3#-18
    #pers = np.linspace(5,100,20)/100
   # pers = np.linspace(10,100,5)/100
    mag_sim = np.abs(I_sim)
    phase_sim = np.angle(I_sim)
    mag_err_roj = []
    phase_err_roj = []
    psnr_vector = []
    #ks = 2*np.pi*f/c #
    str_alg = 'OMP'
    for per in pers:
        #UnderSample(r_c,riel_p,R_t,Nf,Np,per,Ski,f)
        print('Percentage: ', per)
        A_rand, b_rand =  UnderSample_frequency(r_c,riel_p,R_t,Nf,Np,per,Ski,f)#,x_c, y_c)
        N = A_rand.shape[0]
        Im_rand = GetOMPSolution(A_rand, b_rand, sigma,wavelet,x_c,y_c)
        mag_rand = np.abs(Im_rand)#19*np.log10(np.abs(Im_rand))
        #mask = mag_rand>=0.1
#        mask1 = mag_ran<0.3
        where_ind = np.where(mag_sim>=0.3)
        ########### Aplicar máscara sobre imagen simulada.
        phase_rand = np.angle(Im_rand)
        #phase_rand =  GetMaskedArray(phase_rand, mag_rand,cut_off)
        error = math.sqrt(mean_squared_error(mag_sim,mag_rand))
        phase_error = math.sqrt(mean_squared_error(phase_sim[where_ind],phase_rand[where_ind]))
        #phase_error = math.sqrt(mean_squared_error(phase_sim[mask],phase_rand[mask]))

        psnr = PSNR(R_100,Im_rand)
        mag_err_roj.append(error)
        phase_err_roj.append(phase_error)
        psnr_vector.append(psnr)
        PlotReco(x_c, y_c,Im_rand, sigma, per,str_alg, str_target, wavelet)
    return mag_err_roj, phase_err_roj,psnr_vector

##############################################################################################
def GetErrorsFreqBoth(x_c,y_c,r_c,riel_p,R_t,Np,per,Ski,wavelet,f,sigma,R_100,I_sim,pers):
    mode = 'periodization'
    mag_sim = np.abs(I_sim)
    phase_sim = np.angle(I_sim)
    mag_err_roj = []
    phase_err_roj = []
    psnr_vector = []
    mag_err_roj_spgl1 = []
    phase_err_roj_spgl1 = []
    psnr_vector_spgl1 = []
    Nf = len(f)
    str_alg = 'OMP'
    for per in pers:
        print('Percentage: ', per)
        A_rand, b_rand =  UnderSample_frequency(r_c,riel_p,R_t,Nf,Np,per,Ski,f)#UnderSample(r_c,riel_p,R_t,Nf,Np,per,Ski,f)#,x_c, y_c)
        N = A_rand.shape[0]
        Im_rand = GetOMPSolution(A_rand, b_rand, sigma,wavelet,x_c,y_c)
        Im_rand_spgl1 = GetSPGL1Solution(A_rand, b_rand, sigma,wavelet,x_c,y_c)
        mag_rand = np.abs(Im_rand)#19*np.log10(np.abs(Im_rand))
        mag_rand_spgl1 = np.abs(Im_rand_spgl1)
        mag_sim = np.abs(I_sim)
        where_ind = np.where(mag_sim>=0.3)
        ########### Aplicar máscara sobre imagen simulada.
        phase_rand = np.angle(Im_rand)
        error = math.sqrt(mean_squared_error(mag_sim[where_ind],mag_rand[where_ind]))
        phase_error = math.sqrt(mean_squared_error(phase_sim[where_ind],phase_rand[where_ind]))
        #######################################################################################
        phase_rand_spgl1 = np.angle(Im_rand_spgl1)
        error_spgl1 = math.sqrt(mean_squared_error(mag_sim[where_ind],mag_rand_spgl1[where_ind]))
        phase_error_spgl1 = math.sqrt(mean_squared_error(phase_sim[where_ind],phase_rand_spgl1[where_ind]))
        psnr = PSNR(R_100,Im_rand)
        mag_err_roj.append(error)
        phase_err_roj.append(phase_error)
        psnr_vector.append(psnr)
        #################################33
        psnr_spgl1 = PSNR(R_100,Im_rand_spgl1)
        mag_err_roj_spgl1.append(error_spgl1)
        phase_err_roj_spgl1.append(phase_error_spgl1)
        psnr_vector_spgl1.append(psnr_spgl1)
        #####################################
        del A_rand, b_rand, Im_rand, Im_rand_spgl1
    return mag_err_roj, phase_err_roj,psnr_vector,mag_err_roj_spgl1, phase_err_roj_spgl1,psnr_vector_spgl1
###########################################################################################################
##############################################################################################
def GetErrorsMixedBoth(x_c,y_c,r_c,riel_p,R_t,Np,per,Ski,wavelet,f,sigma,R_100,I_sim,pers):
    mode = 'periodization'
    mag_sim = np.abs(I_sim)
    phase_sim = np.angle(I_sim)
    mag_err_roj = []
    phase_err_roj = []
    psnr_vector = []
    mag_err_roj_spgl1 = []
    phase_err_roj_spgl1 = []
    psnr_vector_spgl1 = []
    Nf = len(f)
    str_alg = 'OMP'
    for per in pers:
        print('Percentage: ', per)
        A_rand, b_rand =  UnderSample_mixed(r_c,riel_p,R_t,Nf,Np,per,Ski,f)#UnderSample(r_c,riel_p,R_t,Nf,Np,per,Ski,f)#,x_c, y_c)
        N = A_rand.shape[0]
        Im_rand = GetOMPSolution(A_rand, b_rand, sigma,wavelet,x_c,y_c)
        Im_rand_spgl1 = GetSPGL1Solution(A_rand, b_rand, sigma,wavelet,x_c,y_c)
        mag_rand = np.abs(Im_rand)#19*np.log10(np.abs(Im_rand))
        mag_rand_spgl1 = np.abs(Im_rand_spgl1)
        mag_sim = np.abs(I_sim)
        where_ind = np.where(mag_sim>=0.3)
        ########### Aplicar máscara sobre imagen simulada.
        phase_rand = np.angle(Im_rand)
        error = math.sqrt(mean_squared_error(mag_sim[where_ind],mag_rand[where_ind]))
        phase_error = math.sqrt(mean_squared_error(phase_sim[where_ind],phase_rand[where_ind]))
        #######################################################################################
        phase_rand_spgl1 = np.angle(Im_rand_spgl1)
        error_spgl1 = math.sqrt(mean_squared_error(mag_sim[where_ind],mag_rand_spgl1[where_ind]))
        phase_error_spgl1 = math.sqrt(mean_squared_error(phase_sim[where_ind],phase_rand_spgl1[where_ind]))
        psnr = PSNR(R_100,Im_rand)
        mag_err_roj.append(error)
        phase_err_roj.append(phase_error)
        psnr_vector.append(psnr)
        #################################33
        psnr_spgl1 = PSNR(R_100,Im_rand_spgl1)
        mag_err_roj_spgl1.append(error_spgl1)
        phase_err_roj_spgl1.append(phase_error_spgl1)
        psnr_vector_spgl1.append(psnr_spgl1)
        #####################################
        del A_rand, b_rand, Im_rand, Im_rand_spgl1
    return mag_err_roj, phase_err_roj,psnr_vector,mag_err_roj_spgl1, phase_err_roj_spgl1,psnr_vector_spgl1


###############################################################################################
def GetErrorsSPGL1(x_c,y_c,r_c,riel_p,R_t,Nf,Np,per,Ski,wavelet,f,sigma,R_100, I_sim,cut_offi, str_target,pers):
    mode = 'periodization'
#    sigma=1e-18
#    pers = np.linspace(5,100,20)/100
    #pers = np.linspace(10,100,5)/100
    mag_sim = np.abs(I_sim)
    phase_sim = np.angle(I_sim)
    mag_err_roj = []
    phase_err_roj = []
    #ks = 2*np.pi*f/c #
    psnr_vector = []
    str_alg = 'BPDN'
    for per in pers:
        print('Percentage: ', per)
        A_rand, b_rand =  UnderSample(r_c,riel_p,R_t,Nf,Np,per,Ski,f)
        #N = A_rand.shape[1]
        Im_rand = GetSPGL1Solution(A_rand, b_rand, sigma,wavelet,x_c,y_c)
        mag_rand = np.abs(Im_rand)#20*np.log10(np.abs(Im_rand))
        #mask = mag_rand>=0.1
        #mask1 = mag_rand<0.3
        where_ind = np.where(mag_sim>=0.3)
        phase_rand = np.angle(Im_rand)
        #phase_rand =  GetMaskedArray(phase_rand, mag_rand,cut_off)
        error = math.sqrt(mean_squared_error(mag_sim,mag_rand))
        phase_error = math.sqrt(mean_squared_error(phase_sim[where_ind],phase_rand[where_ind]))
#        phase_error = math.sqrt(mean_squared_error(phase_sim[mask],phase_rand[mask]))
#        phase_error = math.sqrt(mean_squared_error(phase_sim,phase_rand))
        psnr = PSNR(R_100,Im_rand)
        mag_err_roj.append(error)
        phase_err_roj.append(phase_error)
        psnr_vector.append(psnr)
        PlotReco(x_c, y_c,Im_rand, sigma, per,str_alg, str_target, wavelet)
    return mag_err_roj, phase_err_roj,psnr_vector
##############################################################################################
def GetErrorsSPGL1Freq(x_c,y_c,r_c,riel_p,R_t,Nf,Np,per,Ski,wavelet,f,sigma,R_100, I_sim,cut_offi, str_target,pers):
    mode = 'periodization'
#    sigma=1e-18
    #pers = np.linspace(5,100,20)/100
    #pers = np.linspace(10,100,5)/100
    mag_sim = np.abs(I_sim)
    phase_sim = np.angle(I_sim)
    mag_err_roj = []
    phase_err_roj = []
    #ks = 2*np.pi*f/c #
    psnr_vector = []
    str_alg = 'BPDN'
    for per in pers:
        print('Percentage: ', per)
        A_rand, b_rand =  UnderSample_frequency(r_c,riel_p,R_t,Nf,Np,per,Ski,f)
        #N = A_rand.shape[1]
        Im_rand = GetSPGL1Solution(A_rand, b_rand, sigma,wavelet,x_c,y_c)
        mag_rand = np.abs(Im_rand)#20*np.log10(np.abs(Im_rand))
        #mask = mag_rand>=0.1
        #mask1 = mag_rand<0.3
        where_ind = np.where(mag_sim>=0.3)
        phase_rand = np.angle(Im_rand)
        #phase_rand =  GetMaskedArray(phase_rand, mag_rand,cut_off)
        error = math.sqrt(mean_squared_error(mag_sim,mag_rand))
        phase_error = math.sqrt(mean_squared_error(phase_sim[where_ind],phase_rand[where_ind]))
#        phase_error = math.sqrt(mean_squared_error(phase_sim[mask],phase_rand[mask]))
#        phase_error = math.sqrt(mean_squared_error(phase_sim,phase_rand))
        psnr = PSNR(R_100,Im_rand)
        mag_err_roj.append(error)
        phase_err_roj.append(phase_error)
        psnr_vector.append(psnr)
        PlotReco(x_c, y_c,Im_rand, sigma, per,str_alg, str_target, wavelet)
    return mag_err_roj, phase_err_roj,psnr_vector

##############################################################################################
def RMSE_Artifacts(I_sim,I_rand,cut):

    mag_rang = np.abs(I_rand)
    module_sim = np.abs(I_sim)
    ### Región de interés para el 
    ### cálculo de las métricas
    where_ind = np.where(mag_rand>=cut)
    ### Región de interés para el 
    ### estudio de artifacts
    where_not = np.where(mag_rand<cut)
    ########### Aplicar máscara sobre imagen simulada.
    mse = mean_squared_error(module_sim[where_not],mag_rand[where_not])
    artifacts_error = math.sqrt(mse)
    
    return artifacts_error
##############################################################################################
def SelectTarget(target,err):

    if target == 'T':
        #w,h = 5,5
        I_t, R_t, x_c,y_c, N_r, N_c,w, h, dw, dh, str_target = get_matrix_data1()

    if target == 'ROJ':
        #w,h = 10,4
        I_t, R_t, x_c,y_c, N_r, N_c,w, h, dw, dh,  str_target = get_matrix_data3()

    if target == 'UNI':
       # w,h = 10,4
        I_t, R_t, x_c,y_c, N_r, N_c,w, h, dw, dh,  str_target = get_matrix_data5()

    if target == 'elliptical':
        #w,h = 600, 600
        I_t, R_t, x_c,y_c, N_r, N_c,w, h, dw, dh, str_target = GetEllipticalTarget(err)

    
    return I_t, R_t, x_c,y_c, N_r, N_c, w, h,dw,dh, str_target 
##############################################################################################
