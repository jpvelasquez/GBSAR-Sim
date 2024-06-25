from matplotlib.patches import Rectangle
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
from decimal import Decimal
############################################################################
rugosity = 0.1
target='elliptical'
SNR = 1e4 # 0dB
#1e4#40 #dB
SNR_dB = int(10*np.log10(SNR))
print(SNR_dB)
c = 0.3 #0.299792458 # Velocidad de la luz (x 1e9 m/s)
fc = 15 # Frecuencia Central(GHz)
BW = 0.6 # Ancho de banda(GHz)
Ls = 1.2 # 4 # 0.6 # Longitud del riel (m)
Ro = 0 # Constante debido al delay de los cables(m)
theta = 90 # Angulo azimuth de vision de la imagen final(grados sexagesimales E [0-90])
