
# coding: utf-8

# In[1]:
# Robinson Pineda Arroyave
#Estephania Arenas Marulanda
# Se importan las librerias necesarias para la ejecucion de la libreria 
import numpy as np
import matplotlib.pyplot as plt
import LinearFIR# esta funcion fue suministrada por el docente 
from sklearn.metrics import  r2_score
from scipy import stats,signal


# In[2]:


def cargar(a,b):
    #Función que permite cargar el archivo .txt y seleccionar el canal con el que se desea trabajar
    # a= Nombre del archivo .txt que contiene los datos
    #b= Canal que se desea cargar
    data=np.loadtxt(a,delimiter=',',skiprows=6,usecols=[1,2,3,4,5,6,7,8]) #Se carga el archivo diciendo quien es el
    # separador de datos, cuantas filas se tienen de ecabezado y las columnas que se desean cargar.
    print(len(data)) #Se imprime la longitud de los datos
    a=data.shape 
    print(a)
    print(data)
    #Dependiendo del canal que se quiera se escoje alguna de las opciones en donde se toman los datos de toda la
    #columna seleccionada.
    if b== 0:
        canal=data[:,0]   
    elif b == 1:
        canal=data[:,1]
    elif b == 2:
        canal=data[:,2]
    elif b == 3:
        canal=data[:,3]
    elif b == 4:
        canal=data[:,4]
    elif b == 5:
        canal=data[:,5]
    elif b == 6:
        canal=data[:,6]
    elif b == 7:
        canal=data[:,7]  
    canal1_sinF=signal.detrend(canal) # funcion para quitar el off-set del canal cargado
    canal_filtrado=LinearFIR.eegfiltnew(canal1_sinF,250,1,50,0,0)#Se aplica un filtro pasa-banda de 1-50Hz de acuerdo con la teoria 
    return canal1_sinF,canal_filtrado #devuelve el canal sin aplicar filtro y el canal filtrado


# In[3]:


def epocas(c,fm,e):
    #c-canal
    #fm- frecuencia de muestreo
    #e-numero de epocas
    chanel=np.squeeze(c) #Se le quita la dimension de más al canal
    t=np.arange(0,(chanel.shape[0]/fm),1/fm) #Vector de tiempo
    x=round(max(t)) #x es el último valor del vector de tiempo
    s=x/e # División para tomar el valor de tiempo equivalente al número de épocas que el usuario desea
    nmuestras=((s*chanel.shape[0])/x) # Mediante una regla de tres, se determina el número de muestras equivalente a las épocas elegidas
    b=abs(nmuestras)-abs(int(nmuestras)) # Se saca la parte flotante del número de muestras, esto con el fin de saber si hay error al ingresar datos
    if (s>x) or (b!=0.0)or (type(e)!=int): # Se toman las restricciones tal que el número de muestras no sea entera, que las épocas sean incoherentes  y que la señal sea divida en un tiempo mayor al de la misma
        raise Exception ('Error, Ingresó datos inválidos.')
    else:    
        matriz=np.zeros((1,chanel.shape[0]))#Se Crea una matriz de ceros con la misma dimensión de la matriz original
        matriz[0,::]=chanel# Hacer que el vector quede de forma horizontal, donde se llena con los valores de la señal ingresada
        nmuestras=int(nmuestras)# Se convierte la señal como entera para que no se generen errores a la hora de recorrer el vector de tiempo
        d=int(chanel.shape[0]/nmuestras)#Se crea la variable d con el fin de obtener el número hasta el cual se va hacer ciclo for
        b=np.zeros((e,nmuestras))# Se crea el vector donde se almacenará la época divida
        for i in range(1,d+1): #Se hace el ciclo para llenar cada época 
            b[i-1,::]=matriz[0,nmuestras*(i-1):nmuestras*i:1] #Se llena la matriz, donde en las diferentes filas se ubican las epocas dividas
    print (b)
    return (b)


# In[4]:


def val_extr(c,b):
#Esta funcion arroja la matriz de épocas sin las que se hallan por el método de valores extremos
#c: Canal original
#b: Matriz de épocas
    mu=np.mean(c) # Se saca el promedio del canal
    ga=np.std(c) #Se saca la desviación estandar
    d=0
    a=0
    lista=[] # lista vacía que va a contener las épocas atípicas
    for j in range(0,len(np.array(b))): #for para recorrer el arreglo de la señal separada en épocas
        a=0
        z=0
        d=0
        l=np.array(b[j,::])#se toman todos los valores de la columna evaluada en el contador
        vc=np.squeeze(l) #Se elimina una dimensión
        for i in vc: # para evaluar la desviació estándar y el promedio de las épocas
            z=(vc[d]-mu)/ga # se evalua para encontrar las épocas atípicas
            d=d+1
            if (abs(z)>=3.5): #cuando se tenga un valor mayor a 3.5 se considera una época atípica
                a=1
        if a==1:
            lista.append(j+1) #se adiciona esa época a la lista
    print (lista)
    b=np.delete(b,lista,0) #se elimina del arreglo original
    b=b.ravel() #se convierte la matriz en vector para luego poder graficar
    return b # se devuelve el vector de épocas sin contener las atípicas


# In[5]:


def regr_lineal(b,fm):
    #b: Matriz de épocas
    #fm: frecuencia de muestreo
# Función que permite conocer las épocas atípicas por medio del método de regresión lineal
    x,y=b.shape # se toma las dimensiones de  la matriz
    t=np.arange(len(b.reshape((x*y))))*(1/fm) # se genera el vector de tiempo 
    t=t.reshape((x,y))#se le da una nueva forma al vector con las dimensiones de la matriz de muestra 
    for i in np.arange(x):#ciclo for que recorre las columnas de la matriz de muestra b 
        m,z,r,p,e=stats.linregress(t[i],b[i])#función que regresa pendiente, intercepto,coeficiente de correlacion,error estandar de coeficiente estimado 
            tendencia= m*t[i]+z
            b[i]=b[i]-tendencia
    b=b.ravel()        
    return b# me retorna la matriz con las tendencias lineales restadas


# In[6]:


def metod_curtosis(b,valor_min,valor_max):
    #b: Matriz de épocas
    #valor_min: valor mínimo
    #valor_max: valor máximo
#Función que permite hallar las épocas atípicas por el método de curtosis
    epocas_eliminar=np.array([])# Arreglo vacío para almacenar las épocas a eliminar
    x,y=b.shape # se toma la forma de la matriz de épocas
    for epoca in np.arange(x): #for para recorrer la matriz
        if (valor_min>stats.kurtosis(b[epoca]-np.mean(b[epoca]))>valor_max): # se evalua la condición del valor mínimo y máximo
            epocas_eliminar=np.concatenate((epocas_eliminar,[int(epoca)])) # se adicionan a la lista vacía

    eliminar=np.unique(epocas_eliminar)# se toman los valores unicos ordenados de la matriz 
    b=np.delete(b,epocas_eliminar,0)# se eliminan las épocas atípicas
    b=b.ravel() # se convierte en vector para graficar
    return b # devuelve la matriz con las epocas eliminadas 


# In[7]:


def Spectral_method(b,umbral,fs,canal):
    #b: Matriz de épocas
    # umbral:
    #fs: frecuencia de muestreo
    #canal:canal original
    x,y=b.shape # se toma la forma de la matriz de épocas
    f,pxx=signal.welch(canal,fs)# se le aplica filtro welch
    Prom_xcanal=np.mean(pxx) # se halla el promedio
    epocas_eliminar=np.array([]) #  Arreglo vacío para almacenar las épocas a eliminar
    for epoca in np.arange(x): #for para recorrer la matriz
        f,PxEpoca=signal.welch(b[0:epoca],fs)#se toma el espectro de potencia 
        if np.any(np.absolute(PxEpoca-Prom_xcanal)>umbral): # se evalua la condición para conocer las épocas atípicas
            epocas_eliminar=np.concatenate((epocas_eliminar,[int(epoca)])) # se adicionan a la lista vacía
            epocas_eliminar=np.unique(epocas_eliminar)
            b = np.delete(b,epocas_eliminar,0) # se eliminan las épocas atípicas
    b=b.ravel() # se convierte en vector para graficar
    return b # devuelve la matriz con las epocas eliminadas


# In[8]:


def showAll(b):
    #b: matriz de épocas
    print(type(b))
    #esta funcion grafica todas la epocas de la matriz ingresada por el usuario
    plt.figure(figsize=(15,10)) # tamaño de la figura
    for d in range (b.shape[0]):  # for para recorrer la matriz
        vector=np.array(b[d-1,::])
        vc=np.squeeze(vector)
        t=np.arange(0,((vc.shape[0])/250)+1,1/250)
        plt.subplot(int(b.shape[0]/(2)),3,d+1)
        plt.plot(t[0:len(vc)],vc)
        plt.title('Epoca '+str(d+1))
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Amplitud [uV]')
        plt.grid()
    plt.show()


# In[9]:


def show(b, d):
    #b: matriz de épocas
    #d: epoca que se quiere graficar
#Esta funcion graficará el numero de la epoca que el usuario desee
    vector=np.array(b[d-1,::])
    vc=np.squeeze(vector)
    t=np.arange(0,((vc.shape[0])/250)+1,1/250)
    plt.figure(figsize=(15,5))
    plt.plot(t[0:len(vc)],vc,'y')
    plt.title('Epoca '+str(d))
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud [uV]')
    plt.grid()
    plt.show() 


# In[10]:


def graficar(s,s1,fm):
    # s señal sin filtrar
    # s1 señal filtrada
    #fm frecuencia de muestreo
    s=np.squeeze(s) #Se le quita la dimension de más al canal
    s1=np.squeeze(s1) #Se le quita la dimension de más a la señal filtrada
    t=np.arange(0,((s.shape[0])/fm),(1/fm)) #Vector de tiempo
    plt.figure(figsize=(10,5))  #Tamaño de la grafica
    plt.subplot(2,1,1) #Para graficar 2 filas, 1 columna, en la posicion 1
    plt.plot(t[0:s.shape[0]:1],s[0:s.shape[0]:1]) #Se grafica el canal
    plt.xlabel('Tiempo [s]') #Nombre del eje x
    plt.ylabel('Amplitud [uV] ') #Nombre del eje y
    plt.title('Señal sin filtrar') #Titulo de la grafica
    plt.grid() #Cuadricula
    plt.figure(figsize=(10,5)) #Tamaño de la grafica
    plt.subplot(2,1,2) #Para graficar 2 filas, 1 columna, en la posicion 2
    plt.plot(t[0:s.shape[0]:1],s1[0:s.shape[0]:1],'m') #Se grafica la señal filtrada
    plt.xlabel('Tiempo [s]') #Nombre del eje x
    plt.ylabel('Amplitud [uV] ') #Nombre del eje y
    plt.title('Señal Filtrada') #Titulo de la grafica
    plt.grid() #Cuadricula
    plt.show() #Se muestra


# In[23]:


def filtro_welch(procesada,sin_procesar,fm) :
    # Función para graficar la densidad espectral de potencia de la señal procesada y sin procesar
    # recibe como argumentos la señal después de todo el proceso , la señal original y la frecuencia de muestreo
    f1,dep1=signal.welch(procesada,fm,nperseg=1024)# se calculan los dos periodogramas de welch
    f,dep=signal.welch(sin_procesar,250,nperseg=1024)
    plt.figure(figsize=(15,10))
    plt.subplot(2,2,2)# se grafica la señal 
    plt.semilogy(f1,dep1,'m');
    plt.xlabel('Frecuencia [Hz] ')
    plt.ylabel('PSD V**2/Hz ')
    plt.grid()
    plt.title('Periodograma de Welch canal filtrada ')
    plt.subplot(2,2,1)
    plt.semilogy(f,dep,'c');
    plt.xlabel('Frecuencia [Hz] ')
    plt.ylabel('PSD V**2/Hz ')
    plt.grid()
    plt.title('Periodograma de Welch canal sin filtrar ')


# In[12]:


c1,c2=cargar('3min.txt',1)


# In[14]:


b=epocas(c2,250,4)


# In[15]:


valextr=val_extr(6,b)


# In[16]:


regrLineal=regr_lineal(b,250)


# In[17]:


curtosis=metod_curtosis(b,1,3)


# In[18]:


spectral=Spectral_method(b,150,250,c2)


# In[19]:


showAll(b)


# In[20]:


show(b,3)


# In[21]:


plot=graficar(c1,c2,250)


# In[24]:


filtro=filtro_welch(valextr,c1,250)

