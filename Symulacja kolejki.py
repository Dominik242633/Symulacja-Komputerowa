from numpy import sqrt,exp,log,sin,cos,pi,array,zeros ,arange,linspace
from numpy.random import rand,randn,randint  
from itertools import product
import numpy as np
import random
from scipy.stats import ttest_ind
import scipy.stats as sc

###############################
### wariant 1
#kasa1
kasa1 = [0]
kolejka1 = []
#kasa2
kasa2 = [0]
kolejka2 = []

### wariant 2
#kasa
kasa3 = [0]
kolejka3 = []
#samoobs
kasa4 = [0]
kasa5 = [0]
kolejka4 = []

#symulacja
czas = 0
czas_max = 60*60*8
czas_a = 0
przybycie = 0

sredni_czas = 0 
obsluzeni = 0
###############################
wynik_lista = []
it = 0
### atrybuty
fi = [0.5, 0.9]
Tk = [60,120]
k = [1.1, 2]
lam = [70, 90]   # B
#Ts = Tk * k

random.seed(777)
np.random.seed(777)
scen = list(product([1,0],[1,0],[1,0],[1,0],))


def pref(fi): # rozklad rownomierny
    zal = random.uniform(0,1)
    if zal < fi:
        return 'k'
    else:
        return 'b'
   
def czasK(Tk):
    return Tk
def czasS(Tk, k):
    return Tk * k

def przyb(lam): # rozklad wykladniczy
    return np.random.exponential(lam)

#klient = [pref(fi[i[0]]), Tk[i[1]], Tk[i[1]]*k[i[2]], czas]
#przybycie = przyb(lam[i[3]])
####### funkcje

#1] kasa -> exit
def wyjscie(kasa):
    global sredni_czas, czas, obsluzeni
    sredni_czas += czas - kasa[1][3] 
    obsluzeni += 1
    kasa.pop(1)
  
#2] kolejka -> kasa
def do_kasy(kasa, kolejka, v):
    kasa.append(kolejka.pop(0))
    kasa[0] = kasa[1][v]
 
def dlugosc_kolejki(kasa, kolejka, z):
    x = 0
    for i in range(len(kasa)):
        x += kasa[i][0]
    for i in range(len(kolejka)):
        x += kolejka[i][1]
    return x/z

def czas_minus(czas_a, kasy):
    global przybycie
    przybycie -= czas_a
    for i in range(len(kasy)):
        if len(kasy[i]) == 2:
            kasy[i][0] -= czas_a

def min_czas(kasy, przybycie):
    m = []
    m.append(przybycie)
    for i in range(len(kasy)):
        if len(kasy[i]) ==2:
            m.append(kasy[i][0])
    return min(m)


for i in scen:
    
    for j in range(3):
        #symulacja
        czas = 0
        czas_a = 0
        przybycie = 0
        sredni_czas = 0 
        obsluzeni = 0
        ### wariant 1
        #kasa1
        kasa1 = [0]
        kolejka1 = []
        #kasa2
        kasa2 = [0]
        kolejka2 = []
        
        it += 1
        ####### wariant 1 (2 kasy normalne)
        while czas < czas_max:  
            czas_a = min_czas([kasa1, kasa2], przybycie)
            czas_minus(czas_a, [kasa1, kasa2])
            czas += czas_a
            if czas > czas_max:
                czas -= czas_a
                break
                
            f = True
            while f == True:
                f = False
                ### 3] przybycie do kolejki
                if przybycie == 0:
                    klient = [pref(fi[i[0]]), Tk[i[1]], Tk[i[1]]*k[i[2]], czas]
                    if dlugosc_kolejki([kasa1], kolejka1, 1) <= dlugosc_kolejki([kasa2], kolejka2, 1):
                        kolejka1.append(klient)
                    else:
                        kolejka2.append(klient)
                    przybycie = przyb(lam[i[3]])
                    #print(przybycie)
                    f = True
                ### 1] kasa -> exit
                if kasa1[0] == 0 and len(kasa1) == 2:
                    wyjscie(kasa1)
                    f = True
                if kasa2[0] == 0 and len(kasa2) == 2:
                    wyjscie(kasa2)
                    f = True  
                ### 2] kolejka -> kasa
                if len(kasa1) == 1 and bool(kolejka1):
                    do_kasy(kasa1, kolejka1, 1)
                    f = True     
                if len(kasa2) == 1 and bool(kolejka2):
                    do_kasy(kasa2, kolejka2, 1)
                    f = True
             
        wynik_lista.append(['1',fi[i[0]], Tk[i[1]], Tk[i[1]]*k[i[2]], lam[i[3]], obsluzeni, sredni_czas, sredni_czas/ obsluzeni , it])       
        print(it,'/96')
        
        
    for j in range(3):
        #symulacja
        czas = 0
        czas_a = 0
        przybycie = 0
        sredni_czas = 0 
        obsluzeni = 0
        
            
        ### wariant 2
        #kasa
        kasa3 = [0]
        kolejka3 = []
        #samoobs
        kasa4 = [0]
        kasa5 = [0]
        kolejka4 = []
            
        ####### wariant 2 (kasa normalna + 2 kasy samoobsługowe)   
        it += 1    
        while czas < czas_max:
                
            czas_a = min_czas([kasa3, kasa4, kasa5], przybycie)
            czas_minus(czas_a, [kasa3, kasa4,kasa5])
            czas += czas_a
            if czas > czas_max:
                czas -= czas_a
                break
            
                
            
            f = True
            while f == True:
                f = False
                ### 3] przybycie do kolejki
                if przybycie == 0:
                    klient = [pref(fi[i[0]]), Tk[i[1]], Tk[i[1]]*k[i[2]], czas]
                    if klient[0] == 'b':
                        kolejka3.append(klient)
                    elif klient[0] == 'k':
                        if dlugosc_kolejki([kasa3], kolejka3, 1) <= dlugosc_kolejki([kasa4, kasa5], kolejka4, 2):
                            kolejka3.append(klient)
                        else:
                            kolejka4.append(klient)
                    przybycie = przyb(lam[i[3]])
                    f = True
                ### 1] kasa -> exit
                if kasa3[0] == 0 and len(kasa3) == 2:
                    wyjscie(kasa3)
                    f = True
                if kasa4[0] == 0 and len(kasa4) == 2:
                    wyjscie(kasa4)
                    f = True
                if kasa5[0] == 0 and len(kasa5) == 2:
                    wyjscie(kasa5)
                    f = True
                ### 2] kolejka -> kasa
                if len(kasa3) == 1 and bool(kolejka3):
                    do_kasy(kasa3, kolejka3, 1)
                    f = True
                if len(kasa4) == 1 and bool(kolejka4):
                    do_kasy(kasa4, kolejka4, 2)
                    f = True
                if len(kasa5) == 1 and bool(kolejka4):
                    do_kasy(kasa5, kolejka4, 2)
                    f = True
                        
        wynik_lista.append(['2',fi[i[0]], Tk[i[1]], Tk[i[1]]*k[i[2]], lam[i[3]], obsluzeni, sredni_czas, sredni_czas/ obsluzeni , it])
        print(it,'/96')
    
        
for i in range(len(wynik_lista)):
    wn = wynik_lista
    if i%3 ==0:
        print('-'*3)
    print(wn[i][8],'|wariant:', wn[i][0], '|fi',wn[i][1], '|czasK:', wn[i][2], '|czasS:', wn[i][3], '|lam:', wn[i][4],'|obsl:',wn[i][5], '|czas:', wn[i][6] ,'|Q:', wn[i][7])

kryt_jak_n = []
kryt_jak_s = []

for i in range(len(wynik_lista)):
    if wn[i][0] == '1':
        kryt_jak_n.append(wn[i][7])
    else:
        kryt_jak_s.append(wn[i][7])

kryt_jak_n = np.array(kryt_jak_n)
kryt_jak_s = np.array(kryt_jak_s)

srednie_kryt_jak_n = np.mean(kryt_jak_n)
srednie_kryt_jak_s = np.mean(kryt_jak_s)

mediana_kryt_jak_n = np.median(kryt_jak_n)
mediana_kryt_jak_s = np.median(kryt_jak_s)

odch_st_kryt_jak_n = np.sqrt(np.var(kryt_jak_n))
odch_st_kryt_jak_s = np.sqrt(np.var(kryt_jak_s))

print('-'*10)
print("Dane dla 2 normalnych kolejek")
print('Średnia: '+ str(round(srednie_kryt_jak_n,2)))
print('Mediana: ' + str(round(mediana_kryt_jak_n,2)))
print("Wariancja : " + str(round(np.var(kryt_jak_n),2)))
print('Odchylenie standardowe: ' + str(round(odch_st_kryt_jak_n,2)))
print("Skośność : " + str(round(sc.skew(kryt_jak_n),2)))
print("Kurtoza : " + str(round(sc.kurtosis(kryt_jak_n),2)))
print('-'*10)
print("Dane dla 1 normalnej kolejki i 2 samoobsługowych")
print('Średnia: '+ str(round(srednie_kryt_jak_s,2)))
print('Mediana: ' + str(round(mediana_kryt_jak_s,2)))
print("Wariancja : " + str(round(np.var(kryt_jak_s),2)))
print('Odchylenie standardowe: ' + str(round(odch_st_kryt_jak_s,2)))
print("Skośność : " + str(round(sc.skew(kryt_jak_s),2)))
print("Kurtoza : " + str(round(sc.kurtosis(kryt_jak_s),2)))
print('-'*10)

# ttest, pValue = ttest_ind(kryt_jak_n,kryt_jak_s)
# print("pValue dla testu t-Studenta równości dwóch średnich: " + str(round(pValue,2)))
# if pValue < 0.05:
#     print("Można odrzucić hipotezę zerową, że średnie są równe, przy poziomie istotności 0.05")
# else:
#     print("Nie ma podstaw do odrzucenia hipotezy zerowej, że średnie są równe, przy poziomie istotności 0.05")
ttest, pValue = ttest_ind(kryt_jak_s, kryt_jak_n)
print("pValue dla testu t-Studenta: " + str(round(pValue,2)))
print("Wartość statystyki testowej dla testu t-Studenta: " + str(round(ttest,2)))
alpha = 0.05
if (ttest < 0) & (pValue/2 < alpha):
    print("Można odrzucić hipotezę zerową i przyjąć, że 1 normalna kolejka i 2 samoonsbługowe są lepsze od 2 normanych kolejek")
else:
    print("Nie ma podstaw do odrzucenia hipotezy zerowej")

############################# wykres

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


labels = ['']
w1 = [round(srednie_kryt_jak_n,2)]
w2 = [round(srednie_kryt_jak_s,2)]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, w1, width, label='2 normalne kasy')
rects2 = ax.bar(x + width/2, w2, width, label='1 kasa + 2 samoobsługowe')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Q')
ax.set_title('Wskaźnik Q dla wariantów')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='best', bbox_to_anchor=(1, -0.5, 0, 0.5))


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 0),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()

tk1_1 = 0
tk1_2 = 0
tk2_1 = 0
tk2_2 = 0

# Pomocnicze do testów
a60 = []
a120 = []
b60 = []
b120 = []


for i in range(0,len(wynik_lista)):
    if wynik_lista[i][0] == '1':
        if wynik_lista[i][2] == 120:
            tk1_1 += wynik_lista[i][7] # 1 wariant 120
            a120.append(wynik_lista[i][7])
        else:
            tk1_2 += wynik_lista[i][7] # 1 wariant 60
            a60.append(wynik_lista[i][7])
    else:
        if wynik_lista[i][2] == 120:
            tk2_1 += wynik_lista[i][7] # 2 wariant 120
            b120.append(wynik_lista[i][7])
        else:
            tk2_2 += wynik_lista[i][7] # 2 wariant 60
            b60.append(wynik_lista[i][7])

#tk1_1 = tk1_1 +tk1_2
#tk2_1 = tk2_1 +tk2_2

tk1_1 = tk1_1/24
tk1_2 = tk1_2/24
tk2_1 = tk2_1/24
tk2_2 = tk2_2/24


labels = ['Tk = 120', 'Tk = 60']
w1 = [round(tk1_1, 2), round(tk1_2, 2)]
w2 = [round(tk2_1, 2), round(tk2_2, 2)]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, w1, width, label='2 normalne kasy')
rects2 = ax.bar(x + width/2, w2, width, label='1 kasa + 2 samoobsługowe')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Q')
ax.set_title('Wskaźnik Q dla wariantów')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()



autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()


print('-'*10)
print("Dane dla 2 normalnych kolejek Tk = 60")
print('Średnia: '+ str(round(np.mean(a60),2)))
print('Mediana: ' + str(round(np.median(a60),2)))
print("Wariancja : " + str(round(np.var(a60),2)))
print('Odchylenie standardowe: ' + str(round(np.sqrt(np.var(a60)),2)))
print("Skośność : " + str(round(sc.skew(a60),2)))
print("Kurtoza : " + str(round(sc.kurtosis(a60),2)))
print('-'*10)
print("Dane dla 2 normalnych kolejek Tk = 120")
print('Średnia: '+ str(round(np.mean(a120),2)))
print('Mediana: ' + str(round(np.median(a120),2)))
print("Wariancja : " + str(round(np.var(a120),2)))
print('Odchylenie standardowe: ' + str(round(np.sqrt(np.var(a120)),2)))
print("Skośność : " + str(round(sc.skew(a120),2)))
print("Kurtoza : " + str(round(sc.kurtosis(a120),2)))
print('-'*10)
print("Dane dla 1 normalnej i 2 samoobsługowych kolejek Tk = 60")
print('Średnia: '+ str(round(np.mean(b60),2)))
print('Mediana: ' + str(round(np.median(b60),2)))
print("Wariancja : " + str(round(np.var(b60),2)))
print('Odchylenie standardowe: ' + str(round(np.sqrt(np.var(b60)),2)))
print("Skośność : " + str(round(sc.skew(b60),2)))
print("Kurtoza : " + str(round(sc.kurtosis(b60),2)))
print('-'*10)
print("Dane dla 1 normalnej i 2 samoobsługowych kolejek Tk = 120")
print('Średnia: '+ str(round(np.mean(b120),2)))
print('Mediana: ' + str(round(np.median(b120),2)))
print("Wariancja : " + str(round(np.var(b120),2)))
print('Odchylenie standardowe: ' + str(round(np.sqrt(np.var(b120)),2)))
print("Skośność : " + str(round(sc.skew(b120),2)))
print("Kurtoza : " + str(round(sc.kurtosis(b120),2)))
print('-'*10)

ttest, pValue = ttest_ind(a60, b60)
print("Tk = 60")
print("pValue dla testu t-Studenta: " + str(pValue))
print("Wartość statystyki testowej dla testu t-Studenta: " + str(round(ttest,2)))
alpha = 0.05
if (ttest < 0) & (pValue/2 < alpha):
    print("Można odrzucić hipotezę zerową i przyjąć, że 1 normalna kolejka i 2 samoonsbługowe są lepsze od 2 normanych kolejek")
else:
    print("Nie ma podstaw do odrzucenia hipotezy zerowej")

ttest, pValue = ttest_ind(a120, b120)
print("Tk = 120")
print("pValue dla testu t-Studenta: " + str(pValue))
print("Wartość statystyki testowej dla testu t-Studenta: " + str(round(ttest,2)))
alpha = 0.05
if (ttest < 0) & (pValue/2 < alpha):
    print("Można odrzucić hipotezę zerową i przyjąć, że 1 normalna kolejka i 2 samoonsbługowe są lepsze od 2 normanych kolejek")
else:
    print("Nie ma podstaw do odrzucenia hipotezy zerowej")