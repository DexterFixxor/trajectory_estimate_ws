import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import path

MARKERI = 6
ALPHA=0.2



def sfera_x(rows,gr_1,gr_2):
    # Tacke x svih 6 sfera podeljene sa vidljivim brojem markera
    suma = 0
    sfera_x = []
    br_deljivih_markera = 0
    br_praznih_x = 0
    for niz in rows[gr_1:gr_2]:
        #print("niz:", niz[2:])
        br_praznih_x = 0
        suma = 0
        for x in niz[2::3]:
            #print("x:", x)
            if x == '':
                br_praznih_x = br_praznih_x + 1
            else:
                suma = suma + (float)(x)
                #print("suma:",suma)
        if br_praznih_x < 6:
            #print("Broj praznih polja:",br_praznih_x)
            br_deljivih_markera = MARKERI - br_praznih_x
            #print("Deljivi markeri:",br_deljivih_markera)
            rez = suma / br_deljivih_markera
            sfera_x.append(rez)
        #print("Broj praznih polja x:", br_praznih_x)
        #print("Sfera x osa:", sfera_x)

    return sfera_x

def sfera_y(rows,gr_1,gr_2):
    # Tacke y svih 6 sfera podeljene sa vidljivim brojem markera
    suma = 0
    sfera_y = []
    br_deljivih_markera = 0
    br_praznih_y = 0
    for niz in rows[gr_1:gr_2]:
        #print("niz:", niz[2:])
        br_praznih_y = 0
        suma = 0
        for y in niz[3::3]:
            #print("x:", x)
            if y == '':
                br_praznih_y = br_praznih_y + 1
            else:
                suma = suma + (float)(y)
                #print("suma:",suma)
        if br_praznih_y < 6:
            #print("Broj praznih polja:",br_praznih_x)
            br_deljivih_markera = MARKERI - br_praznih_y
            #print("Deljivi markeri:",br_deljivih_markera)
            rez = suma / br_deljivih_markera
            sfera_y.append(rez)
        #print("Broj praznih polja x:", br_praznih_x)
        #print("Sfera y osa:", sfera_y)

    return sfera_y

def sfera_z(rows,gr_1,gr_2):
    # Tacke z svih 6 sfera podeljene sa vidljivim brojem markera
    suma = 0
    sfera_z = []
    br_deljivih_markera = 0
    br_praznih_z = 0
    for niz in rows[gr_1:gr_2]:
        #print("niz:", niz[2:])
        br_praznih_z = 0
        suma = 0
        for z in niz[4::3]:
            #print("x:", x)
            if z == '':
                br_praznih_z = br_praznih_z + 1
            else:
                suma = suma + (float)(z)
                #print("suma:",suma)
        if br_praznih_z < 6:
            #print("Broj praznih polja:",br_praznih_x)
            br_deljivih_markera = MARKERI - br_praznih_z
            #print("Deljivi markeri:",br_deljivih_markera)
            rez = suma / br_deljivih_markera
            sfera_z.append(rez)
        #print("Broj praznih polja x:", br_praznih_x)
        #print("Sfera x osa:", sfera_x)

    return sfera_z


# Ucitavanje i sredjivanje RAW podataka

# ucitavanje file path-a
file_path = 'RAW podaci vicon' #input("Enter the file path: ") 
file_name = input("Enter the file name with extension (.csv) (RAW podaci): ")
full_path = path.join(file_path, file_name)

print(f"Full file address is: {full_path}")
# csv file name
filename = full_path

# initialzing the titles and rows list
fields = []
rows = []
# reading csv file
with open(filename, 'r') as csvfile:
    # creating a csv reader object
    csvreader = csv.reader(csvfile)

    # extracting field names through first row
    fields = next(csvreader)

    # extracting each data row one by one
    for row in csvreader:
        rows.append(row)

    # get total number of rows
    print("Total no. of rows: %d" % (csvreader.line_num))

granica_1 = int(input("Unesi prvu granicu iz excel-a za sfere:"))
granica_2 = int(input("Unesi drugu granicu iz excel-a za sfere:"))

usrednjena_sfera_x=sfera_x(rows,granica_1,granica_2)
print("Tacke x:",usrednjena_sfera_x)

usrednjena_sfera_y = sfera_y(rows,granica_1,granica_2)
print("Tacke y:",usrednjena_sfera_y)

usrednjena_sfera_z = sfera_z(rows,granica_1,granica_2)
print("Tacke z:",usrednjena_sfera_z)

# nov csv fajl sa usrednjenom vrednostima
df = pd.DataFrame(list(zip(usrednjena_sfera_x,usrednjena_sfera_y,usrednjena_sfera_z)),columns=['S1-6 X osa','S1-6 Y osa','S1-6 Z osa'])
df.to_csv('usr_'+ file_name,index=False,header=True)


# Ucitavanje Tx, Ty i Tz iz fajla
Tx = []
Ty = []
Tz = []
for niz in rows[6:200]:
    print("Niz:",niz[5:])
    for i in range(len(niz)):
        #print("i:",i)
        if i == 5:
            Tx.append((niz[i]))
        elif i == 6:
            Ty.append((niz[i]))
        elif i == 7:
            Tz.append((niz[i]))







print("Tx:",Tx)
print("Ty:",Ty)
print("Tz:",Tz)

res_x =[(float)(item) for item in Tx if item != '']
res_y = [(float)(item) for item in Ty if item != '']
res_z = [(float)(item) for item in Tz if item != '']
print("Tx:",res_x)
print("Ty:",res_y)
print("Tz:",res_z)


# data plotting

plt.axes(projection='3d')
plt.plot(usrednjena_sfera_x,usrednjena_sfera_y,usrednjena_sfera_z,'g',label='Usrednjene vredn.')
plt.plot(res_x,res_y,res_z,'r',label='Tx, Ty, i Tz')

plt.title(file_name + " trajektorije")
plt.legend()
plt.show()


# plotovanje samo Tx, Ty i Tz zasebno
def plot_Tx_Ty_Tz():
    frame = list(range(0,160))
    figure, axis = plt.subplots(3)

    # za Tx
    Tx = np.array(res_x)
    axis[0].plot(frame,Tx)
    axis[0].set_title("Tx")

    # za Ty
    Ty = np.array(res_y)
    axis[1].plot(frame,Ty)
    axis[1].set_title("Ty")

    # za Tz
    Tz = np.array(res_z)
    axis[2].plot(frame,Tz)
    axis[2].set_title("Tz")

    plt.show()