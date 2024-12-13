import csv
import pandas as pd


def sfere_koord(rows,i):
    # Tacke x,y,z za Sfere
    sfera_x = []
    sfera_y = []
    sfera_z = []
    br_deljivih_markera = 0
    br_praznih_x = 0
    for niz in rows[210:250]:
        print("niz:", niz)
        br_praznih_polja = 0
        for x in niz[2::3]:
            print("x:",x)
            if x == '':
                br_praznih_x = br_praznih_x + 1
            else:
                sfera_x.append(x)
        print("Broj praznih polja x:",br_praznih_x)
        print("Sfera x osa:",sfera_x)





    yield sfera_x
    yield sfera_y
    yield sfera_z

def print_sfere_koord(koord_x, koord_y, koord_z,sfera):
    print(f"Sfera {sfera}, x koord: ", koord_x)
    print(f"Sfera {sfera}, y koord: ", koord_y)
    print(f"Sfera {sfera}, z koord: ", koord_z)


def normalizacija_tacaka_x(sfera_koord_x,referenca):
    sfera_tacke_norm_x = []
    for i in sfera_koord_x:
        novi_x = (float)(i) - (float)(referenca[0])
        sfera_tacke_norm_x.append(novi_x)
    return sfera_tacke_norm_x

def normalizacija_tacaka_y(sfera_koord_y,referenca):
    sfera_tacke_norm_y = []
    for y in sfera_koord_y:
        novi_y = (float)(y) - (float)(referenca[1])
        sfera_tacke_norm_y.append(novi_y)
    return sfera_tacke_norm_y

def normalizacija_tacaka_z(sfera_koord_z,referenca):
    sfera_tacke_norm_z = []
    for z in sfera_koord_z:
        novi_z = (float)(z) - (float)(referenca[2])
        sfera_tacke_norm_z.append(novi_z)
    return sfera_tacke_norm_z

def print_tacke_norm(norm_x, norm_y, norm_z,sfera):
    print(f"Sfera {sfera}, x koord norm: ", norm_x)
    print(f"Sfera {sfera}, y koord norm: ", norm_y)
    print(f"Sfera {sfera}, z koord norm: ", norm_z)

# csv file name
filename = "RAW podaci vicon\Cet_1_08.csv"

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

# printing the field names
#print('Field names are:' + ', '.join(field for field in fields))

# printing first 5 rows
'''print('\nFirst 5 rows are:\n')
for row in rows[:5]:
    # parsing each column of a row
    for col in row:
        print("%10s" % col, end=" ")
    print('\n')'''

# uzimanje prve tacke sfere 1 za referentni koord pocetak
print("\nMarker 1 prva tacka x, y, z\n")
marker1_referenca = []
for row in rows[210:400]:
    for col in row[2:5]:
        #print(col, end=' ')
        marker1_referenca.append(col)
    #print('\n')


print(f"Referenca: x = {marker1_referenca[0]} , y = {marker1_referenca[1]}, z = {marker1_referenca[2]}")

# Tacke x,y,z za Sferu 1
sfera1_tacke_x, sfera1_tacke_y, sfera1_tacke_z = sfere_koord(rows,0)
print("Sfera1 x:",sfera1_tacke_x)
print(len(sfera1_tacke_x))
print("Sfera1 y:",sfera1_tacke_y)
print(len(sfera1_tacke_y))
print("Sfera1 z:",sfera1_tacke_z)
print(len(sfera1_tacke_z))



# normalizacija tacaka sfere 1
sfera1_norm_x = normalizacija_tacaka_x(sfera1_tacke_x,marker1_referenca)

sfera1_norm_y = normalizacija_tacaka_y(sfera1_tacke_y,marker1_referenca)

sfera1_norm_z = normalizacija_tacaka_z(sfera1_tacke_z,marker1_referenca)

# Tacke x,y,z za Sferu 2
sfera2_tacke_x, sfera2_tacke_y, sfera2_tacke_z = sfere_koord(rows,3)
print_sfere_koord(sfera2_tacke_x,sfera2_tacke_y,sfera2_tacke_z,2)

# normalizacija tacaka sfere 2
sfera2_norm_x = normalizacija_tacaka_x(sfera2_tacke_x,marker1_referenca)

sfera2_norm_y = normalizacija_tacaka_y(sfera2_tacke_y,marker1_referenca)

sfera2_norm_z = normalizacija_tacaka_z(sfera2_tacke_z,marker1_referenca)

print_tacke_norm(sfera2_norm_x,sfera2_norm_y,sfera2_norm_z,2)

# Tacke x,y,z za Sferu 3
sfera3_tacke_x, sfera3_tacke_y, sfera3_tacke_z = sfere_koord(rows,6)
print_sfere_koord(sfera3_tacke_x,sfera3_tacke_y,sfera3_tacke_z,3)

# normalizacija tacaka sfere 3
sfera3_norm_x = normalizacija_tacaka_x(sfera3_tacke_x,marker1_referenca)

sfera3_norm_y = normalizacija_tacaka_y(sfera3_tacke_y,marker1_referenca)

sfera3_norm_z = normalizacija_tacaka_z(sfera3_tacke_z,marker1_referenca)

#print_tacke_norm(sfera3_norm_x,sfera3_norm_y,sfera3_norm_z,3)

# Tacke x,y,z za Sferu 4
sfera4_tacke_x, sfera4_tacke_y, sfera4_tacke_z = sfere_koord(rows,9)
print_sfere_koord(sfera4_tacke_x,sfera4_tacke_y,sfera4_tacke_z,4)

# normalizacija tacaka sfere 4
sfera4_norm_x = normalizacija_tacaka_x(sfera4_tacke_x,marker1_referenca)

sfera4_norm_y = normalizacija_tacaka_y(sfera4_tacke_y,marker1_referenca)

sfera4_norm_z = normalizacija_tacaka_z(sfera4_tacke_z,marker1_referenca)

#print_tacke_norm(sfera4_norm_x,sfera4_norm_y,sfera4_norm_z,4)

# Tacke x,y,z za Sferu 5
sfera5_tacke_x, sfera5_tacke_y, sfera5_tacke_z = sfere_koord(rows,12)
print_sfere_koord(sfera5_tacke_x,sfera5_tacke_y,sfera5_tacke_z,5)

# normalizacija tacaka sfere 5
sfera5_norm_x = normalizacija_tacaka_x(sfera5_tacke_x,marker1_referenca)

sfera5_norm_y = normalizacija_tacaka_y(sfera5_tacke_y,marker1_referenca)

sfera5_norm_z = normalizacija_tacaka_z(sfera5_tacke_z,marker1_referenca)

#print_tacke_norm(sfera5_norm_x,sfera5_norm_y,sfera5_norm_z,5)

# Tacke x,y,z za Sferu 6
sfera6_tacke_x, sfera6_tacke_y, sfera6_tacke_z = sfere_koord(rows,15)
print_sfere_koord(sfera6_tacke_x,sfera6_tacke_y,sfera6_tacke_z,6)


# normalizacija tacaka sfere 6
sfera6_norm_x = normalizacija_tacaka_x(sfera6_tacke_x,marker1_referenca)

sfera6_norm_y = normalizacija_tacaka_y(sfera6_tacke_y,marker1_referenca)

sfera6_norm_z = normalizacija_tacaka_z(sfera6_tacke_z,marker1_referenca)

#print_tacke_norm(sfera6_norm_x,sfera6_norm_y,sfera6_norm_z,6)


# nov csv fajl sa normalizovanim koord po sferama
df1 = pd.DataFrame(list(zip(sfera1_norm_x,sfera1_norm_y,sfera1_norm_z)),columns=['S1 X osa','S1 Y osa','S1 Z osa'])
df2 = pd.DataFrame(list(zip(sfera2_norm_x,sfera2_norm_y,sfera2_norm_z)),columns=['S2 X osa','S2 Y osa','S2 Z osa'])
df3 = pd.DataFrame(list(zip(sfera3_norm_x,sfera3_norm_y,sfera3_norm_z)),columns=['S3 X osa','S3 Y osa','S3 Z osa'])
df4 = pd.DataFrame(list(zip(sfera4_norm_x,sfera4_norm_y,sfera4_norm_z)),columns=['S4 X osa','S4 Y osa','S4 Z osa'])
df5 = pd.DataFrame(list(zip(sfera5_norm_x,sfera5_norm_y,sfera5_norm_z)),columns=['S5 X osa','S5 Y osa','S5 Z osa'])
df6 = pd.DataFrame(list(zip(sfera6_norm_x,sfera6_norm_y,sfera6_norm_z)),columns=['S6 X osa','S6 Y osa','S6 Z osa'])
rez_sfere=pd.concat([df1,df2,df3,df4,df5,df6],axis=1)



rez_sfere.to_csv('norm_po sferama.csv',index=False,header=True)
#print(rez_sfere)



# nov csv fajl sa normalizovanim koord po osama
df_x = pd.DataFrame(list(zip(sfera1_norm_x,sfera2_norm_x,sfera3_norm_x,sfera4_norm_x,sfera5_norm_x,sfera6_norm_x))
                   ,columns=['S1 X osa','S2 X osa','S3 X osa','S4 X osa','S5 X osa','S6 X osa'])
df_y = pd.DataFrame(list(zip(sfera1_norm_y,sfera2_norm_y,sfera3_norm_y,sfera4_norm_y,sfera5_norm_y,sfera6_norm_y))
                   ,columns=['S1 Y osa','S2 Y osa','S3 Y osa','S4 Y osa','S5 Y osa','S6 Y osa'])
df_z = pd.DataFrame(list(zip(sfera1_norm_z,sfera2_norm_z,sfera3_norm_z,sfera4_norm_z,sfera5_norm_z,sfera6_norm_z))
                   ,columns=['S1 Z osa','S2 Z osa','S3 Z osa','S4 Z osa','S5 Z osa','S6 Z osa'])

rez_ose = pd.concat([df_x,df_y,df_z],axis=1)




rez_ose.to_csv('norm_po_osama.csv',index=False,header=True)
#print(rez_ose)


sfera1,sfera2, sfera3 = sfere_koord(rows,0)


