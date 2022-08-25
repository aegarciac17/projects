import numpy as np
import pandas as pd
import sklearn.metrics

#Pedimos al usuario que puntue las 10 series más vistas de netflix
usuario_puntuación=[]
series=['La casa de papel', 'Stranger Things','Los Bridgerton','El juego del calamar','Gambito de dama', 'The Witcher', 'Por 13 razones', 'Lupin', 'Sex Education', 'Cobra kai']
print('A continuación deberás puntuar 10 series del 0 al 10, siendo 0 horrible o que no la has visto y 10 que te parece una obra de arte.')
for i in series:
  punt=input(f"¿Del 0 al 10 como puntuarías {i}? ")

  #Nos aseguramos que el imput contiene el formato de dato adecuado
  while punt.replace('.','',1).isnumeric()==False or float(punt)<0.0 or float(punt)>10.0:
    punt=input(f"Por favor, puntúa {i} del 0 al 10 ")
  usuario_puntuación.append(punt)

#Creamos el diccionario con las puntuaciones del equipo y del usuario
nodd3r_punt={'Alberto':[8,8.5,np.nan,8.7,8.8,8.3,np.nan,9,7,8.3],
             'Agustín':[np.nan,7.5,np.nan,7.2,8.5,7.2,6.5,np.nan,6.8,5.7],
             'Alejandra':[7,9.2,6,6.5,9,np.nan,np.nan,8,8,np.nan],
             'Christian':[1,2,3,6,9,7,7.3,5.6,8.8,4.2],
             'Irene':[9,np.nan,np.nan,8,np.nan,np.nan,7.5,np.nan,np.nan,np.nan],
             'Yeraldine':[9,8,7,6,5,4,3,2,1,9],
             'Usuario':usuario_puntuación}

#Lo convertimos a dataframe, colocando las puntuaciones de cada serie por columnas 
nodd3r_punt=pd.DataFrame(nodd3r_punt).T
nodd3r_punt.columns=series


#Las series que no han sido vistas se le dan un valor de 0
nodd3r_punt=nodd3r_punt.fillna(0)
display(nodd3r_punt)
#Calculamos la matriz de similaridad mediante la distancia del coseno (puesto que devuelve valores entre 0 y 1)
sim_matrix=1-sklearn.metrics.pairwise.cosine_distances(nodd3r_punt)

#Buscamos al miembro del equipo que tiene mayor similitud con el usuario y lo mostramos
mas_similar=np.argmax(sim_matrix[-1][:-1])
resultado=nodd3r_punt.index[mas_similar]
print(f'Te pareces a: {resultado}')
