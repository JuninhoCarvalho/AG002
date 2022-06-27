import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import os

def sep():
    print('\n'+'='*70+'\n')

data = pd.read_csv('tic-tac-toe.csv')

tree = DecisionTreeClassifier()

x = data.loc[:,['1', '2', '3', '4', '5', '6', '7', '8', '9']]
y = data['resultado']

x = x.replace(['x', 'b', 'o'], [1, -1, 0])

x_train, x_test, y_train, y_test = train_test_split(x.values, y.values, test_size=0.2, random_state=0)

tree.fit(x_train, y_train)

tree_predict = tree.predict(x_test)

tree_predict = np.array(tree_predict)

tree_score = accuracy_score(y_test, tree_predict)

print(f"\nPontuação Decision Tree:{tree_score}\n")

while(True):
    entrada = input('1. Jogo da velha\n2. Sair\n')

    if(entrada == '1'):
        jogada = input('Entre com o resultado de um jogo: ').split()

        jogada = pd.DataFrame(jogada).transpose()

        jogada = jogada.replace(['x', 'b', 'o'], [1, -1, 0]).to_numpy()

        resultado = tree.predict(jogada)

        if(resultado[0] == 'positivo'):
            print('\nResultado: x venceu')
        else:
            print('\nResultado: x perdeu') 
        
        sep()    

    if(entrada == '2'):
        break
