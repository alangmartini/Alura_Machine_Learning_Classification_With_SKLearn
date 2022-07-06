from sklearn.svm import LinearSVC

#Cada elemento 0, 1 diz se tal característica está presente ou não no animal

#corpo comprido
#pelagem laranja
#perna curta
#late

raposa1 =  [1, 1, 1, 0]
raposa2 =  [0, 1, 0, 0]
raposa3 =  [0, 0, 1, 1]

cachorro1 = [1, 1, 0, 1]
cachorro2 = [0, 0, 0, 1]
cachorro3 = [1, 0, 1, 1]

#Aqui em classes eu digo: os tres primeiros são raposa(1) e os tres ultimos são cachorros
#Pra que o sklearn pegue as características e associe-as
#A cada resultado

#1 raposa, 0 cachorro
train_x = [raposa1, raposa2, raposa3, cachorro1, cachorro2, cachorro3]
train_y = [1, 1, 1, 0, 0, 0] #labels

model = LinearSVC() 

#Ensinamos o sklearn
model.fit(train_x, train_y)


#Definimos 3 animais para ele "predizer" sua classe (cachorro ou raposa)
misterious1 = [1, 1, 1, 1]
misterious2 = [1, 0, 1, 0]
misterious3 = [0, 1, 0, 1]

#LinearSVC recebe um array de arrays.
test_x = [misterious1, misterious2, misterious3]
#What the mysterious animals really were
test_y = [0, 1, 0]

predicts = model.predict(test_x)

# A resposta são 1 e 0s (raposa e cachorro, respectivamente)
print(predicts)


#Testing accuracy ( manually )
comparacao = test_y == predicts
print(comparacao)

n_corretos = comparacao.sum()
test_size = len(test_x)
accuracy = n_corretos/test_size
print(f"A taxa de acerto é: {accuracy * 100}%")

from sklearn.metrics import accuracy_score

accuracy_sklearn = accuracy_score(test_x, predicts)
print(f"A taxa de acerto é: {accuracy_sklearn * 100}%")