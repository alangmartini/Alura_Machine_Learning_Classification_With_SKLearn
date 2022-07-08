from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import pandas as pd
from sensitive import uri


dados = my_csv = pd.read_csv(uri)
#dados.head()

dicti = {
    "home": "iniciar",
    "how_it_works": "como_funciona",
    "contact": "contato",
    "bought": "se_comprou"
}
dados = dados.rename(columns = dicti)

x = dados[["iniciar", "como_funciona", "contato"]]
y = dados["se_comprou"]

print(dados.shape)

treino_x = x[:75] #to 75th element
treino_y = y[:75]
teste_x = x[75:]
teste_y = y[75:]
print(f"Training with  {len(treino_x)} and testing with {len(teste_x)}" )
model = LinearSVC()

model.fit(treino_x, treino_y)

predicao_y = model.predict(teste_x)

acc = accuracy_score(teste_y, predicao_y)
print(f"The accuracy is: {round(acc*100, 2)}" )
