from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sensitive import uri

#The SEED is responsible for decreasing the randomness of the test, so it can be realiably reproduced.
SEED = 20


dados = my_csv = pd.read_csv(uri)
#Here we map the columns name to something that makes more sence to us, making it easier to be worked
map = {
    "home": "iniciar",
    "how_it_works": "como_funciona",
    "contact": "contato",
    "bought": "se_comprou"
}

dados = dados.rename(columns = map)


x = dados[["iniciar", "como_funciona", "contato"]]
y = dados["se_comprou"]

train_x, test_x, train_y, test_y = train_test_split(x, y, 
                                                    stratify = y, #Stratify is responsible for proportionally separating train_y and test_y, so it has the a close proportion of values
                                                    random_state = SEED, 
                                                    test_size = 0.25) 

print(f"Training with  {len(train_x)} and testing with {len(test_x)}%" )

model = LinearSVC()
model.fit(train_x, train_y)
predict_y = model.predict(test_x)

acc = accuracy_score(test_y, predict_y)
print(f"The accuracy is: {round(acc*100, 2)}%" )

print(train_y.value_counts())
print(test_y.value_counts())