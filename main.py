import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class Modelo():
    def __init__(self, cenario):
        self.cenario = cenario
        self.results = pd.DataFrame()
        pass

    def CarregarDataset(self, path):
        
        names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
        self.df = pd.read_csv(path, names=names)

    def TratamentoDeDados(self):
       
        sns.pairplot(self.df, hue="Species", markers=["o", "s", "D"])
        plt.savefig("análise.png")

        plt.figure(figsize=(6, 4))
        sns.countplot(x="Species", data=self.df)
        plt.xlabel("Espécies")
        plt.ylabel("Quantidade")
        plt.title("Distribuição das espécies")
        plt.savefig("distribuição.png")

        df_numeric = self.df.select_dtypes(include=[float, int])  # selecionr as colunas numericas
        plt.figure(figsize=(8, 6))
        sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title("Matriz de correlação")
        plt.savefig(f'correlação.png')

        #print("Visão geral do dataframe:")
        #print(self.df.head())

        #print("\nValores ausentes:")
        #print(self.df.isna().sum())

        if self.cenario == 2:
            self.df.drop_duplicates(inplace=True)
        elif self.cenario == 3:
            self.df.drop("PetalWidthCm", axis=1, inplace=True)
        elif self.cenario == 4:
            self.df.drop("PetalLengthCm", axis=1, inplace=True)
        elif self.cenario == 5:
            self.df.drop("SepalLengthCm", axis=1, inplace=True)
        elif self.cenario == 6:
            self.df.drop_duplicates(inplace=True)
            self.df.drop("PetalWidthCm", axis=1, inplace=True)
        elif self.cenario == 7:
            self.df.drop_duplicates(inplace=True)
            self.df.drop("PetalLengthCm", axis=1, inplace=True)
        elif self.cenario == 8:
            self.df.drop_duplicates(inplace=True)
            self.df.drop("SepalLengthCm", axis=1, inplace=True)
        
        pass

    def Treinamento(self):

        features = self.df.drop(columns=['Species'], axis=1)
        target = self.df['Species']

        features_train, features_test, target_train, target_test = train_test_split(features, 
                                                                                    target, 
                                                                                    test_size=0.3,
                                                                                    random_state=38)
        
        rf_model = RandomForestClassifier()
        rf_model.fit(features_train, target_train)
        rf_predictions = rf_model.predict(features_test)
        self.Teste(target_test, rf_predictions, 'Ramdom forest')
        
        svm_model = SVC()
        svm_model.fit(features_train, target_train)
        svm_predictions = svm_model.predict(features_test)
        self.Teste(target_test, svm_predictions, 'SVC')
        
        lr_model = LogisticRegression()
        lr_model.fit(features_train, target_train)
        lr_predictions = lr_model.predict(features_test)
        self.Teste(target_test, lr_predictions, 'Regressão logística')
        
        dt_model = DecisionTreeClassifier()
        dt_model.fit(features_train, target_train)
        dt_predictions = dt_model.predict(features_test)
        self.Teste(target_test, dt_predictions, 'Árvore de decisão')
        
        knn_model = KNeighborsClassifier(n_neighbors=8)
        knn_model.fit(features_train, target_train)
        knn_predictions = knn_model.predict(features_test)
        self.Teste(target_test, knn_predictions, 'KNeighbors')
        
        pass

    def Teste(self, target_test, predicoes, model_name="Modelo"):
        
        acuracia = accuracy_score(target_test, predicoes)
        precisao = precision_score(target_test, predicoes, average='weighted')
        recall = recall_score(target_test, predicoes, average='weighted')
        f1 = f1_score(target_test, predicoes, average='weighted')
        
        data = {
            "Modelo": [model_name],
            "Acurácia": [acuracia],
            "Precisão": [precisao],
            "Recall": [recall],
            "F1-score": [f1]
        }
        
        new_row = pd.DataFrame(data)

        self.results = pd.concat([self.results, new_row], ignore_index=True)
        
        plt.figure(figsize=(8, 6))
        matriz_confusao = confusion_matrix(target_test, predicoes)
        sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Matriz de Confusão - {model_name}")
        plt.xlabel("Predição")
        plt.ylabel("Real")
        plt.savefig(f'cenario{self.cenario}/confusão_{model_name}_cenario{self.cenario}')

    def Train(self):
        
        self.CarregarDataset("iris.data")  
        self.TratamentoDeDados()
        self.Treinamento()  

for x in range(8):
    if not os.path.isdir(f'cenario{x+1}'): os.mkdir(f'cenario{x+1}')
    
    modelo = Modelo(cenario=(x+1))
    modelo.Train()

    modelo.results.to_csv(f'cenario{x+1}/desempenho_modelos.csv', index=False)
