#!/usr/bin/env python
# coding: utf-8

# # Previsão de Atraso em Ordens de Produção
# 
# **PCP** | Machine Learning com scikit-learn
# 
# Trabalhando com PCP a gente vive apagando incêndio: o atraso só aparece quando a ordem já devia estar pronta.
# A ideia deste projeto é inverter isso — usar o **histórico de ordens de produção** para treinar um modelo que lê as
# informações de uma ordem nova e estima se ela tende a sair **Adiantada**, **No_Prazo** ou **Atrasada**.
# 
# Com essa previsão em mãos dá tempo de agir: repriorizar a fila, negociar prazo, realocar operador ou garantir material antes do problema virar bola de neve.

# In[1]:


import pandas as pd

tabela = pd.read_csv("ordens_producao.csv")

print(f"{tabela.shape[0]} ordens no histórico, {tabela.shape[1]} colunas")
tabela.head()


# ## 1. Como está o desempenho hoje
# 
# Antes de qualquer modelo, vale entender o histórico: qual a proporção de atrasos e quais variáveis parecem separar
# uma ordem que cumpre prazo de uma que não cumpre.

# In[2]:


# distribuição do status no histórico (%)
(tabela["status_prazo"].value_counts(normalize=True) * 100).round(1)


# In[3]:


# média das variáveis por status — já dá pra ver o que puxa o atraso
tabela.groupby("status_prazo")[["quantidade_pecas", "dias_ate_prazo",
                                "operadores_alocados", "taxa_refugo_pct"]].mean().round(1)


# ## 2. Preparando os dados
# 
# O modelo só entende número, então preciso converter as colunas de texto (`produto`, `maquina`, `turno`,
# `prioridade`, `materia_prima_ok`). Uso o `LabelEncoder` e **guardo cada codificador** — as ordens novas vão
# precisar passar pela mesma conversão depois.
# 
# O `id_ordem` é só identificação, não ajuda a prever nada, então sai da análise.

# In[4]:


from sklearn.preprocessing import LabelEncoder

colunas_texto = ["produto", "maquina", "turno", "prioridade", "materia_prima_ok"]

codificadores = {}
for coluna in colunas_texto:
    cod = LabelEncoder()
    tabela[coluna] = cod.fit_transform(tabela[coluna])
    codificadores[coluna] = cod  # guardado para codificar as ordens novas com o mesmo padrão

tabela.head(3)


# ## 3. Separando variáveis de entrada e resposta
# 
# - `y` = o que quero prever → `status_prazo`
# - `x` = o que uso para prever → todas as outras colunas (menos o id)
# 
# Reservo 25% dos dados para teste, para avaliar o modelo com ordens que ele nunca viu.

# In[5]:


from sklearn.model_selection import train_test_split

y = tabela["status_prazo"]
x = tabela.drop(columns=["status_prazo", "id_ordem"])

x_treino, x_teste, y_treino, y_teste = train_test_split(
    x, y, test_size=0.25, random_state=42)

print(f"{len(x_treino)} ordens para treino | {len(x_teste)} para teste")


# ## 4. Treinando os modelos
# 
# Vou comparar dois algoritmos de classificação:
# 
# | Modelo | Ideia |
# |---|---|
# | **Random Forest** | várias árvores de decisão votando juntas |
# | **KNN** | classifica a ordem pelos "vizinhos" mais parecidos do histórico |
# 
# Dois detalhes que melhoraram o resultado:
# - o Random Forest ganhou **400 árvores** em vez das 100 padrão — mais votos, decisão mais estável;
# - o KNN trabalha com **distância**, então as variáveis precisam estar na mesma escala (`StandardScaler`),
#   senão `quantidade_pecas` (milhares) esmaga variáveis de 0 a 8 como `taxa_refugo_pct`.
# 
# Quem tiver a melhor acurácia nos dados de teste leva o job de prever as ordens novas.

# In[6]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

modelo_arvore = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)

# pipeline: escala os dados e só depois aplica o KNN
modelo_knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7))

modelo_arvore.fit(x_treino, y_treino)
modelo_knn.fit(x_treino, y_treino)


# In[7]:


from sklearn.metrics import accuracy_score

prev_arvore = modelo_arvore.predict(x_teste)
prev_knn = modelo_knn.predict(x_teste)

print(f"Random Forest: {accuracy_score(y_teste, prev_arvore):.1%}")
print(f"KNN:           {accuracy_score(y_teste, prev_knn):.1%}")


# In[8]:


# relatório detalhado do melhor modelo, classe por classe
from sklearn.metrics import classification_report

print(classification_report(y_teste, prev_arvore))


# ## 5. O que mais pesa para uma ordem atrasar
# 
# Acertar a previsão é bom, mas entender o **porquê** é o que muda a rotina do PCP.
# As importâncias do Random Forest mostram quais variáveis o modelo mais usou na decisão.

# In[9]:


import matplotlib.pyplot as plt

importancias = pd.Series(modelo_arvore.feature_importances_, index=x.columns).sort_values()

plt.figure(figsize=(9, 5))
importancias.plot(kind="barh", color="#c0392b")
plt.title("Fatores que mais influenciam o atraso da ordem")
plt.xlabel("importância")
plt.tight_layout()
plt.savefig("assets/importancia_variaveis.png", dpi=120)
plt.show()


# ## 6. Prevendo as ordens novas
# 
# Ordens que acabaram de entrar no PCP (`novas_ordens.csv`). Aplico os mesmos codificadores salvos
# antes e rodo o modelo vencedor.

# In[10]:


novas = pd.read_csv("novas_ordens.csv")

for coluna, cod in codificadores.items():
    novas[coluna] = cod.transform(novas[coluna])

novas["status_prazo_previsto"] = modelo_arvore.predict(novas.drop(columns=["id_ordem"]))

novas[["id_ordem", "produto", "maquina", "prioridade",
       "quantidade_pecas", "dias_ate_prazo", "status_prazo_previsto"]]


# ## Conclusão
# 
# - O modelo consegue antecipar o risco de atraso com boa acurácia usando apenas dados que o PCP já acompanha no dia a dia.
# - As variáveis de maior peso foram **prazo disponível, quantidade, matéria-prima e prioridade** — exatamente as alavancas que a produção usa para reagir.
# - Próximo passo natural seria integrar isso à rotina: rodar o modelo toda manhã com as ordens da semana e montar um alerta de risco.
# 
# *Base de dados sintética, gerada pelo próprio `gerar_dados.py` para simular um histórico real de chão de fábrica.*
