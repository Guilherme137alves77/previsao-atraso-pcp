# PROMPT — continuar amanhã cedo

> Copia o bloco abaixo e cola no opencode para retomar de onde parei.

---

## Contexto rápido

Projeto: **previsao-atrasos-pcp** — ML para prever se uma ordem de produção vai sair
Adiantada / No_Prazo / Atrasada (tema PCP), inspirado no projeto antigo de score de crédito da pasta `IA`.

## O que JÁ está feito

- `gerar_dados.py` → gera `ordens_producao.csv` (6000 ordens) e `novas_ordens.csv` (10 ordens novas)
- `analise_atrasos.ipynb` → notebook completo e EXECUTADO (LabelEncoder, train/test split,
  RandomForest vs KNN, classification_report, gráfico de importância em `assets/importancia_variaveis.png`)
- Resultado atual: **Random Forest 61.9%** | KNN 42.5%

## O que FALTA

1. **Melhorar a acurácia** (61.9% tá fraco) — ideias: ajustar o `gerar_dados.py` pra deixar os padrões
   mais claros, testar hiperparâmetros do RandomForest, escalar dados pro KNN
2. Criar `README.md` bonito pra GitHub (descrição, tabela de colunas, como rodar, resultados, print do gráfico)
3. Criar `requirements.txt` e `.gitignore`
4. `git init` + primeiro commit + subir pro GitHub

---

## Prompt pronto (copiar daqui pra baixo)

```
Bom dia! Continua o projeto na pasta previsao-atrasos-pcp. Lê o PROMPT.md pra ver o contexto.

O notebook analise_atrasos.ipynb já está funcionando com Random Forest em 61.9% de acurácia.
Faz o seguinte, nessa ordem:

1. Melhora a acurácia do modelo pra pelo menos 80% — pode ajustar o gerar_dados.py pra deixar
   as relações entre variáveis e o status_prazo mais claras/regenerar os dados, e/ou tunar o
   RandomForest (n_estimators, max_depth etc). Reexecuta o notebook depois.
2. Cria um README.md caprichado pra GitHub: nome do projeto, problema de PCP que resolve,
   tabela explicando as colunas dos dados, como rodar (venv + pip install -r requirements.txt),
   resultados com as acurácias finais e o gráfico assets/importancia_variaveis.png embutido.
3. Cria requirements.txt (pandas, scikit-learn, matplotlib) e .gitignore (.venv/, __pycache__/,
   .ipynb_checkpoints/, .idea/).
4. git init, adiciona tudo e faz o commit inicial.
5. IMPORTANTE: depois de terminar, me explica a LÓGICA do projeto inteiro passo a passo,
   como quem ensina alguém que ainda não tem base: o que faz cada parte (gerar_dados.py,
   LabelEncoder, train_test_split, RandomForest vs KNN, acurácia, feature importance),
   POR QUE cada etapa existe e como os dados viram uma previsão. Usa linguagem simples,
   com analogias do chão de fábrica, sem jogar termo técnico sem explicar.

Não inventa nada além disso. No final me mostra o resumo do que mudou e a acurácia nova.
```
