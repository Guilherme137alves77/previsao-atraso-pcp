"""
Gera a base de dados do projeto: histórico de ordens de produção de uma
indústria metalúrgica, com o status de prazo de cada ordem.

Os dados são sintéticos, mas seguem uma lógica de chão de fábrica:
- falta de matéria-prima e prazo curto aumentam muito a chance de atraso
- prioridade alta faz a ordem "pular a fila" e adianta a produção
- mais operadores alocados e menos refugo ajudam a cumprir o prazo
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

PRODUTOS = ["Engrenagem", "Eixo", "Flange", "Suporte", "Carcaça"]
MAQUINAS = ["CNC-01", "CNC-02", "Torno-03", "Fresadora-01", "Mandrilhadora-A"]
TURNOS = ["Manhã", "Tarde", "Noite"]
PRIORIDADES = ["Baixa", "Média", "Alta", "Urgente"]


def gerar_ordens(qtd: int) -> pd.DataFrame:
    mes = rng.integers(1, 13, qtd)
    produto = rng.choice(PRODUTOS, qtd)
    maquina = rng.choice(MAQUINAS, qtd)
    turno = rng.choice(TURNOS, qtd, p=[0.4, 0.35, 0.25])
    quantidade = rng.integers(50, 5000, qtd)
    setup = np.round(rng.uniform(15, 180, qtd), 0)
    horas_maquina = np.round(quantidade * rng.uniform(0.01, 0.05, qtd), 2)
    prioridade = rng.choice(PRIORIDADES, qtd, p=[0.2, 0.4, 0.3, 0.1])
    materia_prima = rng.choice(["Sim", "Não"], qtd, p=[0.82, 0.18])
    operadores = rng.integers(1, 7, qtd)
    refugo = np.round(rng.uniform(0.5, 8, qtd), 2)
    paradas = rng.integers(0, 6, qtd)
    dias_prazo = rng.integers(3, 46, qtd)

    # score de risco: quanto maior, maior a chance de atrasar
    risco = (
        1.0
        + (materia_prima == "Não") * 3.0
        + (quantidade / 5000) * 2.5
        + (45 - dias_prazo) / 42 * 2.5
        + (6 - operadores) / 5 * 1.2
        + (refugo / 8) * 1.0
        + (setup / 180) * 0.6
        + (turno == "Noite") * 0.3
        + paradas * 0.15
        - pd.Series(prioridade).map({"Baixa": 0.0, "Média": 0.3, "Alta": 0.6, "Urgente": 0.9}).values
        + rng.normal(0, 0.4, qtd)
    )

    limites = np.quantile(risco, [0.30, 0.68])
    status = np.select(
        [risco <= limites[0], risco <= limites[1]],
        ["Adiantado", "No_Prazo"],
        default="Atrasado",
    )

    df = pd.DataFrame({
        "id_ordem": [f"OP-{i:05d}" for i in range(1, qtd + 1)],
        "mes": mes,
        "produto": produto,
        "maquina": maquina,
        "turno": turno,
        "quantidade_pecas": quantidade,
        "tempo_setup_min": setup,
        "horas_maquina_previstas": horas_maquina,
        "prioridade": prioridade,
        "materia_prima_ok": materia_prima,
        "operadores_alocados": operadores,
        "taxa_refugo_pct": refugo,
        "paradas_manutencao_mes": paradas,
        "dias_ate_prazo": dias_prazo,
        "status_prazo": status,
    })
    return df


if __name__ == "__main__":
    historico = gerar_ordens(6000)
    historico.to_csv("ordens_producao.csv", index=False)

    # ordens novas que chegaram hoje no PCP, sem o status (é o que queremos prever)
    novas = gerar_ordens(10).drop(columns=["status_prazo"])
    novas.to_csv("novas_ordens.csv", index=False)

    print(historico["status_prazo"].value_counts())
    print(f"\nordens_producao.csv: {len(historico)} linhas")
    print(f"novas_ordens.csv: {len(novas)} linhas")
