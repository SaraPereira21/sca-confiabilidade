# ================================================================
# MOTOR DE CONFIABILIDADE — NÚCLEO REUTILIZÁVEL
# ================================================================
# Motor estatístico validado para:
# - falhas completas e censura à direita;
# - Kaplan-Meier;
# - Exponencial, Weibull 2P, Lognormal 2P, Normal, Gamma 2P,
#   Logística e Valor Extremo (Gumbel máximo);
# - MLE, Log-Likelihood, AIC, BIC, Delta AIC e Peso de Akaike;
# - MTTF, desvio, mediana, quartis, B10/B50/B90;
# - confiabilidade/probabilidade de falha em uma missão;
# - tempo para uma confiabilidade-alvo;
# - IC 95% Weibull 2P, inclusive com censura à direita;
# - diagnóstico contra Kaplan-Meier e coerência física.
#
# Convenção:
# evento = 1 -> falha observada
# evento = 0 -> censura à direita
# ================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import (
    CensoredData,
    expon,
    weibull_min,
    lognorm,
    norm,
    gamma,
    logistic,
    gumbel_r,
)


DISTRIBUICOES = {
    "Exponencial": {"dist": expon, "fixar_loc_zero": True, "k": 1},
    "Weibull": {"dist": weibull_min, "fixar_loc_zero": True, "k": 2},
    "Lognormal": {"dist": lognorm, "fixar_loc_zero": True, "k": 2},
    "Normal": {"dist": norm, "fixar_loc_zero": False, "k": 2},
    "Gamma": {"dist": gamma, "fixar_loc_zero": True, "k": 2},
    "Logística": {"dist": logistic, "fixar_loc_zero": False, "k": 2},
    "Valor Extremo": {"dist": gumbel_r, "fixar_loc_zero": False, "k": 2},
}


def _validar_entradas(tempos, eventos, missao, confiabilidade_alvo):
    tempos = np.asarray(tempos, dtype=float)

    if tempos.ndim != 1 or len(tempos) == 0:
        raise ValueError("A coluna de tempos deve conter pelo menos uma observação.")

    if not np.all(np.isfinite(tempos)):
        raise ValueError("Existem tempos vazios, infinitos ou não numéricos.")

    if np.any(tempos <= 0):
        raise ValueError("Todos os tempos devem ser maiores que zero.")

    if eventos is None:
        eventos = np.ones(len(tempos), dtype=int)
    else:
        eventos = np.asarray(eventos, dtype=int)

    if eventos.shape != tempos.shape:
        raise ValueError("Tempos e eventos precisam ter o mesmo número de linhas.")

    if not np.all(np.isin(eventos, [0, 1])):
        raise ValueError("Evento aceita somente 1=Falha e 0=Censura à direita.")

    if np.sum(eventos == 1) == 0:
        raise ValueError("É necessária pelo menos uma falha observada para ajustar os modelos.")

    if missao is not None:
        missao = float(missao)
        if not np.isfinite(missao) or missao <= 0:
            raise ValueError("A missão deve ser um número maior que zero.")

    confiabilidade_alvo = float(confiabilidade_alvo)
    if not 0 < confiabilidade_alvo < 1:
        raise ValueError("A confiabilidade-alvo deve estar entre 0 e 1. Ex.: 0.80.")

    return tempos, eventos, missao, confiabilidade_alvo


def kaplan_meier(tempos, eventos):
    tempos = np.asarray(tempos, dtype=float)
    eventos = np.asarray(eventos, dtype=int)

    ordem = np.argsort(tempos)
    t = tempos[ordem]
    e = eventos[ordem]

    sobrevivencia = 1.0
    linhas = []

    for tempo in np.unique(t):
        em_risco = int(np.sum(t >= tempo))
        falhas = int(np.sum((t == tempo) & (e == 1)))
        censuras = int(np.sum((t == tempo) & (e == 0)))

        if falhas > 0:
            sobrevivencia *= 1 - falhas / em_risco

        linhas.append({
            "Tempo": float(tempo),
            "Em risco": em_risco,
            "Falhas": falhas,
            "Censuras": censuras,
            "Confiabilidade KM": float(sobrevivencia),
        })

    return pd.DataFrame(linhas)


def _loglik_censurada(dist, parametros, tempos, eventos):
    mf = eventos == 1
    mc = eventos == 0

    ll = float(np.sum(dist.logpdf(tempos[mf], *parametros)))

    if np.any(mc):
        ll += float(np.sum(dist.logsf(tempos[mc], *parametros)))

    return ll


def _classificar_delta(delta):
    if np.isclose(delta, 0):
        return "Modelo preferencial"
    if delta <= 2:
        return "Alternativo competitivo"
    if delta <= 7:
        return "Modelo preferencial apresenta vantagem"
    return "Baixo suporte relativo"


def _interpretar_beta(beta):
    if beta < 1:
        return (
            "A taxa de falha estimada pela Weibull é decrescente. "
            "Esse padrão pode ser compatível com mortalidade infantil/"
            "falhas iniciais, mas beta isoladamente não identifica a causa física."
        )
    if np.isclose(beta, 1, atol=0.05):
        return (
            "O comportamento Weibull é aproximadamente exponencial, "
            "com taxa de falha aproximadamente constante."
        )
    return (
        "A taxa de falha estimada pela Weibull é crescente. "
        "Esse comportamento é compatível com envelhecimento/desgaste, "
        "mas beta > 1 não identifica sozinho o mecanismo físico da falha."
    )


def _ic_weibull(tempos, eventos, beta, eta):
    d = int(np.sum(eventos == 1))
    log_razao = np.log(tempos / eta)
    z = (tempos / eta) ** beta

    h_bb = -d / beta**2 - np.sum(z * log_razao**2)
    h_ee = beta / eta**2 * (d - (1 + beta) * np.sum(z))
    h_be = (-d + np.sum(z) + beta * np.sum(z * log_razao)) / eta

    info = -np.array([[h_bb, h_be], [h_be, h_ee]], dtype=float)

    try:
        cov = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        return {"disponivel": False, "motivo": "Matriz de informação singular."}

    if not np.all(np.isfinite(cov)) or np.any(np.diag(cov) < 0):
        return {"disponivel": False, "motivo": "Covariância Weibull instável."}

    se_beta = float(np.sqrt(cov[0, 0]))
    se_eta = float(np.sqrt(cov[1, 1]))
    z95 = 1.959963984540054

    def q_ic(p):
        c = -np.log(1 - p)
        q = eta * c ** (1 / beta)
        grad = np.array([
            q * (-np.log(c) / beta**2),
            q / eta,
        ])
        var = float(grad @ cov @ grad)
        se = float(np.sqrt(max(var, 0.0)))
        return {
            "estimativa": float(q),
            "erro_padrao": se,
            "ic95_inferior": float(q - z95 * se),
            "ic95_superior": float(q + z95 * se),
        }

    return {
        "disponivel": True,
        "beta": {
            "estimativa": float(beta),
            "erro_padrao": se_beta,
            "ic95_inferior": float(beta - z95 * se_beta),
            "ic95_superior": float(beta + z95 * se_beta),
        },
        "eta": {
            "estimativa": float(eta),
            "erro_padrao": se_eta,
            "ic95_inferior": float(eta - z95 * se_eta),
            "ic95_superior": float(eta + z95 * se_eta),
        },
        "B10": q_ic(0.10),
        "B50": q_ic(0.50),
        "B90": q_ic(0.90),
    }


def analisar_confiabilidade(
    tempos,
    eventos=None,
    missao=None,
    confiabilidade_alvo=0.80,
    nome_equipamento="Equipamento",
    unidade="h",
):
    """
    Executa a análise completa e devolve resultados estruturados.
    Não imprime e não abre gráficos: ideal para interface web/API.
    """
    tempos, eventos, missao, confiabilidade_alvo = _validar_entradas(
        tempos, eventos, missao, confiabilidade_alvo
    )

    n = len(tempos)
    n_falhas = int(np.sum(eventos == 1))
    n_censuras = int(np.sum(eventos == 0))
    tem_censura = n_censuras > 0

    km = kaplan_meier(tempos, eventos)

    dados_ajuste = (
        CensoredData.right_censored(tempos, eventos == 0)
        if tem_censura
        else tempos
    )

    modelos = {}
    linhas = []

    for nome, cfg in DISTRIBUICOES.items():
        dist = cfg["dist"]

        try:
            if cfg["fixar_loc_zero"]:
                parametros = dist.fit(dados_ajuste, floc=0)
            else:
                parametros = dist.fit(dados_ajuste)

            ll = _loglik_censurada(dist, parametros, tempos, eventos)
            k = cfg["k"]
            aic = 2 * k - 2 * ll
            bic = k * np.log(n) - 2 * ll

            mttf = float(dist.mean(*parametros))
            sd = float(dist.std(*parametros))
            mediana = float(dist.median(*parametros))
            q1 = float(dist.ppf(0.25, *parametros))
            q3 = float(dist.ppf(0.75, *parametros))
            b10 = float(dist.ppf(0.10, *parametros))
            b50 = float(dist.ppf(0.50, *parametros))
            b90 = float(dist.ppf(0.90, *parametros))
            p_neg = float(dist.cdf(0, *parametros))

            r_missao = None
            f_missao = None
            if missao is not None:
                r_missao = float(dist.sf(missao, *parametros))
                f_missao = float(dist.cdf(missao, *parametros))

            tempo_alvo = float(
                dist.isf(confiabilidade_alvo, *parametros)
            )

            # Diagnóstico descritivo contra KM nos tempos observados.
            r_modelo_km = dist.sf(km["Tempo"].to_numpy(), *parametros)
            erros = r_modelo_km - km["Confiabilidade KM"].to_numpy()

            mae = float(np.mean(np.abs(erros)))
            rmse = float(np.sqrt(np.mean(erros**2)))
            desvio_max = float(np.max(np.abs(erros)))

            linhas.append({
                "Distribuição": nome,
                "Log-Likelihood": ll,
                "AIC": aic,
                "BIC": bic,
                "MTTF": mttf,
                "Desvio Modelo": sd,
                "Mediana": mediana,
                "Q1": q1,
                "Q3": q3,
                "B10": b10,
                "B50": b50,
                "B90": b90,
                "R Missão": r_missao,
                "F Missão": f_missao,
                "Tempo R Alvo": tempo_alvo,
                "P Tempo Negativo": p_neg,
                "MAE KM": mae,
                "RMSE KM": rmse,
                "Desvio Máx KM": desvio_max,
            })

            modelos[nome] = {
                "dist": dist,
                "parametros": tuple(float(x) for x in parametros),
            }

        except Exception as exc:
            modelos[nome] = {
                "erro": str(exc),
                "parametros": None,
            }

    if not linhas:
        raise RuntimeError("Nenhum modelo pôde ser ajustado aos dados fornecidos.")

    ranking = pd.DataFrame(linhas).sort_values("AIC").reset_index(drop=True)
    menor_aic = float(ranking["AIC"].min())
    ranking["Delta AIC"] = ranking["AIC"] - menor_aic

    ev = np.exp(-0.5 * ranking["Delta AIC"].to_numpy())
    ranking["Peso Akaike"] = ev / ev.sum()
    ranking["Ranking"] = np.arange(1, len(ranking) + 1)
    ranking["Classificação"] = ranking["Delta AIC"].apply(_classificar_delta)

    melhor_nome = str(ranking.iloc[0]["Distribuição"])
    competitivos = ranking[
        (ranking["Delta AIC"] > 0) & (ranking["Delta AIC"] <= 2)
    ]["Distribuição"].tolist()

    p_weibull = modelos["Weibull"]["parametros"]
    beta = float(p_weibull[0])
    eta = float(p_weibull[2])
    ic_weibull = _ic_weibull(tempos, eventos, beta, eta)

    linha_w = ranking[ranking["Distribuição"] == "Weibull"].iloc[0]

    alertas = []
    for _, linha in ranking.iterrows():
        pneg = float(linha["P Tempo Negativo"])
        if pneg > 0.01:
            alertas.append(
                f"{linha['Distribuição']}: estima {pneg*100:.2f}% "
                "de probabilidade para tempos negativos; avaliar coerência física."
            )

    return {
        "estudo": {
            "nome_equipamento": nome_equipamento,
            "unidade": unidade,
            "n": n,
            "falhas": n_falhas,
            "censuras": n_censuras,
            "possui_censura": tem_censura,
            "missao": missao,
            "confiabilidade_alvo": confiabilidade_alvo,
        },
        "descritiva": {
            "media_observada": float(np.mean(tempos)),
            "mediana_observada": float(np.median(tempos)),
            "desvio_padrao_amostral": (
                float(np.std(tempos, ddof=1)) if n > 1 else np.nan
            ),
            "minimo": float(np.min(tempos)),
            "maximo": float(np.max(tempos)),
        },
        "kaplan_meier": km,
        "ranking": ranking,
        "modelos": modelos,
        "selecao": {
            "modelo_preferencial": melhor_nome,
            "modelos_competitivos": competitivos,
            "mensagem": (
                f"Modelo preferencial pelo AIC: {melhor_nome}."
                + (
                    " Modelos alternativos competitivos: "
                    + ", ".join(competitivos) + "."
                    if competitivos else ""
                )
            ),
        },
        "weibull": {
            "beta": beta,
            "eta": eta,
            "mttf": float(linha_w["MTTF"]),
            "b10": float(linha_w["B10"]),
            "b50": float(linha_w["B50"]),
            "b90": float(linha_w["B90"]),
            "interpretacao_beta": _interpretar_beta(beta),
            "ic95": ic_weibull,
        },
        "alertas_coerencia": alertas,
    }


def resumo_texto(resultado):
    """Resumo curto para terminal, log ou futura camada de interface."""
    e = resultado["estudo"]
    s = resultado["selecao"]
    w = resultado["weibull"]

    linhas = [
        "=" * 88,
        "ANÁLISE DE CONFIABILIDADE",
        "=" * 88,
        f"Equipamento: {e['nome_equipamento']}",
        f"Observações: {e['n']} | Falhas: {e['falhas']} | Censuras: {e['censuras']}",
        s["mensagem"],
        f"Weibull: beta={w['beta']:.6f} | eta={w['eta']:.6f}",
        f"B10={w['b10']:.6f} | B50={w['b50']:.6f} | B90={w['b90']:.6f}",
        w["interpretacao_beta"],
    ]

    if e["missao"] is not None:
        ranking = resultado["ranking"]
        pref = ranking.iloc[0]
        linhas.append(
            f"No modelo preferencial, R({e['missao']})="
            f"{float(pref['R Missão'])*100:.2f}% e "
            f"F({e['missao']})={float(pref['F Missão'])*100:.2f}%."
        )

    return "\n".join(linhas)

def executar_relatorio_completo(exibir_graficos=True):
    """Executa o relatório detalhado e as quatro figuras preservadas do motor validado."""
    # ================================================================

    # 1. CONFIGURAÇÃO DO ESTUDO

    # ================================================================

    nome_equipamento = "Quebra de bombas"

    unidade = "tempo"

    # Valores usados para demonstrar os cálculos de missão/alvo.

    missao = 200

    confiabilidade_alvo = 0.80

    # ================================================================

    # 2. DADOS

    # ================================================================

    dados = np.array([

        76, 90, 98, 103, 105, 114, 121, 124,

        136, 140, 140, 145, 150, 150, 158, 164,

        168, 176, 178, 200, 202, 205, 212, 230,

        245, 246, 247, 249

    ], dtype=float)

    # 1 = falha observada

    # 0 = censura

    evento = np.ones(len(dados), dtype=int)

    # ================================================================

    # 3. VALIDAÇÃO DOS DADOS

    # ================================================================

    if len(dados) != len(evento):

        raise ValueError(

            "Os vetores 'dados' e 'evento' devem possuir o mesmo tamanho."

        )

    if len(dados) < 3:

        raise ValueError("São necessárias pelo menos 3 observações.")

    if np.any(~np.isfinite(dados)):

        raise ValueError("Existem valores inválidos nos dados.")

    if np.any(dados <= 0):

        raise ValueError("Todos os tempos devem ser maiores que zero.")

    if not np.all(np.isin(evento, [0, 1])):

        raise ValueError("O indicador de evento deve conter apenas 0 ou 1.")

    if not 0 < confiabilidade_alvo < 1:

        raise ValueError("A confiabilidade alvo deve estar entre 0 e 1.")

    n = len(dados)

    n_falhas = int(np.sum(evento == 1))

    n_censuras = int(np.sum(evento == 0))

    # ================================================================

    # 4. ESTATÍSTICA DESCRITIVA

    # ================================================================

    media_amostral = np.mean(dados)

    mediana_amostral = np.median(dados)

    desvio_amostral = np.std(dados, ddof=1)

    minimo = np.min(dados)

    maximo = np.max(dados)

    print("\n" + "=" * 104)

    print("ASSISTENTE DE CONFIABILIDADE")

    print("=" * 104)

    print("\n1. DADOS DO ESTUDO")

    print("-" * 104)

    print(f"Equipamento / estudo        : {nome_equipamento}")

    print(f"Número de observações       : {n}")

    print(f"Falhas observadas           : {n_falhas}")

    print(f"Observações censuradas      : {n_censuras}")

    print(f"Média amostral              : {media_amostral:.6f}")

    print(f"Mediana amostral            : {mediana_amostral:.6f}")

    print(f"Desvio padrão amostral      : {desvio_amostral:.6f}")

    print(f"Menor tempo                 : {minimo:.6f}")

    print(f"Maior tempo                 : {maximo:.6f}")

    # ================================================================

    # 5. KAPLAN-MEIER

    # ================================================================

    def calcular_kaplan_meier(tempos, eventos):

        """

        Calcula Kaplan-Meier.

        Para cada tempo t_i:

            S(t_i) = S(t_i-) * (1 - d_i / n_i)

        d_i = número de falhas em t_i

        n_i = número de unidades em risco imediatamente antes de t_i

        Censuras reduzem o conjunto sob risco nos tempos seguintes,

        mas não provocam queda na sobrevivência.

        """

        ordem = np.argsort(tempos)

        tempos_ord = tempos[ordem]

        eventos_ord = eventos[ordem]

        tempos_unicos = np.unique(tempos_ord)

        sobrevivencia = 1.0

        registros = []

        for tempo in tempos_unicos:

            em_risco = np.sum(tempos_ord >= tempo)

            falhas_tempo = np.sum(

                (tempos_ord == tempo) & (eventos_ord == 1)

            )

            censuras_tempo = np.sum(

                (tempos_ord == tempo) & (eventos_ord == 0)

            )

            if falhas_tempo > 0:

                sobrevivencia *= (1 - falhas_tempo / em_risco)

            registros.append({

                "Tempo": float(tempo),

                "Em risco": int(em_risco),

                "Falhas": int(falhas_tempo),

                "Censuras": int(censuras_tempo),

                "Kaplan-Meier": float(sobrevivencia)

            })

        return pd.DataFrame(registros)

    km = calcular_kaplan_meier(dados, evento)

    print("\n" + "=" * 104)

    print("2. CONFIABILIDADE OBSERVADA — KAPLAN-MEIER")

    print("=" * 104)

    print(

        km.to_string(

            index=False,

            float_format=lambda x: f"{x:.6f}"

        )

    )

    # ================================================================

    # 6. SEGURANÇA PARA CENSURA

    # ================================================================

    tem_censura = n_censuras > 0

    if tem_censura:

        dados_ajuste = CensoredData.right_censored(

            dados,

            evento == 0

        )

    else:

        # Mantém exatamente o caminho já validado para falhas completas.

        dados_ajuste = dados

    # ================================================================

    # 7. MODELOS PARAMÉTRICOS

    # ================================================================

    distribuicoes = {

        "Exponencial": {

            "dist": expon,

            "fixar_loc_zero": True,

            "k": 1

        },

        "Weibull": {

            "dist": weibull_min,

            "fixar_loc_zero": True,

            "k": 2

        },

        "Lognormal": {

            "dist": lognorm,

            "fixar_loc_zero": True,

            "k": 2

        },

        "Normal": {

            "dist": norm,

            "fixar_loc_zero": False,

            "k": 2

        },

        "Gamma": {

            "dist": gamma,

            "fixar_loc_zero": True,

            "k": 2

        },

        "Logística": {

            "dist": logistic,

            "fixar_loc_zero": False,

            "k": 2

        },

        "Valor Extremo": {

            "dist": gumbel_r,

            "fixar_loc_zero": False,

            "k": 2

        }

    }

    # ================================================================

    # 8. AJUSTE POR MÁXIMA VEROSSIMILHANÇA

    # ================================================================

    resultados = []

    modelos = {}

    for nome_modelo, config in distribuicoes.items():

        dist = config["dist"]

        k = config["k"]

        if config["fixar_loc_zero"]:

            parametros = dist.fit(dados_ajuste, floc=0)

        else:

            parametros = dist.fit(dados_ajuste)

        mascara_falha = evento == 1

        mascara_censura = evento == 0

        log_likelihood = float(

            np.sum(

                dist.logpdf(

                    dados[mascara_falha],

                    *parametros

                )

            )

        )

        if np.any(mascara_censura):

            log_likelihood += float(

                np.sum(

                    dist.logsf(

                        dados[mascara_censura],

                        *parametros

                    )

                )

            )

        aic = 2 * k - 2 * log_likelihood

        bic = k * np.log(n) - 2 * log_likelihood

        mttf = dist.mean(*parametros)

        desvio_modelo = dist.std(*parametros)

        mediana_modelo = dist.median(*parametros)

        q1 = dist.ppf(0.25, *parametros)

        q3 = dist.ppf(0.75, *parametros)

        p10 = dist.ppf(0.10, *parametros)

        p50 = dist.ppf(0.50, *parametros)

        p90 = dist.ppf(0.90, *parametros)

        R_missao = dist.sf(missao, *parametros)

        F_missao = dist.cdf(missao, *parametros)

        tempo_R_alvo = dist.isf(

            confiabilidade_alvo,

            *parametros

        )

        prob_negativo = dist.cdf(

            0,

            *parametros

        )

        resultados.append({

            "Distribuição": nome_modelo,

            "Log-Likelihood": log_likelihood,

            "AIC": aic,

            "BIC": bic,

            "MTTF": mttf,

            "Desvio Modelo": desvio_modelo,

            "Mediana": mediana_modelo,

            "Q1": q1,

            "Q3": q3,

            "P10": p10,

            "P50": p50,

            "P90": p90,

            "R Missão": R_missao,

            "F Missão": F_missao,

            "Tempo R Alvo": tempo_R_alvo,

            "P Tempo Negativo": prob_negativo

        })

        modelos[nome_modelo] = {

            "dist": dist,

            "parametros": parametros

        }

    # ================================================================

    # 9. RANKING AIC

    # ================================================================

    ranking = pd.DataFrame(resultados)

    ranking = ranking.sort_values(

        "AIC"

    ).reset_index(drop=True)

    menor_aic = ranking["AIC"].min()

    ranking["Delta AIC"] = (

        ranking["AIC"] - menor_aic

    )

    evidencia_relativa = np.exp(

        -0.5 * ranking["Delta AIC"]

    )

    ranking["Peso Akaike"] = (

        evidencia_relativa / evidencia_relativa.sum()

    )

    ranking["Ranking"] = np.arange(

        1,

        len(ranking) + 1

    )

    def classificar_delta(delta):

        if np.isclose(delta, 0):

            return "Modelo preferencial"

        if delta <= 2:

            return "Alternativo competitivo"

        if delta <= 7:

            return "Modelo preferencial apresenta vantagem"

        return "Baixo suporte relativo"

    ranking["Classificação"] = ranking[

        "Delta AIC"

    ].apply(classificar_delta)

    print("\n" + "=" * 104)

    print("3. COMPARAÇÃO ESTATÍSTICA DOS MODELOS")

    print("=" * 104)

    print(

        ranking[

            [

                "Ranking",

                "Distribuição",

                "Log-Likelihood",

                "AIC",

                "BIC",

                "Delta AIC",

                "Peso Akaike",

                "Classificação"

            ]

        ].to_string(

            index=False,

            float_format=lambda x: f"{x:.6f}"

        )

    )

    # ================================================================

    # 10. SELEÇÃO ASSISTIDA

    # ================================================================

    melhor = ranking.iloc[0]

    melhor_nome = melhor["Distribuição"]

    competitivos = ranking[

        (ranking["Delta AIC"] > 0)

        &

        (ranking["Delta AIC"] <= 2)

    ].copy()

    print("\n" + "=" * 104)

    print("4. SELEÇÃO ASSISTIDA")

    print("=" * 104)

    print(

        f"\nModelo preferencial pelo AIC : {melhor_nome}"

    )

    print(

        f"AIC                          : {melhor['AIC']:.6f}"

    )

    print(

        f"Peso de Akaike               : "

        f"{melhor['Peso Akaike'] * 100:.2f}%"

    )

    if len(competitivos) > 0:

        print("\nModelos alternativos competitivos:")

        for _, linha in competitivos.iterrows():

            print(

                f"- {linha['Distribuição']} | "

                f"ΔAIC = {linha['Delta AIC']:.4f} | "

                f"Peso = {linha['Peso Akaike'] * 100:.2f}%"

            )

        print(

            "\nA preferência estatística não é dominante. "

            "Diagnóstico de aderência e coerência física "

            "devem complementar a decisão."

        )

    # ================================================================

    # 11. PARÂMETROS ESTIMADOS

    # ================================================================

    print("\n" + "=" * 104)

    print("5. PARÂMETROS ESTIMADOS POR MÁXIMA VEROSSIMILHANÇA")

    print("=" * 104)

    for nome_modelo in ranking["Distribuição"]:

        parametros = modelos[nome_modelo]["parametros"]

        print(f"\n{nome_modelo}")

        if nome_modelo == "Exponencial":

            loc = parametros[0]

            scale = parametros[1]

            taxa = 1 / scale

            print(f"loc                  = {loc:.9f}")

            print(f"escala / MTTF        = {scale:.9f}")

            print(f"taxa lambda          = {taxa:.9f}")

        elif nome_modelo == "Weibull":

            beta_temp = parametros[0]

            loc = parametros[1]

            eta_temp = parametros[2]

            print(f"Beta (forma)         = {beta_temp:.9f}")

            print(f"loc                  = {loc:.9f}")

            print(f"Eta (escala)         = {eta_temp:.9f}")

        elif nome_modelo == "Lognormal":

            sigma = parametros[0]

            loc = parametros[1]

            escala = parametros[2]

            mu_log = np.log(escala)

            print(f"Sigma                = {sigma:.9f}")

            print(f"Mu logarítmico       = {mu_log:.9f}")

            print(f"loc                  = {loc:.9f}")

            print(f"escala               = {escala:.9f}")

        elif nome_modelo == "Normal":

            mu = parametros[0]

            sigma = parametros[1]

            print(f"Média / loc          = {mu:.9f}")

            print(f"Sigma                = {sigma:.9f}")

        elif nome_modelo == "Gamma":

            forma = parametros[0]

            loc = parametros[1]

            escala = parametros[2]

            print(f"Forma                = {forma:.9f}")

            print(f"loc                  = {loc:.9f}")

            print(f"Escala               = {escala:.9f}")

        elif nome_modelo == "Logística":

            loc = parametros[0]

            escala = parametros[1]

            print(f"loc                  = {loc:.9f}")

            print(f"escala               = {escala:.9f}")

        elif nome_modelo == "Valor Extremo":

            loc = parametros[0]

            escala = parametros[1]

            print(f"loc                  = {loc:.9f}")

            print(f"escala               = {escala:.9f}")

    # ================================================================

    # 12. WEIBULL — VALIDAÇÃO E INTERPRETAÇÃO

    # ================================================================

    parametros_weibull = modelos["Weibull"]["parametros"]

    beta = parametros_weibull[0]

    eta = parametros_weibull[2]

    linha_weibull = ranking[

        ranking["Distribuição"] == "Weibull"

    ].iloc[0]

    print("\n" + "=" * 104)

    print("6. RESULTADOS WEIBULL")

    print("=" * 104)

    print(f"\nBeta (β)             : {beta:.9f}")

    print(f"Eta (η)              : {eta:.9f}")

    print(f"MTTF                 : {linha_weibull['MTTF']:.9f}")

    print(f"Desvio padrão modelo : {linha_weibull['Desvio Modelo']:.9f}")

    print(f"Mediana              : {linha_weibull['Mediana']:.9f}")

    print(f"Q1                   : {linha_weibull['Q1']:.9f}")

    print(f"Q3                   : {linha_weibull['Q3']:.9f}")

    print(f"P10                  : {linha_weibull['P10']:.9f}")

    print(f"P50                  : {linha_weibull['P50']:.9f}")

    print(f"P90                  : {linha_weibull['P90']:.9f}")

    print("\nInterpretação:")

    if beta < 1:

        print(

            "A taxa de falha estimada pela Weibull é decrescente."

        )

    elif np.isclose(beta, 1, atol=0.05):

        print(

            "O comportamento Weibull é aproximadamente exponencial, "

            "com taxa de falha aproximadamente constante."

        )

    else:

        print(

            "A taxa de falha estimada pela Weibull é crescente."

        )

        print(

            "Esse comportamento é compatível com envelhecimento/"

            "desgaste, mas β > 1 não identifica sozinho o mecanismo "

            "físico da falha."

        )

    # ================================================================

    # 12B. INCERTEZA ESTATÍSTICA — WEIBULL 2P

    # ================================================================

    # Validado para falhas completas. Para censura à direita, os ICs

    # permanecem bloqueados até uma validação específica da Hessiana.

    # ================================================================

    if tem_censura:

        print("\n" + "=" * 104)

        print("6B. INCERTEZA ESTATÍSTICA — WEIBULL 2P")

        print("=" * 104)

        print(

            "\nIC 95% não publicado para dados censurados nesta etapa. "

            "O ajuste MLE censurado, AIC/BIC, Kaplan-Meier e as funções "

            "de confiabilidade permanecem disponíveis."

        )

    else:

        log_razao = np.log(dados / eta)

        z_weibull = (dados / eta) ** beta

        d2l_dbeta2 = (

            -n / (beta ** 2)

            - np.sum(z_weibull * (log_razao ** 2))

        )

        d2l_deta2 = (

            beta / (eta ** 2)

            * (

                n

                - (1 + beta) * np.sum(z_weibull)

            )

        )

        d2l_dbeta_deta = (

            (

                -n

                + np.sum(z_weibull)

                + beta * np.sum(z_weibull * log_razao)

            )

            / eta

        )

        informacao_observada = -np.array(

            [

                [d2l_dbeta2, d2l_dbeta_deta],

                [d2l_dbeta_deta, d2l_deta2],

            ],

            dtype=float,

        )

        cov_weibull = np.linalg.inv(

            informacao_observada

        )

        se_beta = float(

            np.sqrt(cov_weibull[0, 0])

        )

        se_eta = float(

            np.sqrt(cov_weibull[1, 1])

        )

        z_975 = 1.959963984540054

        ic_beta = (

            beta - z_975 * se_beta,

            beta + z_975 * se_beta,

        )

        ic_eta = (

            eta - z_975 * se_eta,

            eta + z_975 * se_eta,

        )

        def quantil_weibull_com_ic(probabilidade):

            c = -np.log(1 - probabilidade)

            quantil = eta * (c ** (1 / beta))

            dq_dbeta = (

                quantil

                * (

                    -np.log(c)

                    / (beta ** 2)

                )

            )

            dq_deta = quantil / eta

            gradiente = np.array(

                [dq_dbeta, dq_deta],

                dtype=float,

            )

            variancia = float(

                gradiente

                @ cov_weibull

                @ gradiente

            )

            erro_padrao = np.sqrt(

                max(variancia, 0.0)

            )

            return {

                "Estimativa": float(quantil),

                "Erro padrão": float(erro_padrao),

                "IC 95% inferior": float(

                    quantil - z_975 * erro_padrao

                ),

                "IC 95% superior": float(

                    quantil + z_975 * erro_padrao

                ),

            }

        ic_b10 = quantil_weibull_com_ic(0.10)

        ic_b50 = quantil_weibull_com_ic(0.50)

        ic_b90 = quantil_weibull_com_ic(0.90)

        print("\n" + "=" * 104)

        print("6B. INCERTEZA ESTATÍSTICA — WEIBULL 2P")

        print("=" * 104)

        print(

            "\nParâmetro      Estimativa       Erro padrão"

            "       IC 95% inferior     IC 95% superior"

        )

        print(

            f"Beta       {beta:>14.9f}"

            f" {se_beta:>17.9f}"

            f" {ic_beta[0]:>21.9f}"

            f" {ic_beta[1]:>21.9f}"

        )

        print(

            f"Eta        {eta:>14.9f}"

            f" {se_eta:>17.9f}"

            f" {ic_eta[0]:>21.9f}"

            f" {ic_eta[1]:>21.9f}"

        )

        print(

            "\nQuantil        Estimativa       Erro padrão"

            "       IC 95% inferior     IC 95% superior"

        )

        for rotulo, resultado_ic in [

            ("B10", ic_b10),

            ("B50", ic_b50),

            ("B90", ic_b90),

        ]:

            print(

                f"{rotulo:<10}"

                f" {resultado_ic['Estimativa']:>14.9f}"

                f" {resultado_ic['Erro padrão']:>17.9f}"

                f" {resultado_ic['IC 95% inferior']:>21.9f}"

                f" {resultado_ic['IC 95% superior']:>21.9f}"

            )

        print(

            "\nLeitura: os intervalos de confiança representam a "

            "incerteza estatística das estimativas."

        )

        print(

            "Eles não alteram o valor pontual já calculado; mostram "

            "uma faixa plausível em torno dele."

        )

    # 13. COERÊNCIA FÍSICA

    # ================================================================

    print("\n" + "=" * 104)

    print("7. COERÊNCIA FÍSICA DOS MODELOS")

    print("=" * 104)

    for _, linha in ranking.iterrows():

        nome = linha["Distribuição"]

        p_neg = linha["P Tempo Negativo"]

        if p_neg <= 0.001:

            status = "OK"

        elif p_neg <= 0.01:

            status = "ATENÇÃO"

        else:

            status = "ALERTA"

        print(

            f"{nome:<16} | "

            f"P(T < 0) = {p_neg * 100:>8.4f}% | "

            f"{status}"

        )

    # ================================================================

    # 14. DIAGNÓSTICO DESCRITIVO CONTRA KAPLAN-MEIER

    # ================================================================

    #

    # MAE, RMSE e desvio máximo são diagnósticos descritivos.

    # NÃO são testes formais de goodness-of-fit.

    # ================================================================

    tempos_km = km["Tempo"].to_numpy()

    R_km = km["Kaplan-Meier"].to_numpy()

    diagnosticos = []

    for nome_modelo in ranking["Distribuição"]:

        dist = modelos[nome_modelo]["dist"]

        parametros = modelos[nome_modelo]["parametros"]

        R_modelo = dist.sf(

            tempos_km,

            *parametros

        )

        residuos = R_modelo - R_km

        mae = np.mean(

            np.abs(residuos)

        )

        rmse = np.sqrt(

            np.mean(residuos ** 2)

        )

        desvio_maximo = np.max(

            np.abs(residuos)

        )

        indice_maximo = np.argmax(

            np.abs(residuos)

        )

        tempo_desvio_max = tempos_km[

            indice_maximo

        ]

        diagnosticos.append({

            "Distribuição": nome_modelo,

            "MAE": mae,

            "RMSE": rmse,

            "Desvio Máximo": desvio_maximo,

            "Tempo Desvio Máx": tempo_desvio_max

        })

    diagnosticos = pd.DataFrame(

        diagnosticos

    )

    diagnosticos = diagnosticos.merge(

        ranking[

            [

                "Distribuição",

                "Ranking",

                "AIC",

                "Delta AIC"

            ]

        ],

        on="Distribuição"

    )

    diagnosticos = diagnosticos.sort_values(

        "RMSE"

    ).reset_index(drop=True)

    print("\n" + "=" * 104)

    print("8. DIAGNÓSTICO DESCRITIVO DE ADERÊNCIA")

    print("=" * 104)

    print(

        diagnosticos[

            [

                "Distribuição",

                "AIC",

                "Delta AIC",

                "MAE",

                "RMSE",

                "Desvio Máximo",

                "Tempo Desvio Máx"

            ]

        ].to_string(

            index=False,

            float_format=lambda x: f"{x:.6f}"

        )

    )

    print(

        "\nMAE, RMSE e Desvio Máximo medem proximidade descritiva "

        "com Kaplan-Meier. Não substituem AIC nem constituem, nesta "

        "versão, testes formais de aderência."

    )

    # ================================================================

    # 15. TABELA KAPLAN-MEIER x MODELOS

    # ================================================================

    comparacao_curvas = km[

        ["Tempo", "Kaplan-Meier"]

    ].copy()

    for nome_modelo in ranking["Distribuição"]:

        dist = modelos[nome_modelo]["dist"]

        parametros = modelos[nome_modelo]["parametros"]

        comparacao_curvas[nome_modelo] = dist.sf(

            comparacao_curvas["Tempo"],

            *parametros

        )

    print("\n" + "=" * 104)

    print("9. KAPLAN-MEIER x MODELOS PARAMÉTRICOS")

    print("=" * 104)

    print(

        comparacao_curvas.to_string(

            index=False,

            float_format=lambda x: f"{x:.6f}"

        )

    )

    # ================================================================

    # 16. RESULTADOS DE VIDA POR DISTRIBUIÇÃO

    # ================================================================

    print("\n" + "=" * 104)

    print("10. RESUMO DE VIDA POR DISTRIBUIÇÃO")

    print("=" * 104)

    tabela_vida = ranking[

        [

            "Ranking",

            "Distribuição",

            "AIC",

            "Delta AIC",

            "Peso Akaike",

            "MTTF",

            "Desvio Modelo",

            "Mediana",

            "Q1",

            "Q3",

            "P10",

            "P90"

        ]

    ].copy()

    tabela_vida["Peso Akaike"] *= 100

    print(

        tabela_vida.to_string(

            index=False,

            float_format=lambda x: f"{x:.4f}"

        )

    )

    # ================================================================

    # 17. MISSÃO E CONFIABILIDADE ALVO

    # ================================================================

    print("\n" + "=" * 104)

    print("11. MISSÃO E CONFIABILIDADE ALVO")

    print("=" * 104)

    print(

        f"\nTempo de missão de demonstração: {missao:.2f}"

    )

    tabela_missao = ranking[

        [

            "Distribuição",

            "R Missão",

            "F Missão",

            "Tempo R Alvo"

        ]

    ].copy()

    tabela_missao["R Missão"] *= 100

    tabela_missao["F Missão"] *= 100

    print(

        tabela_missao.to_string(

            index=False,

            float_format=lambda x: f"{x:.4f}"

        )

    )

    print(

        f"\nConfiabilidade alvo usada: "

        f"{confiabilidade_alvo * 100:.2f}%"

    )

    # ================================================================

    # 18. PREPARAÇÃO DOS GRÁFICOS

    # ================================================================

    x_max = max(

        maximo * 1.12,

        missao * 1.05

    )

    x_grafico = np.linspace(

        0.001,

        x_max,

        800

    )

    x_km_plot = np.concatenate(

        (

            [0],

            km["Tempo"].to_numpy()

        )

    )

    y_km_plot = np.concatenate(

        (

            [1],

            km["Kaplan-Meier"].to_numpy()

        )

    )

    # ================================================================

    # 19. DADOS GRÁFICOS DA WEIBULL

    # ================================================================

    dist_weibull = modelos[

        "Weibull"

    ]["dist"]

    R_weibull = dist_weibull.sf(

        x_grafico,

        *parametros_weibull

    )

    f_weibull = dist_weibull.pdf(

        x_grafico,

        *parametros_weibull

    )

    h_weibull = np.divide(

        f_weibull,

        R_weibull,

        out=np.zeros_like(f_weibull),

        where=R_weibull > 1e-12

    )

    # ================================================================

    # 20. FIGURA 1 — OVERVIEW WEIBULL

    # ================================================================

    fig1, eixos = plt.subplots(

        2,

        2,

        figsize=(15, 10)

    )

    fig1.suptitle(

        (

            "Overview de Confiabilidade — Weibull\n"

            f"β = {beta:.4f}   |   "

            f"η = {eta:.4f}   |   "

            f"MTTF = {linha_weibull['MTTF']:.4f}"

        ),

        fontsize=17,

        fontweight="bold"

    )

    # --- 20A. Função de risco

    ax = eixos[0, 0]

    ax.plot(

        x_grafico,

        h_weibull,

        linewidth=2.5

    )

    ax.set_title(

        "Função de risco — h(t)",

        fontsize=13,

        fontweight="bold"

    )

    ax.set_xlabel("Tempo")

    ax.set_ylabel("Taxa instantânea de falha h(t)")

    ax.set_xlim(0, x_max)

    ax.grid(alpha=0.25)

    # --- 20B. Kaplan-Meier x Weibull

    ax = eixos[0, 1]

    ax.step(

        x_km_plot,

        y_km_plot,

        where="post",

        linewidth=2.4,

        label="Kaplan-Meier observado"

    )

    ax.plot(

        x_grafico,

        R_weibull,

        linewidth=2.5,

        label="Weibull ajustada"

    )

    ax.set_title(

        "Confiabilidade observada × Weibull",

        fontsize=13,

        fontweight="bold"

    )

    ax.set_xlabel("Tempo")

    ax.set_ylabel("Confiabilidade R(t)")

    ax.set_xlim(0, x_max)

    ax.set_ylim(0, 1.03)

    ax.grid(alpha=0.25)

    ax.legend(fontsize=9)

    # --- 20C. Densidade

    ax = eixos[1, 0]

    ax.plot(

        x_grafico,

        f_weibull,

        linewidth=2.5

    )

    ax.axvline(

        eta,

        linestyle="--",

        linewidth=1.5,

        label=f"η = {eta:.1f}"

    )

    ax.set_title(

        "Densidade de probabilidade — f(t)",

        fontsize=13,

        fontweight="bold"

    )

    ax.set_xlabel("Tempo")

    ax.set_ylabel("Densidade")

    ax.set_xlim(0, x_max)

    ax.grid(alpha=0.25)

    ax.legend(fontsize=9)

    # --- 20D. Confiabilidade Weibull

    ax = eixos[1, 1]

    ax.plot(

        x_grafico,

        R_weibull,

        linewidth=2.5

    )

    t_R50 = dist_weibull.isf(

        0.50,

        *parametros_weibull

    )

    ax.axhline(

        0.50,

        linestyle="--",

        linewidth=1.2,

        alpha=0.7

    )

    ax.axvline(

        t_R50,

        linestyle="--",

        linewidth=1.2,

        alpha=0.7,

        label=f"R(t)=50% em t≈{t_R50:.1f}"

    )

    ax.set_title(

        "Curva de confiabilidade — R(t)",

        fontsize=13,

        fontweight="bold"

    )

    ax.set_xlabel("Tempo")

    ax.set_ylabel("Confiabilidade R(t)")

    ax.set_xlim(0, x_max)

    ax.set_ylim(0, 1.03)

    ax.grid(alpha=0.25)

    ax.legend(fontsize=9)

    fig1.subplots_adjust(

        left=0.07,

        right=0.97,

        bottom=0.08,

        top=0.87,

        hspace=0.38,

        wspace=0.22

    )

    # ================================================================

    # 21. FIGURA 2 — COMPARAÇÃO ENTRE MODELOS

    # ================================================================

    fig2, ax2 = plt.subplots(

        figsize=(15, 8)

    )

    ax2.step(

        x_km_plot,

        y_km_plot,

        where="post",

        linewidth=3,

        label="Kaplan-Meier observado"

    )

    for _, linha in ranking.iterrows():

        nome_modelo = linha["Distribuição"]

        dist = modelos[nome_modelo]["dist"]

        parametros = modelos[nome_modelo]["parametros"]

        R_modelo = dist.sf(

            x_grafico,

            *parametros

        )

        if nome_modelo == melhor_nome:

            ax2.plot(

                x_grafico,

                R_modelo,

                linewidth=3,

                label=f"{nome_modelo} — preferencial AIC"

            )

        else:

            ax2.plot(

                x_grafico,

                R_modelo,

                linewidth=1.7,

                alpha=0.80,

                label=nome_modelo

            )

    ax2.set_title(

        (

            "Comparação dos modelos de confiabilidade\n"

            f"Modelo preferencial pelo AIC: {melhor_nome}"

        ),

        fontsize=16,

        fontweight="bold"

    )

    ax2.set_xlabel("Tempo", fontsize=11)

    ax2.set_ylabel("Confiabilidade R(t)", fontsize=11)

    ax2.set_xlim(0, x_max)

    ax2.set_ylim(0, 1.03)

    ax2.grid(alpha=0.25)

    ax2.legend(

        fontsize=9,

        ncol=2,

        loc="upper right"

    )

    fig2.tight_layout()

    # ================================================================

    # 22. FIGURA 3 — RANKING VISUAL POR DELTA AIC

    # ================================================================

    ranking_plot = ranking.sort_values(

        "Delta AIC",

        ascending=True

    ).copy()

    fig3, ax3 = plt.subplots(

        figsize=(12, 7)

    )

    ax3.barh(

        ranking_plot["Distribuição"],

        ranking_plot["Delta AIC"]

    )

    ax3.axvline(

        2,

        linestyle="--",

        linewidth=1.5,

        label="ΔAIC = 2"

    )

    ax3.axvline(

        7,

        linestyle="--",

        linewidth=1.5,

        label="ΔAIC = 7"

    )

    ax3.set_title(

        "Comparação estatística dos modelos — ΔAIC",

        fontsize=15,

        fontweight="bold"

    )

    ax3.set_xlabel("ΔAIC")

    ax3.set_ylabel("Distribuição")

    ax3.grid(

        axis="x",

        alpha=0.25

    )

    ax3.legend()

    # Como o DataFrame foi ordenado para o gráfico, usamos as

    # posições efetivas das categorias na ordem exibida.

    for posicao, (_, linha) in enumerate(

        ranking_plot.iterrows()

    ):

        ax3.text(

            linha["Delta AIC"] + 0.35,

            posicao,

            f"{linha['Delta AIC']:.2f}",

            va="center",

            fontsize=10

        )

    fig3.tight_layout()

    # ================================================================

    # 23. FIGURA 4 — DIAGNÓSTICO Q-Q

    # ================================================================

    #

    # Para falhas completas usamos Median Ranks de Benard:

    #

    # F_i = (i - 0.3) / (n + 0.4)

    #

    # X = quantis teóricos do modelo ajustado

    # Y = tempos observados

    #

    # Quanto mais próximos da diagonal, maior a compatibilidade

    # visual entre os quantis do modelo e a amostra.

    # ================================================================

    dados_ord = np.sort(dados)

    i = np.arange(

        1,

        n + 1

    )

    probabilidades_benard = (

        i - 0.3

    ) / (

        n + 0.4

    )

    fig4, eixos_qq = plt.subplots(

        4,

        2,

        figsize=(15, 18)

    )

    eixos_qq = eixos_qq.flatten()

    for indice, (_, linha) in enumerate(

        ranking.iterrows()

    ):

        ax = eixos_qq[indice]

        nome_modelo = linha["Distribuição"]

        dist = modelos[nome_modelo]["dist"]

        parametros = modelos[nome_modelo]["parametros"]

        quantis_teoricos = dist.ppf(

            probabilidades_benard,

            *parametros

        )

        ax.scatter(

            quantis_teoricos,

            dados_ord,

            s=32

        )

        limite_min = min(

            np.min(quantis_teoricos),

            np.min(dados_ord)

        )

        limite_max = max(

            np.max(quantis_teoricos),

            np.max(dados_ord)

        )

        ax.plot(

            [limite_min, limite_max],

            [limite_min, limite_max],

            linewidth=1.5

        )

        ax.set_title(

            (

                f"{int(linha['Ranking'])}º — {nome_modelo}\n"

                f"AIC = {linha['AIC']:.2f}   |   "

                f"ΔAIC = {linha['Delta AIC']:.2f}"

            ),

            fontsize=11,

            fontweight="bold",

            pad=10

        )

        ax.set_xlabel(

            "Quantis teóricos"

        )

        ax.set_ylabel(

            "Tempos observados"

        )

        ax.grid(

            alpha=0.25

        )

    for indice in range(

        len(ranking),

        len(eixos_qq)

    ):

        eixos_qq[indice].axis("off")

    fig4.suptitle(

        "Diagnóstico Q-Q dos modelos",

        fontsize=17,

        fontweight="bold"

    )

    fig4.subplots_adjust(

        left=0.08,

        right=0.97,

        bottom=0.05,

        top=0.94,

        hspace=0.55,

        wspace=0.28

    )

    # ================================================================

    # 24. RESUMO FINAL

    # ================================================================

    print("\n" + "=" * 104)

    print("12. RESUMO PARA O ANALISTA")

    print("=" * 104)

    print(

        f"""

    MODELO PREFERENCIAL PELO AIC

    {melhor_nome}

    AIC:

    {melhor['AIC']:.4f}

    Peso de Akaike:

    {melhor['Peso Akaike'] * 100:.2f}%

    WEIBULL

    Beta (β):

    {beta:.4f}

    Eta (η):

    {eta:.4f}

    MTTF:

    {linha_weibull['MTTF']:.4f}

    """

    )

    if len(competitivos) > 0:

        print(

            "MODELOS ALTERNATIVOS COMPETITIVOS:"

        )

        for _, linha in competitivos.iterrows():

            print(

                f"  • {linha['Distribuição']}: "

                f"ΔAIC = {linha['Delta AIC']:.4f}"

            )

        print(

            "\nA preferência estatística pelo modelo de menor AIC "

            "não é dominante neste conjunto."

        )

    print(

        """

    LEITURA DAS FIGURAS

    Figura 1 — Overview Weibull

    Mostra quatro aspectos do mesmo modelo Weibull:

    função de risco, Kaplan-Meier x Weibull, densidade

    e confiabilidade.

    Figura 2 — Comparação de modelos

    Mostra Kaplan-Meier e as sete distribuições ajustadas

    no mesmo sistema de eixos.

    Figura 3 — Delta AIC

    Mostra visualmente a diferença de suporte relativo entre

    os modelos candidatos.

    Figura 4 — Diagnóstico Q-Q

    Compara quantis teóricos e tempos observados.

    IMPORTANTE:

    Os gráficos são diagnósticos complementares.

    A seleção estatística não deve ser feita somente pela

    aparência de uma curva.

    """

    )

    print("=" * 104)

    print("MOTOR CONCLUÍDO.")

    print("=" * 104)

    # ================================================================

    # 25. EXIBIR FIGURAS

    # ================================================================

    if exibir_graficos:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    executar_relatorio_completo(exibir_graficos=True)
