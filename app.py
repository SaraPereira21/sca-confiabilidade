import io
import math
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from motor_confiabilidade import analisar_confiabilidade

st.set_page_config(
    page_title="SCA — Validação Acadêmica",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp { background:#fbfaf6; color:#22242b; }
.block-container { max-width:1450px; padding-top:1.4rem; }
[data-testid="stSidebar"] { background:#f2efff; border-right:1px solid #ded8f5; }
h1,h2,h3 { color:#272832; }
.sca-card {
    background:white; border:1px solid #e5e1d7; border-radius:18px;
    padding:18px 20px; box-shadow:0 3px 12px rgba(31,35,48,.05);
    margin-bottom:12px;
}
.sca-pill {
    display:inline-block; padding:5px 10px; border-radius:999px;
    background:#eee9ff; color:#5b3fd6; font-size:.75rem; font-weight:800;
    letter-spacing:.04em;
}
.small-note { color:#6f7280; font-size:.88rem; }
</style>
""", unsafe_allow_html=True)

BOMBAS = [
    76,90,98,103,105,114,121,124,136,140,140,145,150,150,
    158,164,168,176,178,200,202,205,212,230,245,246,247,249
]
EIXO = [
    20.99,23.11,32.01,32.78,36.57,38.27,39.33,39.88,40.24,43.44,
    43.64,46.95,48.33,49.97,50.66,53.11,53.56,56.53,59.45,61.68,
    62.22,65.84,66.07,66.90,67.24,68.23,88.19,88.70,90.34,100.57
]

def fase_weibull(beta):
    if beta < 0.95:
        return "Mortalidade infantil", "taxa de falha decrescente"
    if beta <= 1.05:
        return "Vida útil / falhas aleatórias", "taxa de falha aproximadamente constante"
    return "Desgaste / envelhecimento (Wear-out)", "taxa de falha crescente"

def read_upload(upload):
    nome = upload.name.lower()
    if nome.endswith(".csv"):
        df = pd.read_csv(upload)
    else:
        df = pd.read_excel(upload)
    cols = {str(c).strip().lower(): c for c in df.columns}
    tempo_col = next((cols[k] for k in ("tempo","time","vida","life") if k in cols), None)
    falha_col = next((cols[k] for k in ("falha","evento","event","failure") if k in cols), None)
    if tempo_col is None:
        raise ValueError("O arquivo precisa ter uma coluna chamada Tempo.")
    tempos = pd.to_numeric(df[tempo_col], errors="coerce")
    valid = tempos.notna()
    tempos = tempos[valid].astype(float).tolist()
    if falha_col is None:
        eventos = [1] * len(tempos)
    else:
        ev = pd.to_numeric(df.loc[valid, falha_col], errors="coerce").fillna(1)
        eventos = [1 if float(x) != 0 else 0 for x in ev]
    return tempos, eventos

def pdf_resumo(nome, unidade, resultado):
    """Relatório completo: síntese, ranking e tabelas de confiabilidade de cada distribuição."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    except Exception:
        return None

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4),
        leftMargin=12*mm, rightMargin=12*mm, topMargin=12*mm, bottomMargin=12*mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SCASection", parent=styles["Heading2"], fontSize=13, leading=16, spaceBefore=8, spaceAfter=6))
    styles.add(ParagraphStyle(name="SCASmall", parent=styles["BodyText"], fontSize=8, leading=10))
    styles.add(ParagraphStyle(name="SCATitle", parent=styles["Title"], fontSize=19, leading=22, alignment=TA_CENTER))

    ranking = resultado["ranking"].copy()
    estudo = resultado["estudo"]
    w = resultado["weibull"]
    pref = resultado["selecao"]["modelo_preferencial"]

    def fmt(v, casas=3):
        try:
            return "-" if v is None or not np.isfinite(float(v)) else f"{float(v):.{casas}f}"
        except Exception:
            return "-"

    def pct(v, casas=1):
        try:
            return "-" if v is None or not np.isfinite(float(v)) else f"{float(v)*100:.{casas}f}%"
        except Exception:
            return "-"

    def tab(dados, widths=None, fontsize=7):
        t = Table(dados, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#5B3FD6")),
            ("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#D8D5CC")),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("ALIGN",(0,0),(-1,-1),"LEFT"),
            ("FONTSIZE",(0,0),(-1,-1),fontsize),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, colors.HexColor("#F7F5FF")]),
            ("LEFTPADDING",(0,0),(-1,-1),4), ("RIGHTPADDING",(0,0),(-1,-1),4),
        ]))
        return t

    story = [
        Paragraph("SCA - Sistema de Confiabilidade de Ativos", styles["SCATitle"]),
        Paragraph("Relatório completo para validação acadêmica", styles["Heading2"]),
        Spacer(1,4),
        Paragraph(f"<b>Estudo:</b> {nome}", styles["BodyText"]),
        Paragraph(f"<b>Data:</b> {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles["BodyText"]),
        Paragraph(f"<b>Base:</b> {estudo['n']} observações de vida; {estudo['falhas']} falhas; {estudo['censuras']} censuras.", styles["BodyText"]),
        Paragraph(f"<b>Modelo preferencial pelo AIC:</b> {pref}", styles["BodyText"]),
        Spacer(1,8), Paragraph("1. Ranking das distribuições", styles["SCASection"])
    ]

    rows = [["Rank","Distribuição","Log-Likelihood","AIC","BIC","Delta AIC","Peso Akaike","MTTF","B10","B50","B90","Tempo R alvo"]]
    for _,r in ranking.iterrows():
        rows.append([str(int(r["Ranking"])),str(r["Distribuição"]),fmt(r.get("Log-Likelihood")),fmt(r.get("AIC")),
                     fmt(r.get("BIC")),fmt(r.get("Delta AIC")),pct(r.get("Peso Akaike"),2),fmt(r.get("MTTF"),2),
                     fmt(r.get("B10"),2),fmt(r.get("B50"),2),fmt(r.get("B90"),2),fmt(r.get("Tempo R Alvo"),2)])
    story += [tab(rows,[12*mm,34*mm,24*mm,19*mm,19*mm,19*mm,22*mm,20*mm,18*mm,18*mm,18*mm,24*mm],6.5),
              Spacer(1,6),
              Paragraph("Menor AIC = modelo preferencial; Delta AIC <= 2 = modelo competitivo.",styles["SCASmall"]),
              Spacer(1,8), Paragraph("2. Weibull 2P",styles["SCASection"]),
              tab([["Beta",f"Eta ({unidade})",f"MTTF ({unidade})",f"B10 ({unidade})",f"B50 ({unidade})",f"B90 ({unidade})"],
                   [fmt(w["beta"],4),fmt(w["eta"],2),fmt(w["mttf"],2),fmt(w["b10"],2),fmt(w["b50"],2),fmt(w["b90"],2)]],
                  [28*mm]*6,8)]

    beta=float(w["beta"])
    fase = "mortalidade infantil / taxa decrescente" if beta < .95 else ("falhas aleatórias / taxa aproximadamente constante" if beta <= 1.05 else "desgaste/envelhecimento / taxa crescente")
    story += [Spacer(1,5), Paragraph(f"<b>Interpretação:</b> beta={beta:.4f}: {fase}. A leitura é estatística e não identifica sozinha o mecanismo físico.",styles["SCASmall"])]

    for i,(_,rr) in enumerate(ranking.iterrows(),1):
        modelo=str(rr["Distribuição"])
        info=resultado["modelos"][modelo]
        dist=info["dist"]; params=info["parametros"]
        story += [PageBreak(), Paragraph(f"3.{i} Análise completa - {modelo}",styles["SCASection"]),
                  Paragraph(f"<b>Ranking:</b> {int(rr['Ranking'])} | <b>AIC:</b> {fmt(rr.get('AIC'))} | <b>Delta AIC:</b> {fmt(rr.get('Delta AIC'))} | <b>Peso Akaike:</b> {pct(rr.get('Peso Akaike'),2)}",styles["BodyText"]),
                  Spacer(1,5)]

        met=[["Indicador","Resultado"],["Log-Likelihood",fmt(rr.get("Log-Likelihood"),4)],["AIC",fmt(rr.get("AIC"),4)],
             ["BIC",fmt(rr.get("BIC"),4)],["Delta AIC",fmt(rr.get("Delta AIC"),4)],["Peso Akaike",pct(rr.get("Peso Akaike"),2)],
             [f"MTTF ({unidade})",fmt(rr.get("MTTF"),2)],[f"B10 ({unidade})",fmt(rr.get("B10"),2)],
             [f"B50 ({unidade})",fmt(rr.get("B50"),2)],[f"B90 ({unidade})",fmt(rr.get("B90"),2)],
             [f"Tempo na confiabilidade-alvo ({unidade})",fmt(rr.get("Tempo R Alvo"),2)]]
        for k in ("MAE KM","RMSE KM","Desvio Máximo KM","Max Dev KM"):
            if k in rr.index: met.append([k,pct(rr.get(k),2)])
        story += [tab(met,[62*mm,42*mm],7.5),Spacer(1,8),
                  Paragraph("Tabela completa de níveis de confiabilidade",styles["SCASection"]),
                  Paragraph("R(t) é a confiabilidade; F(t)=1-R(t) é a probabilidade acumulada de falha.",styles["SCASmall"]),Spacer(1,3)]

        niveis=[1,.95,.90,.85,.80,.75,.70,.65,.60,.55,.50,.45,.40,.35,.30,.25,.20,.15,.10,.05,.01]
        cr=[["R(t)","F(t)",f"Tempo ({unidade})","Referência"]]
        for rlevel in niveis:
            tempo=0.0 if rlevel==1 else float(dist.isf(rlevel,*params))
            r_display = float(dist.sf(tempo,*params)) if rlevel==1 else rlevel
            ref="B10" if rlevel==.90 else ("B50" if rlevel==.50 else ("B90" if rlevel==.10 else ""))
            cr.append([f"{r_display*100:.2f}%" if rlevel==1 else f"{rlevel*100:.0f}%", f"{(1-r_display)*100:.2f}%" if rlevel==1 else f"{(1-rlevel)*100:.0f}%",fmt(tempo,2),ref])
        cr.append(["0%","100%","assintótico","não há tempo finito em distribuições não limitadas"])
        story += [tab(cr,[25*mm,25*mm,35*mm,80*mm],7),Spacer(1,8)]

        pontos=[("B10",rr.get("B10")),("Meta",rr.get("Tempo R Alvo")),("B50",rr.get("B50")),("MTTF",rr.get("MTTF")),("B90",rr.get("B90"))]
        tr=[["Referência",f"Tempo ({unidade})","R(t)","F(t)"]]
        for rot,tv in pontos:
            try:
                tv=float(tv)
                if np.isfinite(tv):
                    rv=float(dist.sf(tv,*params))
                    tr.append([rot,fmt(tv,2),pct(rv,2),pct(1-rv,2)])
            except Exception: pass
        story += [Paragraph("Pontos principais em escala de tempo",styles["SCASection"]),tab(tr,[35*mm,35*mm,30*mm,30*mm],7.5)]

    story += [PageBreak(),Paragraph("4. Observação para decisão de manutenção",styles["SCASection"]),
              Paragraph("MTTF, B10, B50, B90 e o tempo de uma confiabilidade-alvo são referências estatísticas. O intervalo de manutenção deve também considerar criticidade, modo de falha, detectabilidade e consequências.",styles["BodyText"])]

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()

with st.sidebar:
    st.markdown("## SCA")
    st.caption("Sistema de Confiabilidade de Ativos")
    st.markdown("---")
    pagina = st.radio("Navegação", ["Análise de Vida", "Metodologia", "MTBF e aplicação"], label_visibility="collapsed")
    st.markdown("---")
    st.caption("Versão DEMO • Validação acadêmica\n\nSem login • Sem acesso à base operacional")

st.markdown("# SCA — Sistema de Confiabilidade de Ativos")
st.markdown(
    '<div class="sca-card"><span class="sca-pill">VERSÃO PARA VALIDAÇÃO ACADÊMICA</span>'
    '<h3>Análise estatística aplicada à confiabilidade</h3>'
    '<p>Esta versão não grava dados no banco operacional. Os cálculos são executados apenas durante a sessão.</p></div>',
    unsafe_allow_html=True
)

if pagina == "Metodologia":
    st.markdown("## Metodologia da Análise de Vida")
    st.markdown("""
A unidade estatística desta análise é uma **observação de vida**, e não necessariamente um equipamento distinto. Um mesmo ativo pode contribuir com mais de uma vida quando um componente falha, é substituído/reparado e volta à operação.

**Falha** significa que o tempo até a ocorrência foi observado. **Censura à direita** significa que a falha ainda não foi observada: sabemos apenas que o componente sobreviveu pelo menos até o tempo informado. As censuras não devem ser descartadas, pois carregam informação de sobrevivência.

O SCA usa **Kaplan-Meier** como referência empírica de confiabilidade e ajusta distribuições paramétricas candidatas por **máxima verossimilhança**, considerando falhas e censuras. O **AIC** é o critério principal de comparação: menor AIC indica maior suporte relativo dentro do conjunto de modelos avaliados; **ΔAIC ≤ 2** indica modelos estatisticamente competitivos. Isso não prova que uma distribuição seja a “verdadeira”.

Na **Weibull 2P**, β < 1 é compatível com taxa de falha decrescente; β ≈ 1 com taxa aproximadamente constante; e β > 1 com taxa crescente. Essa leitura descreve o comportamento estatístico e **não identifica, sozinha, o mecanismo físico da falha**.

**R(t)** é a probabilidade estimada de sobreviver além de t; **F(t)=1−R(t)** é a probabilidade acumulada de falhar até t. **B10, B50 e B90** são tempos associados a 10%, 50% e 90% de falhas acumuladas. **MTTF** é a vida média estimada até a falha e não é, automaticamente, um intervalo de manutenção preventiva.

A decisão de manutenção deve combinar os resultados estatísticos com **criticidade, modo de falha, detectabilidade, consequências, condição do ativo e custos**.
""")
    st.info("**Escopo desta tela:** análise de vida de componentes/itens. MTBF é tratado separadamente como indicador de confiabilidade operacional de itens reparáveis.")
    st.stop()

if pagina == "MTBF e aplicação":
    st.markdown("## MTBF e aplicação no SCA")
    st.markdown("""
O **MTBF (Mean Time Between Failures)** responde a uma pergunta diferente da análise de vida: **com que frequência um item reparável apresenta falhas durante sua operação?**

No SCA, o MTBF deve ser tratado em um módulo de **Confiabilidade Operacional**, separado da Análise de Vida. Ele poderá ser analisado nos níveis **frota → ativo → sistema → componente/posição funcional**, desde que esteja claro o que conta como falha e qual foi a exposição operacional.

Para uma estimativa operacional simples, usa-se **tempo total de operação/exposição ÷ número de falhas**. Equipamentos que operaram e não falharam também contribuem com exposição. Já a Análise de Vida utiliza cada vida observada — falha ou censura — para modelar a distribuição do tempo até falhar.
""")
    st.markdown("### Exemplo — cilindro de elevação direito")
    st.dataframe(pd.DataFrame([
        {"Ativo":"EH-151","Vida observada (h)":3200,"Evento":"Falha","Uso na análise":"Vida até falha"},
        {"Ativo":"EH-152","Vida observada (h)":4600,"Evento":"Censura","Uso na análise":"Sobreviveu pelo menos até 4.600 h"},
        {"Ativo":"EH-153","Vida observada (h)":2850,"Evento":"Falha","Uso na análise":"Vida até falha"},
    ]), hide_index=True, use_container_width=True)
    st.warning("**Não confundir:** MTTF descreve a vida média estimada até a falha; MTBF descreve o intervalo/exposição média entre ocorrências de falha de um item reparável. Nenhum deles deve ser convertido automaticamente em periodicidade preventiva.")
    st.stop()

st.markdown("## Análise de Vida")
fonte = st.selectbox(
    "Conjunto de dados",
    ["Exercício — Eixo Mecânico (30 falhas)", "Exercício — Quebra de Bombas (28 falhas)", "Enviar meu arquivo Excel/CSV"],
    index=0
)

if fonte.startswith("Exercício — Eixo"):
    tempos, eventos, nome_default = EIXO, [1]*len(EIXO), "Eixo Mecânico"
elif fonte.startswith("Exercício — Quebra"):
    tempos, eventos, nome_default = BOMBAS, [1]*len(BOMBAS), "Quebra de Bombas"
else:
    upload = st.file_uploader("Arquivo com colunas Tempo e Falha (1 = falha; 0 = censura)", type=["xlsx","xls","csv"], help="Cada linha representa uma observação de vida. A identificação do ativo pode existir no arquivo para rastreabilidade, mas o motor usa Tempo e Falha/Censura no cálculo.")
    if upload is None:
        st.info("Envie um arquivo para iniciar. **Cada linha deve representar uma observação de vida**, e não necessariamente um equipamento diferente.")
        st.stop()
    try:
        tempos, eventos = read_upload(upload)
        nome_default = upload.name.rsplit(".",1)[0]
    except Exception as ex:
        st.error(str(ex))
        st.stop()

st.markdown("""
<div class="sca-card"><span class="sca-pill">COMO LER A BASE</span>
<p><b>Cada linha = uma observação de vida.</b> Falha: o tempo até a falha foi observado. Censura: o item ainda não falhou e sobreviveu pelo menos até o tempo informado.</p>
<p class="small-note">Um mesmo equipamento pode fornecer várias vidas ao longo do histórico após reparos ou substituições do componente.</p></div>
""", unsafe_allow_html=True)

c1,c2,c3 = st.columns(3)
with c1:
    nome = st.text_input("Nome do estudo", value=nome_default)
with c2:
    unidade = st.text_input("Unidade de vida", value="h")
with c3:
    alvo_pct = st.number_input("Confiabilidade-alvo (%)", min_value=1.0, max_value=99.9, value=85.0, step=1.0)

usar_missao = st.checkbox("Avaliar uma missão específica")
missao = None
if usar_missao:
    missao = st.number_input(f"Missão ({unidade})", min_value=0.01, value=float(np.median(tempos)), step=1.0)

if not st.button("Executar análise", type="primary", use_container_width=True):
    st.stop()

try:
    resultado = analisar_confiabilidade(
        tempos=tempos, eventos=eventos, missao=missao,
        confiabilidade_alvo=alvo_pct/100.0,
        nome_equipamento=nome, unidade=unidade
    )
except Exception as ex:
    st.error(f"Erro na análise: {ex}")
    st.stop()

ranking = resultado["ranking"]
estudo = resultado["estudo"]
sel = resultado["selecao"]
w = resultado["weibull"]
pref_nome = sel["modelo_preferencial"]
pref_row = ranking.iloc[0]
pref_info = resultado["modelos"][pref_nome]
fase, comportamento = fase_weibull(float(w["beta"]))

st.markdown("## Síntese da análise")
m1,m2,m3,m4 = st.columns(4)
m1.metric("Observações de vida", estudo["n"])
m2.metric("Falhas", estudo["falhas"])
m3.metric("Censuras", estudo["censuras"])
m4.metric("Modelo preferencial", pref_nome)

st.markdown(
    f'<div class="sca-card"><span class="sca-pill">CONCLUSÃO TÉCNICA</span>'
    f'<h3>{pref_nome} apresentou o menor AIC</h3>'
    f'<p>Weibull 2P: β = <b>{w["beta"]:.2f}</b> — {fase}; {comportamento}.</p>'
    f'<p class="small-note">A seleção do modelo é estatística. A interpretação física deve considerar modos de falha e evidências de campo.</p></div>',
    unsafe_allow_html=True
)

st.markdown("### Curva e tabela de confiabilidade")
dist = pref_info["dist"]; params = pref_info["parametros"]
t_alvo = float(pref_row["Tempo R Alvo"])
refs = [
    ("Meta de confiabilidade", t_alvo),
    ("B10", float(pref_row["B10"])) if pd.notna(pref_row.get("B10")) else None,
    ("B50", float(pref_row["B50"])) if pd.notna(pref_row.get("B50")) else None,
    ("MTTF", float(pref_row["MTTF"])) if pd.notna(pref_row.get("MTTF")) else None,
    ("B90", float(pref_row["B90"])) if pd.notna(pref_row.get("B90")) else None,
]
if missao is not None:
    refs.append(("Missão", float(missao)))
refs = [r for r in refs if r is not None and np.isfinite(r[1])]
rows=[]
for rot,t in sorted(refs,key=lambda z:z[1]):
    r=float(dist.sf(t,*params))
    rows.append({"Referência":rot, f"Tempo ({unidade})":round(t,2), "R(t)":f"{r*100:.1f}%", "F(t)":f"{(1-r)*100:.1f}%"})

cg,ct = st.columns([1.6,1], gap="large")
with cg:
    xmax=max([x[1] for x in refs]+tempos)*1.08
    x=np.linspace(0,xmax,450)
    fig,ax=plt.subplots(figsize=(9,5))
    ax.plot(x,dist.sf(x,*params),label=f"Modelo {pref_nome}",linewidth=2.2)
    ax.axhline(alvo_pct/100,linestyle="--",label=f"Meta {alvo_pct:.0f}%")
    ax.axvline(t_alvo,linestyle=":")
    ax.scatter([t_alvo],[alvo_pct/100],s=55)
    ax.annotate(f"{alvo_pct:.0f}% em {t_alvo:.1f} {unidade}",(t_alvo,alvo_pct/100),xytext=(8,12),textcoords="offset points")
    if missao is not None:
        rm=float(dist.sf(missao,*params))
        ax.scatter([missao],[rm],s=55)
        ax.annotate(f"Missão\nR={rm*100:.1f}%",(missao,rm),xytext=(8,-28),textcoords="offset points")
    ax.set_ylim(0,1.05); ax.set_xlabel(f"Tempo ({unidade})"); ax.set_ylabel("Confiabilidade R(t)")
    ax.grid(alpha=.2); ax.legend(fontsize=8)
    st.pyplot(fig,use_container_width=True); plt.close(fig)
with ct:
    st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    st.success(f"Meta de {alvo_pct:.0f}%: aproximadamente **{t_alvo:.2f} {unidade}**.")

st.markdown("## Comparação entre distribuições")
show = ranking.copy()
for col in ["Peso Akaike","R Missão","F Missão"]:
    if col in show.columns:
        show[col] = show[col].apply(lambda v: f"{v*100:.2f}%" if pd.notna(v) else "")
st.dataframe(show, hide_index=True, use_container_width=True)

st.markdown("## Weibull 2P")
wc1,wc2,wc3,wc4,wc5 = st.columns(5)
wc1.metric("β",f'{w["beta"]:.4f}')
wc2.metric("η",f'{w["eta"]:.2f} {unidade}')
wc3.metric("MTTF",f'{w["mttf"]:.2f} {unidade}')
wc4.metric("B10",f'{w["b10"]:.2f} {unidade}')
wc5.metric("B50",f'{w["b50"]:.2f} {unidade}')
st.info(f"**Interpretação:** {fase}; {comportamento}. β caracteriza o comportamento estatístico da taxa de falha.")

st.markdown("## Diagnósticos por distribuição")
modelo = st.selectbox("Distribuição para inspeção", ranking["Distribuição"].tolist())
rr = ranking.loc[ranking["Distribuição"]==modelo].iloc[0]
mi = resultado["modelos"][modelo]
d=mi["dist"]; p=mi["parametros"]
x=np.linspace(0,max(tempos)*1.10,400)
sf=d.sf(x,*p); pdf=d.pdf(x,*p)
with np.errstate(divide="ignore",invalid="ignore"):
    haz=np.where(sf>1e-12,pdf/sf,np.nan)

g1,g2=st.columns(2)
with g1:
    fig,ax=plt.subplots(figsize=(7,4)); ax.plot(x,sf); ax.set_title(f"Confiabilidade R(t) — {modelo}"); ax.grid(alpha=.2); st.pyplot(fig,use_container_width=True); plt.close(fig)
    st.caption("Interpretação: probabilidade estimada de sobreviver além do tempo t.")
with g2:
    fig,ax=plt.subplots(figsize=(7,4)); ax.plot(x,pdf); ax.set_title(f"Densidade f(t) — {modelo}"); ax.grid(alpha=.2); st.pyplot(fig,use_container_width=True); plt.close(fig)
    st.caption("Interpretação: concentração dos tempos de falha previstos pelo modelo.")

g3,g4=st.columns(2)
with g3:
    fig,ax=plt.subplots(figsize=(7,4)); ax.plot(x,haz); ax.set_title(f"Taxa de risco h(t) — {modelo}"); ax.grid(alpha=.2); st.pyplot(fig,use_container_width=True); plt.close(fig)
    st.caption("Interpretação: intensidade instantânea de falha entre itens que sobreviveram até t.")
with g4:
    falhas=sorted([t for t,e in zip(tempos,eventos) if e==1])
    probs=(np.arange(1,len(falhas)+1)-.5)/len(falhas)
    q=np.asarray(d.ppf(probs,*p),dtype=float); obs=np.asarray(falhas,dtype=float)
    mask=np.isfinite(q)&np.isfinite(obs)
    fig,ax=plt.subplots(figsize=(7,4)); ax.scatter(q[mask],obs[mask],s=24)
    if mask.any():
        mn=min(q[mask].min(),obs[mask].min()); mx=max(q[mask].max(),obs[mask].max()); ax.plot([mn,mx],[mn,mx],linestyle="--")
    ax.set_title(f"Q-Q Plot — {modelo}"); ax.set_xlabel("Quantis teóricos"); ax.set_ylabel("Observado"); ax.grid(alpha=.2)
    st.pyplot(fig,use_container_width=True); plt.close(fig)
    st.caption("Interpretação: proximidade da diagonal indica maior compatibilidade do formato dos dados com o modelo.")

pdf = pdf_resumo(nome, unidade, resultado)
if pdf:
    st.download_button("📄 Baixar relatório completo PDF", data=pdf, file_name=f"SCA_{nome.replace(' ','_')}.pdf", mime="application/pdf", use_container_width=True)

st.caption("SCA • Versão para validação acadêmica • Resultados estatísticos devem ser interpretados em conjunto com evidências de engenharia e manutenção.")
