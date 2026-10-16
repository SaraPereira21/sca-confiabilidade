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
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    except Exception:
        return None
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=15*mm, rightMargin=15*mm, topMargin=15*mm, bottomMargin=15*mm)
    sty = getSampleStyleSheet()
    ranking = resultado["ranking"]
    estudo = resultado["estudo"]
    w = resultado["weibull"]
    pref = resultado["selecao"]["modelo_preferencial"]
    story = [
        Paragraph("SCA — Relatório para Validação Acadêmica", sty["Title"]),
        Spacer(1,8),
        Paragraph(f"<b>Estudo:</b> {nome}", sty["BodyText"]),
        Paragraph(f"<b>Data:</b> {datetime.now().strftime('%d/%m/%Y %H:%M')}", sty["BodyText"]),
        Paragraph(f"<b>Base:</b> {estudo['n']} observações, {estudo['falhas']} falhas, {estudo['censuras']} censuras.", sty["BodyText"]),
        Paragraph(f"<b>Modelo preferencial pelo AIC:</b> {pref}", sty["BodyText"]),
        Spacer(1,8),
        Paragraph("Ranking das distribuições", sty["Heading2"]),
    ]
    rows = [["Rank","Distribuição","AIC","ΔAIC","Peso"]]
    for _,r in ranking.iterrows():
        rows.append([str(int(r["Ranking"])), str(r["Distribuição"]), f'{r["AIC"]:.3f}', f'{r["Delta AIC"]:.3f}', f'{r["Peso Akaike"]*100:.1f}%'])
    t=Table(rows, colWidths=[18*mm,50*mm,30*mm,30*mm,30*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#5b3fd6")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("GRID",(0,0),(-1,-1),.35,colors.HexColor("#d9d5ca")),
        ("FONTSIZE",(0,0),(-1,-1),8),
    ]))
    story += [t, Spacer(1,10),
              Paragraph("Weibull 2P", sty["Heading2"]),
              Paragraph(
                  f"β={w['beta']:.4f}; η={w['eta']:.2f} {unidade}; "
                  f"MTTF={w['mttf']:.2f} {unidade}; B10={w['b10']:.2f} {unidade}; "
                  f"B50={w['b50']:.2f} {unidade}; B90={w['b90']:.2f} {unidade}.",
                  sty["BodyText"]
              )]
    doc.build(story)
    buf.seek(0)
    return buf.getvalue()

with st.sidebar:
    st.markdown("## SCA")
    st.caption("Sistema de Confiabilidade de Ativos")
    st.markdown("---")
    pagina = st.radio("Navegação", ["Análise", "Metodologia"], label_visibility="collapsed")
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
    st.markdown("## Metodologia")
    st.markdown("""
O SCA compara distribuições candidatas por máxima verossimilhança e utiliza o **AIC** como critério principal de seleção.
Modelos com **ΔAIC ≤ 2** são apresentados como estatisticamente competitivos. A curva **Kaplan-Meier** é usada como
referência empírica da confiabilidade observada, inclusive quando há censura à direita.

Na análise Weibull 2P, **β < 1** é compatível com mortalidade infantil, **β ≈ 1** com falhas aproximadamente aleatórias
e **β > 1** com taxa de falha crescente / desgaste. Essa interpretação descreve o comportamento estatístico e não,
isoladamente, o mecanismo físico da falha.

B10, B50, B90, MTTF e tempos associados a uma confiabilidade-alvo são referências estatísticas. O SCA não transforma
automaticamente esses valores em intervalos obrigatórios de manutenção preventiva.
""")
    st.stop()

st.markdown("## Análise")
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
    upload = st.file_uploader("Arquivo com colunas Tempo e Falha (1=falha; 0=censura)", type=["xlsx","xls","csv"])
    if upload is None:
        st.info("Envie um arquivo para iniciar.")
        st.stop()
    try:
        tempos, eventos = read_upload(upload)
        nome_default = upload.name.rsplit(".",1)[0]
    except Exception as ex:
        st.error(str(ex))
        st.stop()

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
m1.metric("Observações", estudo["n"])
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
    st.download_button("📄 Baixar relatório-resumo PDF", data=pdf, file_name=f"SCA_{nome.replace(' ','_')}.pdf", mime="application/pdf", use_container_width=True)
