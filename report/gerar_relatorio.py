"""
Gera o relatorio tecnico em PDF com graficos da EDA.
Execucao: python report/gerar_relatorio.py
"""
import io, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, Image, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

ROOT   = Path(__file__).parent.parent
OUTPUT = Path(__file__).parent / "Relatorio_Tecnico.pdf"
DATA   = ROOT / "data" / "processed" / "olist_analytical.csv"

if not DATA.exists():
    print(f"ERRO: {DATA}\nExecute: python src/data_ingestion.py")
    sys.exit(1)

print("Carregando dados...")
df = pd.read_csv(DATA)
df = df[df["order_status"] == "delivered"].dropna(subset=["review_score"]).copy()
for c in ["order_purchase_timestamp","order_approved_at",
          "order_delivered_carrier_date","order_delivered_customer_date",
          "order_estimated_delivery_date"]:
    df[c] = pd.to_datetime(df[c], errors="coerce")

df["delivery_delay_days"] = (df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]).dt.days
df["shipping_days"]       = (df["order_delivered_carrier_date"]  - df["order_approved_at"]).dt.days
df["is_late"]             = (df["delivery_delay_days"] > 0).astype(int)
df["freight_ratio"]       = df["freight_total"] / df["price_total"].replace(0, np.nan)
df["target"]              = (df["review_score"] >= 4).astype(int)
df["satisfaction"]        = df["target"].map({1: "Satisfeito", 0: "Insatisfeito"})
df["cross_state"]         = (df["customer_state"] != df["seller_state"]).astype(int)

sns.set_theme(style="whitegrid", font_scale=0.85)
plt.rcParams["figure.autolayout"] = True

styles = getSampleStyleSheet()
C1, C2 = "#1a1a2e", "#2c3e50"

titulo_style    = ParagraphStyle("titulo",  parent=styles["Title"],   fontSize=16, leading=20, spaceAfter=4,  alignment=TA_CENTER, textColor=colors.HexColor(C1))
subtitulo_style = ParagraphStyle("sub",     parent=styles["Normal"],  fontSize=10, leading=13, spaceAfter=3,  alignment=TA_CENTER, textColor=colors.HexColor("#444"))
h1_style        = ParagraphStyle("h1",      parent=styles["Heading1"],fontSize=13, leading=16, spaceBefore=10,spaceAfter=3,  textColor=colors.HexColor(C1))
h2_style        = ParagraphStyle("h2",      parent=styles["Heading2"],fontSize=11, leading=14, spaceBefore=7, spaceAfter=2,  textColor=colors.HexColor(C2))
body_style      = ParagraphStyle("body",    parent=styles["Normal"],  fontSize=9.5,leading=13, spaceAfter=4,  alignment=TA_JUSTIFY)
caption_style   = ParagraphStyle("caption", parent=styles["Normal"],  fontSize=8,  leading=10, spaceAfter=6,  alignment=TA_CENTER, textColor=colors.grey)
ref_style       = ParagraphStyle("ref",     parent=styles["Normal"],  fontSize=9,  leading=12, spaceAfter=3,  leftIndent=16, firstLineIndent=-16)

def hr():    return HRFlowable(width="100%", thickness=0.4, color=colors.HexColor("#cccccc"), spaceAfter=4, spaceBefore=2)
def h1(t):   return Paragraph(t, h1_style)
def h2(t):   return Paragraph(t, h2_style)
def p(t):    return Paragraph(t, body_style)
def sp(n=5): return Spacer(1, n)
def cap(t):  return Paragraph(t, caption_style)
def ref(t):  return Paragraph(t, ref_style)

def fig_to_image(fig, width=16*cm):
    w_in, h_in = fig.get_size_inches()
    height = width * (h_in / w_in)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    img = Image(buf, width=width, height=height)
    img.hAlign = "CENTER"
    return img

def make_table(data, col_widths, header_color=C1):
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0), colors.HexColor(header_color)),
        ("TEXTCOLOR",     (0,0),(-1,0), colors.white),
        ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1), 8.5),
        ("LEADING",       (0,0),(-1,-1), 11),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, colors.HexColor("#f0f4f8")]),
        ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#cccccc")),
        ("TOPPADDING",    (0,0),(-1,-1), 4),
        ("BOTTOMPADDING", (0,0),(-1,-1), 4),
    ]))
    return t

# --- Graficos ---
print("Gerando graficos...")

fig, axes = plt.subplots(1, 2, figsize=(9, 2.8))
tc = df["target"].value_counts()
axes[0].bar(["Insatisfeito","Satisfeito"], [tc.get(0,0), tc.get(1,0)],
            color=["#e74c3c","#2ecc71"], edgecolor="white", width=0.5)
axes[0].set_title("Distribuicao da variavel-alvo"); axes[0].set_ylabel("Contagem")
for i,v in enumerate([tc.get(0,0), tc.get(1,0)]):
    axes[0].text(i, v+300, f"{v:,}\n({v/len(df):.1%})", ha="center", fontsize=8)
axes[1].hist(df["review_score"], bins=5, color="#3498db", edgecolor="white", rwidth=0.8)
axes[1].set_title("Distribuicao do review_score"); axes[1].set_xlabel("review_score")
axes[1].set_ylabel("Contagem"); axes[1].set_xticks([1,2,3,4,5])
fig_dist = fig_to_image(fig, width=14*cm)

fig, axes = plt.subplots(1, 2, figsize=(9, 2.8))
late_score = df.groupby("is_late")["review_score"].mean()
axes[0].bar(["No prazo","Com atraso"], late_score.values, color=["#2ecc71","#e74c3c"], edgecolor="white", width=0.5)
axes[0].set_title("Nota media por status de entrega"); axes[0].set_ylabel("Score medio"); axes[0].set_ylim(0,5)
for i,v in enumerate(late_score.values): axes[0].text(i, v+0.1, f"{v:.2f}", ha="center", fontweight="bold", fontsize=9)
sat = df[df["target"]==1]["delivery_delay_days"].clip(-30,60)
ins = df[df["target"]==0]["delivery_delay_days"].clip(-30,60)
axes[1].hist(sat, bins=40, alpha=0.6, color="#2ecc71", label="Satisfeito", density=True)
axes[1].hist(ins, bins=40, alpha=0.6, color="#e74c3c", label="Insatisfeito", density=True)
axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
axes[1].set_title("Dias de atraso por classe"); axes[1].set_xlabel("delivery_delay_days"); axes[1].legend(fontsize=8)
fig_h1 = fig_to_image(fig)

fig, axes = plt.subplots(1, 2, figsize=(9, 2.8))
bp = axes[0].boxplot([df[df["target"]==1]["freight_ratio"].dropna().clip(0,2),
                      df[df["target"]==0]["freight_ratio"].dropna().clip(0,2)],
                     tick_labels=["Satisfeito","Insatisfeito"], patch_artist=True)
bp["boxes"][0].set_facecolor("#2ecc71"); bp["boxes"][1].set_facecolor("#e74c3c")
axes[0].set_title("Freight ratio por classe"); axes[0].set_ylabel("freight_ratio")
sample = df.dropna(subset=["freight_ratio"]).sample(3000, random_state=42)
axes[1].scatter(sample["freight_ratio"].clip(0,2), sample["review_score"], alpha=0.15, color="#3498db", s=6)
axes[1].set_xlabel("freight_ratio"); axes[1].set_ylabel("review_score"); axes[1].set_title("Frete relativo vs Avaliacao")
fig_h2 = fig_to_image(fig)

cat_score = (df.groupby("product_category_name_english")["review_score"]
               .agg(["mean","count"]).query("count >= 200").sort_values("mean").tail(16))
fig, ax = plt.subplots(figsize=(9, 3.8))
bar_c = ["#e74c3c" if v<3.8 else "#f1c40f" if v<4.2 else "#2ecc71" for v in cat_score["mean"]]
ax.barh(cat_score.index, cat_score["mean"], color=bar_c, edgecolor="white", height=0.65)
ax.axvline(df["review_score"].mean(), color="black", linestyle="--", linewidth=1,
           label=f"Media ({df['review_score'].mean():.2f})")
ax.set_xlabel("Nota media"); ax.set_title("Top 16 categorias por nota media (min. 200 pedidos)")
ax.legend(fontsize=8); ax.set_xlim(3.2, 5)
fig_h3 = fig_to_image(fig, width=14*cm)

fig, axes = plt.subplots(1, 2, figsize=(9, 2.8))
cross_late  = df.groupby("cross_state")["is_late"].mean()
cross_score = df.groupby("cross_state")["review_score"].mean()
axes[0].bar(["Mesmo estado","Interestadual"], cross_late.values,  color=["#3498db","#e67e22"], edgecolor="white", width=0.5)
axes[0].set_title("Taxa de atraso por tipo de rota"); axes[0].set_ylabel("Taxa de atraso")
for i,v in enumerate(cross_late.values): axes[0].text(i, v+0.004, f"{v:.1%}", ha="center", fontweight="bold", fontsize=9)
axes[1].bar(["Mesmo estado","Interestadual"], cross_score.values, color=["#3498db","#e67e22"], edgecolor="white", width=0.5)
axes[1].set_title("Nota media por tipo de rota"); axes[1].set_ylabel("Score medio"); axes[1].set_ylim(3.6,4.4)
for i,v in enumerate(cross_score.values): axes[1].text(i, v+0.01, f"{v:.2f}", ha="center", fontweight="bold", fontsize=9)
fig_h4 = fig_to_image(fig)

df["parcelas_grupo"] = pd.cut(df["payment_installments"], bins=[0,1,3,6,24], labels=["1x","2-3x","4-6x","7x+"])
fig, axes = plt.subplots(1, 2, figsize=(9, 2.8))
parc_score = df.groupby("parcelas_grupo", observed=True)["review_score"].mean()
parc_price = df.groupby("parcelas_grupo", observed=True)["price_total"].median()
c4 = ["#2ecc71","#f1c40f","#e67e22","#e74c3c"]
axes[0].bar(parc_score.index, parc_score.values, color=c4, edgecolor="white", width=0.6)
axes[0].set_title("Nota media por parcelas"); axes[0].set_ylabel("Score medio"); axes[0].set_ylim(3.6,4.4)
for i,v in enumerate(parc_score.values): axes[0].text(i, v+0.01, f"{v:.2f}", ha="center", fontweight="bold", fontsize=9)
axes[1].bar(parc_price.index, parc_price.values, color=c4, edgecolor="white", width=0.6)
axes[1].set_title("Ticket mediano por parcelas"); axes[1].set_ylabel("Valor mediano (R$)")
for i,v in enumerate(parc_price.values): axes[1].text(i, v+1, f"R${v:.0f}", ha="center", fontweight="bold", fontsize=9)
fig_h5 = fig_to_image(fig)

# --- Montar PDF ---
print("Montando PDF...")
doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4,
    rightMargin=2.2*cm, leftMargin=2.2*cm, topMargin=2.2*cm, bottomMargin=2.2*cm)

story = []

story += [
    sp(50),
    Paragraph("Previsao de Satisfacao de Clientes", titulo_style),
    Paragraph("Olist Brazilian E-Commerce Dataset", subtitulo_style),
    sp(12), hr(), sp(6),
    Paragraph("Trabalho Final - Data Science", subtitulo_style),
    Paragraph("Universidade de Passo Fundo - UPF", subtitulo_style),
    sp(3), Paragraph("Autor: <b>Murilo Moreira Mognon</b>", subtitulo_style),
    sp(3), Paragraph("Repositorio: https://github.com/mognon-murilo/Trabalho_Final_DataScience__", subtitulo_style),
    sp(3), Paragraph("2025", subtitulo_style),
    PageBreak(),
]

story.append(sp(18))
story += [
    h1("1. Introducao"), hr(),
    p("O presente trabalho desenvolve um modelo de aprendizado de maquina para prever a satisfacao "
      "de clientes de e-commerce. O problema e formulado como <b>classificacao binaria</b>: "
      "dado um conjunto de informacoes sobre um pedido (prazo de entrega, valor, frete, categoria), "
      "o modelo deve prever se o cliente ficara <b>satisfeito</b> (nota >= 4) ou "
      "<b>insatisfeito</b> (nota <= 3). Identificar clientes insatisfeitos antecipadamente "
      "permite acoes proativas como reembolsos automaticos e contato preventivo, "
      "reduzindo churn e aumentando retencao."),
    h2("Hipoteses iniciais"),
    p("<b>H1.</b> Atraso na entrega e o principal fator de insatisfacao.<br/>"
      "<b>H2.</b> Frete alto em relacao ao valor do pedido reduz a satisfacao.<br/>"
      "<b>H3.</b> A categoria do produto influencia a expectativa e a avaliacao.<br/>"
      "<b>H4.</b> Pedidos interestaduais tem mais atrasos e menor satisfacao.<br/>"
      "<b>H5.</b> Pagamentos parcelados estao associados a piores avaliacoes."),
]

story.append(sp(18))
story += [
    h1("2. Dataset"), hr(),
    p("O dataset e o <b>Olist Brazilian E-Commerce Dataset</b> "
      "(https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), "
      "com pedidos reais de 2016 a 2018 em 9 tabelas relacionais. "
      "Apos join: ~99.441 linhas. Apos limpeza: <b>86.811 linhas x 138 colunas</b>."),
    h2("2.1 Dicionario de Variaveis"),
    make_table([
        ["Variavel", "Origem", "Descricao"],
        ["order_status",                  "orders",      "Status do pedido (delivered, canceled...)"],
        ["order_delivered_customer_date", "orders",      "Data de entrega ao cliente"],
        ["order_estimated_delivery_date", "orders",      "Data estimada de entrega"],
        ["price / freight_value",         "order_items", "Preco unitario e frete por item"],
        ["review_score",                  "reviews",     "Nota de 1 a 5 - variavel-alvo"],
        ["payment_type",                  "payments",    "Tipo de pagamento (credit_card, boleto...)"],
        ["payment_installments",          "payments",    "Numero de parcelas"],
        ["customer_state / seller_state", "clientes/vendedores", "Estado do cliente e vendedor"],
        ["product_category_name_english", "products",   "Categoria do produto em ingles"],
    ], [4.5*cm, 3*cm, 8.5*cm]),
    sp(5),
    h2("2.2 Features Criadas"),
    make_table([
        ["Feature",                 "Descricao"],
        ["delivery_delay_days",     "Dias entre entrega real e estimada (negativo = adiantado)"],
        ["shipping_days",           "Dias entre aprovacao do pedido e envio ao transportador"],
        ["estimated_delivery_days", "Prazo estimado em dias no momento da compra"],
        ["is_late",                 "Flag: 1 se chegou apos a data estimada"],
        ["freight_ratio",           "Proporcao do frete sobre o valor total do pedido"],
        ["target",                  "1 = satisfeito (score >= 4), 0 = insatisfeito (score <= 3)"],
    ], [5*cm, 11*cm], header_color=C2),
    sp(5),
    h2("2.3 Pre-processamento"),
    p("<b>1.</b> Filtragem: apenas pedidos 'delivered' (unicos com review confiavel).<br/>"
      "<b>2.</b> Remocao de nulos no target (review_score ausente).<br/>"
      "<b>3.</b> Ausentes: mediana para numericas, 'unknown' para categoricas.<br/>"
      "<b>4.</b> Outliers: IQR x 3,0 em price_total, freight_total, delivery_delay_days, payment_value.<br/>"
      "<b>5.</b> Encoding: One-Hot com drop_first=True em payment_type, customer_state, seller_state, product_category.<br/>"
      "<b>6.</b> Normalizacao: StandardScaler nas 11 features numericas.<br/>"
      "<b>7.</b> Split: 80% treino (69.449) / 20% teste (17.362), estratificado."),
]

story.append(sp(18))
story += [h1("3. Analise Exploratoria"), hr()]

story.append(KeepTogether([
    h2("Distribuicao da Variavel-alvo"),
    fig_dist,
    cap("Figura 1 - Distribuicao do target (esq.) e do review_score (dir.). Dataset desbalanceado: 79% satisfeitos / 21% insatisfeitos."),
]))
story.append(KeepTogether([
    h2("H1 - Atraso na entrega impacta negativamente a satisfacao"),
    fig_h1,
    cap("Figura 2 - Nota media por status de entrega (esq.) e distribuicao de delivery_delay_days por classe (dir.)."),
    p("Pedidos com atraso tem nota media 2,4 vs 4,3 no prazo. 61% dos atrasados receberam nota <= 3. "
      "<b>Hipotese confirmada</b> - is_late e delivery_delay_days sao as features mais importantes."),
]))
story.append(KeepTogether([
    h2("H2 - Frete alto reduz a satisfacao"),
    fig_h2,
    cap("Figura 3 - Freight ratio por classe (esq.) e scatter freight_ratio vs review_score (dir.)."),
    p("Insatisfeitos tem mediana de freight_ratio maior. Pedidos com ratio > 0,3 tem insatisfacao 38% maior. "
      "<b>Hipotese parcialmente confirmada.</b>"),
]))
story.append(KeepTogether([
    h2("H3 - Categoria do produto influencia a avaliacao"),
    fig_h3,
    cap("Figura 4 - Top 16 categorias por nota media (min. 200 pedidos)."),
    p("Variacao significativa entre categorias (3,2 a 4,7). "
      "<b>Hipotese confirmada</b> - categoria incluida via One-Hot Encoding."),
]))
story.append(KeepTogether([
    h2("H4 - Pedidos interestaduais tem mais atrasos"),
    fig_h4,
    cap("Figura 5 - Taxa de atraso e nota media por tipo de rota."),
    p("Interestaduais: taxa de atraso 18% maior e nota 0,3 ponto abaixo. <b>Hipotese confirmada.</b>"),
]))
story.append(KeepTogether([
    h2("H5 - Pagamento parcelado associado a piores avaliacoes"),
    fig_h5,
    cap("Figura 6 - Nota media e ticket mediano por faixa de parcelas."),
    p("Pedidos com 7+ parcelas: ticket 3,2x maior e nota 0,4 ponto abaixo da media. "
      "<b>Hipotese parcialmente confirmada.</b>"),
]))
story.append(KeepTogether([
    h2("Estatisticas Descritivas"),
    make_table([
        ["Feature",              "Media",  "Mediana","Desvio Padrao","Min",    "Max"],
        ["review_score",         "4,09",   "5,00",   "1,32",        "1",      "5"],
        ["delivery_delay_days",  "-10,8",  "-12,0",  "9,7",         "-92",    "52"],
        ["shipping_days",        "2,6",    "2,0",    "2,1",         "0",      "30"],
        ["freight_ratio",        "0,17",   "0,13",   "0,15",        "0,01",   "2,0"],
        ["price_total",          "R$121",  "R$74",   "R$183",       "R$0,85", "R$6.735"],
        ["payment_installments", "2,9",    "1,0",    "3,2",         "1",      "24"],
    ], [4.5*cm, 2*cm, 2*cm, 3*cm, 2*cm, 2.5*cm]),
    cap("Tabela 1 - Estatisticas descritivas das principais features numericas."),
]))

story.append(sp(18))
story += [
    h1("4. Modelagem"), hr(),
    p("Todos os modelos foram implementados em <b>NumPy puro</b>, sem frameworks de ML no treinamento."),
    h2("4.1 Regressao Logistica"),
    p("Gradient descent com regularizacao L2 (C=0,5, lr=0,05, 200 epocas) e class_weight='balanced' "
      "para compensar o desbalanceamento 79/21%. "
      "<b>Justificativa:</b> modelo interpretavel e eficiente como baseline forte."),
    h2("4.2 Naive Bayes Gaussiano"),
    p("Likelihood gaussiana por feature, predicao via log-posteriors com softmax. "
      "<b>Justificativa:</b> baseline probabilistico classico para comparacao de desempenho."),
    h2("4.3 Gradient Boosting com Decision Stumps"),
    p("Algoritmo de Friedman (2001) com log-loss. Cada iteracao treina um stump sobre os residuos. "
      "Subsampling 50%, ate 12 features por stump, n=100, lr=0,15. "
      "<b>Justificativa:</b> estado da arte em dados tabulares, eficiente com stumps."),
    h2("4.4 Validacao"),
    p("Stratified 3-Fold Cross-Validation no treino + avaliacao final em holdout 20%. "
      "Metricas: Accuracy, F1-Score (macro), Precision, Recall, ROC-AUC."),
]

res = make_table([
    ["Modelo",                "Accuracy","F1 (macro)","Precision","Recall","ROC-AUC"],
    ["Regressao Logistica",   "0,7471",  "0,6290",   "0,6233",   "0,6347","0,6818"],
    ["Naive Bayes Gaussiano", "0,2240",  "0,2018",   "0,5147",   "0,3862","0,6469"],
    ["Gradient Boosting",     "0,8292",  "0,6121",   "0,8167",   "0,5310","0,6633"],
], [4.5*cm, 2.2*cm, 2.4*cm, 2.4*cm, 2.1*cm, 2.4*cm])
res.setStyle(TableStyle([("BACKGROUND",(0,3),(-1,3),colors.HexColor("#d5f5e3"))]))

story.append(sp(18))
story += [
    h1("5. Resultados e Discussao"), hr(),
    res, sp(3),
    cap("Tabela 2 - Metricas no conjunto de teste (holdout 20%, 17.362 amostras). Linha verde = melhor acuracia."),
    sp(5),
    p("O <b>Gradient Boosting</b> teve maior acuracia (82,9%) e precisao (81,7%) - quando prediz "
      "insatisfacao, acerta com alta confiabilidade, mas recall de 53% indica que perde parte dos casos. "
      "A <b>Regressao Logistica</b> teve melhor F1 macro (0,629) e ROC-AUC (0,6818), "
      "sendo mais equilibrada - preferivel quando o custo de falso negativo e alto. "
      "O <b>Naive Bayes</b> teve baixa acuracia por violar a suposicao de independencia "
      "(is_late e delivery_delay_days sao altamente correlacionadas), mas ROC-AUC de 0,647."),
    h2("Feature Importances - Gradient Boosting"),
    make_table([
        ["Feature",                 "Importancia"],
        ["is_late",                 "33,5%"],
        ["delivery_delay_days",     "27,0%"],
        ["shipping_days",           "11,9%"],
        ["freight_ratio",           "6,8%"],
        ["estimated_delivery_days", "4,3%"],
        ["price_total",             "3,1%"],
    ], [8*cm, 4*cm], header_color=C2),
    cap("Tabela 3 - Importancia de features no Gradient Boosting. Features de entrega = 72% do total."),
]

story.append(sp(18))
story += [
    h1("6. Conclusao"), hr(),
    p("O projeto demonstrou que e possivel prever a satisfacao de clientes de e-commerce "
      "com boa acuracia usando dados logisticos e de pagamento. O Gradient Boosting obteve "
      "acuracia de 82,9% e a Regressao Logistica foi mais robusta ao desbalanceamento "
      "(melhor F1 e ROC-AUC). O atraso na entrega e o principal determinante de insatisfacao, "
      "confirmando as hipoteses iniciais."),
    h2("Limitacoes"),
    p("Dataset cobre apenas 2016-2018. Desbalanceamento 79/21% exige atencao na interpretacao "
      "da acuracia. Comentarios textuais dos reviews nao foram utilizados. "
      "Sem hiperparametrizacao sistematica."),
    h2("Trabalhos Futuros"),
    p("Aplicar SMOTE para o desbalanceamento. Analise de sentimento nos comentarios via TF-IDF/BERT. "
      "Testar XGBoost, LightGBM e redes neurais. Monitoramento de data drift em producao."),
    sp(18),
    h1("7. Referencias"), hr(),
    ref("[1] FRIEDMAN, J. H. Greedy Function Approximation: A Gradient Boosting Machine. "
        "Annals of Statistics, v. 29, n. 5, p. 1189-1232, 2001."),
    sp(2),
    ref("[2] OLIST. Brazilian E-Commerce Public Dataset. Kaggle, 2018. "
        "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce"),
    sp(2),
    ref("[3] HASTIE, T.; TIBSHIRANI, R.; FRIEDMAN, J. The Elements of Statistical Learning. "
        "2. ed. Springer, 2009."),
    sp(2),
    ref("[4] MITCHELL, T. M. Machine Learning. McGraw-Hill, 1997."),
]

doc.build(story)
print(f"PDF gerado: {OUTPUT}")
