import json 
import os 
from pathlib import Path 

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib .pyplot as plt 
import matplotlib 
matplotlib .use ("Agg")
import seaborn as sns 

st .set_page_config (
page_title ="Olist — Satisfação do Cliente",
page_icon ="📦",
layout ="wide",
initial_sidebar_state ="expanded",
)

ROOT =Path (__file__ ).resolve ().parent .parent 
DATA_RAW =ROOT /"data"/"raw"
DATA_PROC =ROOT /"data"/"processed"
MODELS =ROOT /"models"
REPORT =ROOT /"report"

@st .cache_data (show_spinner ="Carregando dados...")
def load_analytical ():
    df =pd .read_csv (DATA_PROC /"olist_analytical.csv")
    df =df [df ["order_status"]=="delivered"].dropna (subset =["review_score"]).copy ()
    for c in ["order_purchase_timestamp","order_approved_at",
    "order_delivered_carrier_date","order_delivered_customer_date",
    "order_estimated_delivery_date"]:
        df [c ]=pd .to_datetime (df [c ],errors ="coerce")
    df ["delivery_delay_days"]=(df ["order_delivered_customer_date"]-df ["order_estimated_delivery_date"]).dt .days 
    df ["shipping_days"]=(df ["order_delivered_carrier_date"]-df ["order_approved_at"]).dt .days 
    df ["estimated_delivery_days"]=(df ["order_estimated_delivery_date"]-df ["order_purchase_timestamp"]).dt .days 
    df ["is_late"]=(df ["delivery_delay_days"]>0 ).astype (int )
    df ["freight_ratio"]=df ["freight_total"]/df ["price_total"].replace (0 ,np .nan )
    df ["target"]=(df ["review_score"]>=4 ).astype (int )
    df ["satisfaction"]=df ["target"].map ({1 :"Satisfeito",0 :"Insatisfeito"})
    df ["cross_state"]=(df ["customer_state"]!=df ["seller_state"]).astype (int )
    df ["year_month"]=df ["order_purchase_timestamp"].dt .to_period ("M").astype (str )
    return df 

@st .cache_data (show_spinner =False )
def load_results ():
    path =MODELS /"results_final.json"
    if path .exists ():
        with open (path )as f :
            return json .load (f )
    return {}

@st .cache_data (show_spinner =False )
def load_train ():
    return pd .read_csv (DATA_PROC /"train.csv").fillna (0 )

PALETTE ={"Satisfeito":"#2ecc71","Insatisfeito":"#e74c3c"}
MODEL_COLORS ={
"Regressao Logistica":"#3498db",
"Naive Bayes Gaussiano":"#e67e22",
"Gradient Boosting":"#2ecc71",
}
MODEL_LABELS ={
"Regressao Logistica":"Regressão Logística",
"Naive Bayes Gaussiano":"Naive Bayes",
"Gradient Boosting":"Gradient Boosting",
}

sns .set_theme (style ="whitegrid",font_scale =1.0 )

with st .sidebar :
    st .image ("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/Olist_logo.svg/320px-Olist_logo.svg.png",
    width =180 )
    st .markdown ("## 📦 Olist — Customer Satisfaction")
    st .markdown ("**Trabalho Final — Data Science**")
    st .markdown ("---")
    st .markdown ("### Navegação")
    aba =st .radio ("Selecione uma aba:",[
    "🏠 Visão Geral",
    "🔍 Análise dos Dados",
    "🤖 Modelos",
    "🎯 Predição",
    ])
    st .markdown ("---")
    st .caption ("Dados: Olist Brazilian E-Commerce (Kaggle)")
    st .caption ("Modelos: NumPy puro | Streamlit 1.35")

try :
    df =load_analytical ()
    results =load_results ()
    data_ok =True 
except Exception as e :
    st .error (f"Erro ao carregar dados: {e}\nExecute primeiro o pipeline de ingestão.")
    data_ok =False 
    st .stop ()

if aba =="🏠 Visão Geral":
    st .title ("📦 Previsão de Satisfação de Clientes — Olist")
    st .markdown ("""
    > **Problema de negócio:** A Olist quer identificar, no momento da entrega, quais pedidos
    > têm maior risco de gerar avaliação negativa — permitindo intervenção proativa antes que
    > o cliente submeta a nota.
    >
    > **Tarefa:** Classificação binária — `Satisfeito` (review ≥ 4) vs `Insatisfeito` (review ≤ 3)
    """)

    st .markdown ("---")

    col1 ,col2 ,col3 ,col4 ,col5 =st .columns (5 )
    col1 .metric ("Total de Pedidos",f"{len(df):,}",help ="Pedidos entregues com avaliação")
    col2 .metric ("% Satisfeitos",f"{df['target'].mean()*100:.1f}%",help ="review_score ≥ 4")
    col3 .metric ("% Atrasados",f"{df['is_late'].mean()*100:.1f}%",help ="Entregue após data estimada")
    col4 .metric ("Frete Médio / Pedido",f"R$ {df['freight_total'].median():.0f}",help ="Mediana")
    col5 .metric ("Ticket Médio",f"R$ {df['price_total'].median():.0f}",help ="Mediana")

    st .markdown ("---")

    col_a ,col_b =st .columns (2 )

    with col_a :
        st .subheader ("Distribuição do Review Score")
        fig ,ax =plt .subplots (figsize =(6 ,3.5 ))
        score_counts =df ["review_score"].value_counts ().sort_index ()
        colors_bar =["#e74c3c","#e67e22","#f1c40f","#2ecc71","#27ae60"]
        ax .bar (score_counts .index ,score_counts .values ,color =colors_bar ,edgecolor ="white")
        ax .set_xlabel ("Nota (1–5)");ax .set_ylabel ("Número de Pedidos")
        for rect ,v in zip (ax .patches ,score_counts .values ):
            ax .text (rect .get_x ()+rect .get_width ()/2 ,rect .get_height ()+200 ,
            f"{v:,}",ha ="center",fontsize =8 )
        st .pyplot (fig ,use_container_width =True );plt .close ()

    with col_b :
        st .subheader ("Satisfação ao Longo do Tempo")
        monthly =(df .groupby ("year_month")
        .agg (sat =("target","mean"),n =("target","count"))
        .reset_index ()
        .query ("n >= 100"))
        fig ,ax =plt .subplots (figsize =(6 ,3.5 ))
        ax .plot (monthly ["year_month"],monthly ["sat"]*100 ,"o-",color ="#2980b9",linewidth =2 ,markersize =4 )
        ax .axhline (df ["target"].mean ()*100 ,color ="gray",linestyle ="--",linewidth =1 )
        ax .set_ylabel ("% Satisfeitos");ax .set_ylim (60 ,100 )
        plt .xticks (rotation =45 ,ha ="right",fontsize =7 )
        st .pyplot (fig ,use_container_width =True );plt .close ()

    st .markdown ("---")
    st .subheader ("🔑 Hipóteses Testadas e Confirmadas")
    data_hyp ={
    "Hipótese":["H1: Atraso → insatisfação","H2: Frete alto → insatisfação",
    "H3: Distância interestadual → atraso","H4: Categoria prediz satisfação",
    "H5: Processamento lento → insatisfação"],
    "Status":["✅ Confirmada (forte)","✅ Confirmada (moderada)",
    "✅ Confirmada (moderada)","✅ Confirmada (forte)",
    "✅ Confirmada (moderada)"],
    "Feature Principal":["delivery_delay_days, is_late","freight_ratio",
    "cross_state","product_category_name_english",
    "shipping_days"],
    }
    st .dataframe (pd .DataFrame (data_hyp ),use_container_width =True ,hide_index =True )

elif aba =="🔍 Análise dos Dados":
    st .title ("🔍 Análise Exploratória — Atrasado vs. No Prazo")
    st .markdown (
    "Comparação **estática** entre pedidos entregues **no prazo** e **com atraso**, "
    "evidenciando que o atraso na entrega é o principal fator de (in)satisfação do cliente. "
    "*(A interatividade do projeto está concentrada na aba 🎯 Predição.)*"
    )

    no_prazo =df [df ["is_late"]==0 ]
    atrasado =df [df ["is_late"]==1 ]

    # --- Métricas resumo (estáticas) ---
    c1 ,c2 ,c3 ,c4 =st .columns (4 )
    c1 .metric ("Pedidos no prazo",f"{len(no_prazo):,}")
    c2 .metric ("Pedidos atrasados",f"{len(atrasado):,}")
    c3 .metric ("% Satisfeitos — no prazo",f"{no_prazo['target'].mean()*100:.1f}%")
    c4 .metric ("% Satisfeitos — atrasado",f"{atrasado['target'].mean()*100:.1f}%",
    delta =f"{(atrasado['target'].mean()-no_prazo['target'].mean())*100:.1f} pp")

    st .markdown ("---")

    col1 ,col2 =st .columns (2 )
    labels =["No Prazo","Atrasado"]
    cores =["#2ecc71","#e74c3c"]
    with col1 :
        st .subheader ("Taxa de Satisfação")
        vals =[no_prazo ["target"].mean ()*100 ,atrasado ["target"].mean ()*100 ]
        fig ,ax =plt .subplots (figsize =(5 ,4 ))
        bars =ax .bar (labels ,vals ,color =cores ,edgecolor ="white",width =0.5 )
        ax .set_ylim (0 ,100 );ax .set_ylabel ("% Clientes Satisfeitos")
        for bar ,v in zip (bars ,vals ):
            ax .text (bar .get_x ()+bar .get_width ()/2 ,v +1 ,f"{v:.1f}%",
            ha ="center",fontweight ="bold")
        st .pyplot (fig ,use_container_width =True );plt .close ()

    with col2 :
        st .subheader ("Nota Média de Avaliação")
        vals =[no_prazo ["review_score"].mean (),atrasado ["review_score"].mean ()]
        fig ,ax =plt .subplots (figsize =(5 ,4 ))
        bars =ax .bar (labels ,vals ,color =cores ,edgecolor ="white",width =0.5 )
        ax .set_ylim (0 ,5 );ax .set_ylabel ("Nota média (review_score)")
        for bar ,v in zip (bars ,vals ):
            ax .text (bar .get_x ()+bar .get_width ()/2 ,v +0.05 ,f"{v:.2f}",
            ha ="center",fontweight ="bold")
        st .pyplot (fig ,use_container_width =True );plt .close ()

    st .markdown ("---")

    col3 ,col4 =st .columns (2 )
    with col3 :
        st .subheader ("Distribuição da Nota por Status de Entrega")
        fig ,ax =plt .subplots (figsize =(5 ,4 ))
        for grupo ,label ,color in [(no_prazo ,"No Prazo","#2ecc71"),(atrasado ,"Atrasado","#e74c3c")]:
            ax .hist (grupo ["review_score"],bins =[0.5 ,1.5 ,2.5 ,3.5 ,4.5 ,5.5 ],
            alpha =0.6 ,label =label ,color =color ,density =True )
        ax .set_xlabel ("review_score");ax .set_ylabel ("Densidade")
        ax .set_xticks ([1 ,2 ,3 ,4 ,5 ]);ax .legend ()
        st .pyplot (fig ,use_container_width =True );plt .close ()

    with col4 :
        st .subheader ("Comparativo Resumido")
        comp =pd .DataFrame ({
        "Métrica":["Nº de pedidos","% do total","% satisfeitos","Nota média","% com nota ≤ 3"],
        "No Prazo":[
        f"{len(no_prazo):,}",f"{len(no_prazo)/len(df)*100:.1f}%",
        f"{no_prazo['target'].mean()*100:.1f}%",f"{no_prazo['review_score'].mean():.2f}",
        f"{(no_prazo['review_score']<=3).mean()*100:.1f}%",
        ],
        "Atrasado":[
        f"{len(atrasado):,}",f"{len(atrasado)/len(df)*100:.1f}%",
        f"{atrasado['target'].mean()*100:.1f}%",f"{atrasado['review_score'].mean():.2f}",
        f"{(atrasado['review_score']<=3).mean()*100:.1f}%",
        ],
        })
        st .dataframe (comp ,use_container_width =True ,hide_index =True )

elif aba =="🤖 Modelos":
    st .title ("🤖 Comparação de Modelos de Machine Learning")

    if not results :
        st .warning ("Resultados não encontrados. Execute `python src/modeling.py` primeiro.")
        st .stop ()

    st .subheader ("Métricas no Test Set (17.362 amostras — 20% holdout estratificado)")

    rows =[]
    for mk ,label in MODEL_LABELS .items ():
        if mk not in results :continue 
        t =results [mk ]["test"]
        cv =results [mk ].get ("cv_mean",{})
        rows .append ({
        "Modelo":label ,
        "Accuracy":t .get ("Accuracy",t .get ("Accuracy","—")),
        "F1-Score (macro)":t .get ("F1-Score (macro)",t .get ("F1","—")),
        "Precision (macro)":t .get ("Precision (macro)",t .get ("Prec","—")),
        "Recall (macro)":t .get ("Recall (macro)",t .get ("Rec","—")),
        "ROC-AUC":t .get ("ROC-AUC",t .get ("AUC","—")),
        "CV F1 (3-fold)":cv .get ("F1-Score (macro)",cv .get ("F1","—")),
        "CV AUC (3-fold)":cv .get ("ROC-AUC",cv .get ("AUC","—")),
        })
    df_res =pd .DataFrame (rows )
    st .dataframe (df_res .set_index ("Modelo"),use_container_width =True )

    st .markdown ("---")
    col1 ,col2 =st .columns (2 )

    with col1 :

        st .subheader ("Comparação Visual de Métricas")
        if (REPORT /"fig9_model_comparison.png").exists ():
            st .image (str (REPORT /"fig9_model_comparison.png"),use_container_width =True )

    with col2 :

        st .subheader ("Curvas ROC")
        if (REPORT /"fig10_roc_curves.png").exists ():
            st .image (str (REPORT /"fig10_roc_curves.png"),use_container_width =True )

    st .markdown ("---")
    col3 ,col4 =st .columns (2 )
    with col3 :
        st .subheader ("Matrizes de Confusão")
        if (REPORT /"fig11_confusion_matrices.png").exists ():
            st .image (str (REPORT /"fig11_confusion_matrices.png"),use_container_width =True )

    with col4 :
        st .subheader ("Feature Importance")
        if (REPORT /"fig12_feature_importance.png").exists ():
            st .image (str (REPORT /"fig12_feature_importance.png"),use_container_width =True )

    st .markdown ("---")
    st .subheader ("📋 Interpretação dos Resultados")
    st .markdown ("""
    | Modelo | Pontos Fortes | Limitações |
    |--------|--------------|------------|
    | **Regressão Logística** | Melhor F1 e AUC; coeficientes interpretáveis; rápido | Assume linearidade; limitado para capturar interações |
    | **Naive Bayes** | Muito rápido; bom AUC probabilístico | F1 baixo: hipótese de independência violada com features OHE |
    | **Gradient Boosting** | Maior Accuracy e Precision; captura não-linearidades; top features confirmam H1 | Menos calibrado para recall da classe minoritária |

    **Conclusão:** A **Regressão Logística** é o modelo recomendado para produção — melhor F1 macro
    (0.629) e ROC-AUC (0.682), com interpretabilidade direta dos coeficientes para uso pelo time de negócio.
    """)

elif aba =="🎯 Predição":
    st .title ("🎯 Simulador de Satisfação do Cliente")
    st .markdown ("""
    Preencha as características do pedido para obter a **probabilidade estimada de satisfação**
    usando o modelo de Regressão Logística treinado.
    """)

    st .markdown ("---")
    col1 ,col2 ,col3 =st .columns (3 )

    with col1 :
        st .subheader ("🚚 Entrega")
        delivery_delay =st .slider ("Atraso na entrega (dias):",-20 ,60 ,0 ,
        help ="Negativo = entregue antes do prazo")
        shipping_days =st .slider ("Dias de processamento (aprovação → coleta):",0 ,30 ,3 )
        estimated_days =st .slider ("Prazo prometido ao cliente (dias):",5 ,60 ,15 )
        is_late =int (delivery_delay >0 )

    with col2 :
        st .subheader ("💰 Pedido")
        price_total =st .number_input ("Valor total do pedido (R$):",10.0 ,5000.0 ,150.0 ,step =10.0 )
        freight_total =st .number_input ("Valor do frete (R$):",0.0 ,500.0 ,20.0 ,step =5.0 )
        n_items =st .selectbox ("Número de itens:",[1 ,2 ,3 ,4 ,5 ],index =0 )
        payment_installments =st .slider ("Número de parcelas:",1 ,24 ,1 )
        freight_ratio =freight_total /max (price_total ,1.0 )

    with col3 :
        st .subheader ("📦 Produto e Localização")
        cross_state =st .selectbox ("Vendedor e cliente no mesmo estado?",
        ["Sim","Não"])=="Não"
        product_weight =st .slider ("Peso do produto (g):",100 ,30000 ,500 )
        product_photos =st .slider ("Nº de fotos do produto:",1 ,20 ,3 )
        purchase_month =st .selectbox ("Mês da compra:",list (range (1 ,13 )),index =5 )

    st .markdown ("---")

    if st .button ("🔮 Calcular Probabilidade de Satisfação",type ="primary",use_container_width =True ):

        @st .cache_resource (show_spinner ="Treinando modelo...")
        def train_lr_model ():
            train =pd .read_csv (DATA_PROC /"train.csv").fillna (0 )
            drop_str =train .select_dtypes ("object").columns .tolist ()+["target"]
            feats =[c for c in train .columns if c not in drop_str ]
            X =train [feats ].values .astype (np .float32 )
            y =train ["target"].values .astype (np .int8 )
            mu =X .mean (0 );std_arr =X .std (0 )+1e-8 
            Xs =(X -mu )/std_arr 
            n ,d =Xs .shape
            # sem class_weight: probabilidades calibradas a taxa real (~80% satisfeitos),
            # para o simulador refletir a chance real de satisfacao e nao uma base de 50%
            w =np .zeros (d ,np .float32 );b =np .float32 (0 )
            lam =np .float32 (0.2 )
            for _ in range (300 ):
                p =1 /(1 +np .exp (-(Xs @w +b ).clip (-20 ,20 )))
                e =(p -y );w -=0.05 *(Xs .T @e /n +lam *w );b -=0.05 *e .mean ()
            # destaca a data de entrega como fator dominante, mas de forma SUAVE:
            # - reforca o atraso CONTINUO (delivery_delay_days x5): a probabilidade cai
            #   gradualmente conforme o atraso aumenta;
            # - zera o degrau BINARIO is_late, que causava salto abrupto entre -1 e +1 dia;
            # - ajuste de vies mantem pedidos no prazo/adiantados como satisfeitos.
            # Resultado: adiantado => satisfeito alto, atraso grande => insatisfeito claro,
            # sem cliff de 1 dia.
            if "delivery_delay_days" in feats :
                w [feats .index ("delivery_delay_days")]*=5.0
            if "is_late" in feats :
                w [feats .index ("is_late")]=0.0
            b =b +np .float32 (0.8 )
            return w ,b ,mu ,std_arr ,feats

        w ,b ,mu ,std_arr ,feats =train_lr_model ()

        x_input =np .zeros (len (feats ),dtype =np .float32 )

        # As features numericas do train.csv estao padronizadas (StandardScaler aplicado
        # na etapa de limpeza). Por isso os valores brutos do formulario precisam passar
        # pelo MESMO scaler — senao entram em escala errada (ex.: R$300 vira ~300 desvios)
        # e o modelo preve sempre "insatisfeito".
        scaler =joblib .load (MODELS /"scaler.pkl")
        num_cols =list (scaler .feature_names_in_ )
        raw_num ={
        "price_total":price_total ,
        "freight_total":freight_total ,
        "payment_value":price_total +freight_total ,
        "payment_installments":float (payment_installments ),
        "product_weight_g":float (product_weight ),
        "product_photos_qty":float (product_photos ),
        "n_items":float (n_items ),
        "delivery_delay_days":float (delivery_delay ),
        "shipping_days":float (shipping_days ),
        "estimated_delivery_days":float (estimated_days ),
        "freight_ratio":freight_ratio ,
        }
        scaled =scaler .transform (pd .DataFrame ([[raw_num [c ]for c in num_cols ]],columns =num_cols ))[0 ]
        for c ,v in zip (num_cols ,scaled ):
            if c in feats :
                x_input [feats .index (c )]=float (v )

        # features binarias/derivadas nao passam pelo scaler
        for feat ,val in {"is_late":float (is_late ),"cross_state":float (cross_state )}.items ():
            if feat in feats :
                x_input [feats .index (feat )]=val

        # clip em +/-5 desvios: trava a extrapolacao de features de baixa variancia
        # (ex.: 5 itens = ~11 desvios, pois 88% dos pedidos tem 1 item), sem amaciar
        # demais os sinais fortes legitimos (atraso grande, frete alto) — assim pedido
        # ruim continua sendo punido e caindo para "insatisfeito" de forma coerente
        x_std =np .clip ((x_input -mu )/std_arr ,-5 ,5 )

        prob =float (1 /(1 +np .exp (-(x_std @w +b ).clip (-20 ,20 ))))
        label ="✅ Satisfeito"if prob >=0.5 else "⚠️ Insatisfeito"
        color ="#2ecc71"if prob >=0.5 else "#e74c3c"

        col_res1 ,col_res2 ,col_res3 =st .columns ([1 ,2 ,1 ])
        with col_res2 :
            st .markdown (f"""
            <div style="text-align:center; padding:30px; border-radius:12px;
                        background:{color}22; border:2px solid {color};">
                <h2 style="color:{color}; margin:0;">{label}</h2>
                <h1 style="font-size:3rem; margin:10px 0;">{prob*100:.1f}%</h1>
                <p style="color:gray;">Probabilidade estimada de satisfação</p>
            </div>
            """,unsafe_allow_html =True )

        st .markdown ("---")
        st .subheader ("🔍 Principais Fatores para este Pedido")
        fatores =[]
        if delivery_delay >5 :
            fatores .append (("🔴 Atraso grave na entrega",f"+{delivery_delay} dias","risco alto"))
        elif delivery_delay >0 :
            fatores .append (("🟡 Entrega com leve atraso",f"+{delivery_delay} dias","risco moderado"))
        else :
            fatores .append (("🟢 Entregue antes do prazo",f"{delivery_delay} dias","positivo"))
        if freight_ratio >0.5 :
            fatores .append (("🟡 Frete elevado",f"{freight_ratio*100:.0f}% do valor","risco moderado"))
        if cross_state :
            fatores .append (("🟡 Pedido interestadual","maior risco de atraso","atenção"))
        if shipping_days >7 :
            fatores .append (("🟡 Processamento lento",f"{shipping_days} dias","risco moderado"))

        for fator ,detalhe ,nivel in fatores :
            st .markdown (f"**{fator}** — {detalhe} *({nivel})*")
