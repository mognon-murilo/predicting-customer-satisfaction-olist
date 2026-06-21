import pandas as pd 
import numpy as np 
from sklearn .preprocessing import StandardScaler 
import joblib 

from utils import DATA_PROCESSED_DIR ,MODELS_DIR ,get_logger ,ensure_dirs 

logger =get_logger (__name__ )

VALID_STATUS =["delivered"]

# colunas identificadoras que não entram no modelo
COLS_TO_DROP =[
"order_id","customer_id","seller_id","product_id",
"customer_unique_id","customer_zip_code_prefix",
"seller_zip_code_prefix",
"order_purchase_timestamp","order_approved_at",
"order_delivered_carrier_date","review_creation_date",
"review_answer_timestamp",
]

NUMERIC_FEATURES =[
"price_total","freight_total","payment_value",
"payment_installments","product_weight_g","product_photos_qty",
"n_items","delivery_delay_days","shipping_days",
"estimated_delivery_days","freight_ratio",
]

CATEGORICAL_FEATURES =[
"payment_type","customer_state","seller_state",
"product_category_name_english",
]

def create_time_features (df :pd .DataFrame )->pd .DataFrame :
    df =df .copy ()
    df ["delivery_delay_days"]=(
    df ["order_delivered_customer_date"]-df ["order_estimated_delivery_date"]
    ).dt .days 
    df ["shipping_days"]=(
    df ["order_delivered_carrier_date"]-df ["order_approved_at"]
    ).dt .days 
    df ["estimated_delivery_days"]=(
    df ["order_estimated_delivery_date"]-df ["order_purchase_timestamp"]
    ).dt .days 
    df ["is_late"]=(df ["delivery_delay_days"]>0 ).astype (int )
    logger .info ("Features temporais criadas: delivery_delay_days, shipping_days, estimated_delivery_days, is_late.")
    return df 

def create_ratio_features (df :pd .DataFrame )->pd .DataFrame :
    df =df .copy ()
    # proporção do frete em relação ao valor do pedido
    df ["freight_ratio"]=df ["freight_total"]/df ["price_total"].replace (0 ,np .nan )
    logger .info ("Feature criada: freight_ratio.")
    return df 

def filter_valid_orders (df :pd .DataFrame )->pd .DataFrame :
    # só pedidos entregues têm review confiável
    n_before =len (df )
    df =df [df ["order_status"].isin (VALID_STATUS )].copy ()
    logger .info ("Filtragem de status: %d → %d linhas (removidos: %d).",
    n_before ,len (df ),n_before -len (df ))
    return df 

def drop_missing_target (df :pd .DataFrame )->pd .DataFrame :
    n_before =len (df )
    df =df .dropna (subset =["review_score"])
    logger .info ("Remoção de target ausente: %d → %d linhas.",n_before ,len (df ))
    return df 

def handle_missing_values (df :pd .DataFrame )->pd .DataFrame :
    df =df .copy ()
    # numéricas: mediana; categóricas: 'unknown'
    num_cols =df .select_dtypes (include ="number").columns .tolist ()
    for col in num_cols :
        n_missing =df [col ].isna ().sum ()
        if n_missing >0 :
            median_val =df [col ].median ()
            df [col ]=df [col ].fillna (median_val )
            logger .info ("  %s: %d nulos → preenchidos com mediana (%.2f).",col ,n_missing ,median_val )

    cat_cols =df .select_dtypes (include ="object").columns .tolist ()
    for col in cat_cols :
        n_missing =df [col ].isna ().sum ()
        if n_missing >0 :
            df [col ]=df [col ].fillna ("unknown")
            logger .info ("  %s: %d nulos → preenchidos com 'unknown'.",col ,n_missing )

    logger .info ("Tratamento de nulos concluído.")
    return df 

def remove_outliers_iqr (df :pd .DataFrame ,columns :list ,multiplier :float =3.0 )->pd .DataFrame :
    # multiplicador 3.0 para ser conservador e não perder muitos dados
    df =df .copy ()
    n_before =len (df )
    for col in columns :
        if col not in df .columns :
            continue 
        Q1 =df [col ].quantile (0.25 )
        Q3 =df [col ].quantile (0.75 )
        IQR =Q3 -Q1 
        lower =Q1 -multiplier *IQR 
        upper =Q3 +multiplier *IQR 
        mask =(df [col ]>=lower )&(df [col ]<=upper )
        df =df [mask ]
    logger .info ("Remoção de outliers IQR (×%.1f): %d → %d linhas.",multiplier ,n_before ,len (df ))
    return df 

def create_target (df :pd .DataFrame )->pd .DataFrame :
    # score >= 4 = satisfeito (1), <= 3 = insatisfeito (0)
    df =df .copy ()
    df ["target"]=(df ["review_score"]>=4 ).astype (int )
    dist =df ["target"].value_counts (normalize =True ).round (3 )
    logger .info ("Variável-alvo criada. Distribuição:\n%s",dist .to_string ())
    return df 

def encode_categoricals (df :pd .DataFrame )->pd .DataFrame :
    valid_cats =[c for c in CATEGORICAL_FEATURES if c in df .columns ]
    df =pd .get_dummies (df ,columns =valid_cats ,drop_first =True ,dtype =int )
    logger .info ("One-hot encoding aplicado em: %s.",valid_cats )
    return df 

def scale_numerics (df :pd .DataFrame ,fit :bool =True ,scaler_path :str =None )->pd .DataFrame :
    df =df .copy ()
    valid_nums =[c for c in NUMERIC_FEATURES if c in df .columns ]

    if scaler_path is None :
        scaler_path =str (MODELS_DIR /"scaler.pkl")

    if fit :
        scaler =StandardScaler ()
        df [valid_nums ]=scaler .fit_transform (df [valid_nums ])
        joblib .dump (scaler ,scaler_path )
        logger .info ("Scaler ajustado e salvo em: %s.",scaler_path )
    else :
        scaler =joblib .load (scaler_path )
        df [valid_nums ]=scaler .transform (df [valid_nums ])
        logger .info ("Scaler carregado de: %s.",scaler_path )

    return df 

def run_cleaning_pipeline (df :pd .DataFrame )->pd .DataFrame :
    logger .info ("=== Iniciando pipeline de limpeza ===")

    df =filter_valid_orders (df )
    df =drop_missing_target (df )
    df =create_time_features (df )
    df =create_ratio_features (df )
    df =handle_missing_values (df )
    df =remove_outliers_iqr (df ,["price_total","freight_total",
    "delivery_delay_days","payment_value"])
    df =create_target (df )

    cols_to_drop =[c for c in COLS_TO_DROP if c in df .columns ]
    cols_to_drop +=["order_status","review_score",
    "order_estimated_delivery_date",
    "order_delivered_customer_date"]
    df =df .drop (columns =[c for c in cols_to_drop if c in df .columns ],errors ="ignore")

    df =encode_categoricals (df )
    df =scale_numerics (df ,fit =True )

    logger .info ("=== Pipeline concluído: %d linhas × %d colunas ===",*df .shape )
    return df 

if __name__ =="__main__":
    ensure_dirs ()

    input_path =DATA_PROCESSED_DIR /"olist_analytical.parquet"
    if not input_path .exists ():
        raise FileNotFoundError (
        f"Arquivo não encontrado: {input_path}\n"
        "Execute primeiro: python src/data_ingestion.py"
        )

    logger .info ("Carregando tabela analítica de %s...",input_path )
    df_raw =pd .read_parquet (input_path )

    df_clean =run_cleaning_pipeline (df_raw )

    output_path =DATA_PROCESSED_DIR /"olist_clean.parquet"
    df_clean .to_parquet (output_path ,index =False )
    logger .info ("Dataset limpo salvo em: %s",output_path )
    logger .info ("Shape final: %d linhas × %d colunas.",*df_clean .shape )
