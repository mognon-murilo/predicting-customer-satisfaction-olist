import pandas as pd 
import numpy as np 
from sklearn .model_selection import train_test_split 

from utils import DATA_PROCESSED_DIR ,get_logger ,ensure_dirs 

logger =get_logger (__name__ )

TEST_SIZE =0.2 
RANDOM_STATE =42 

def add_seller_features (df_raw :pd .DataFrame ,df_clean :pd .DataFrame )->pd .DataFrame :
    # calcula métricas históricas por vendedor usando o dataset bruto
    seller_stats =(
    df_raw 
    .dropna (subset =["seller_id","review_score"])
    .groupby ("seller_id")
    .agg (
    seller_late_rate =("is_late","mean")if "is_late"in df_raw .columns else ("review_score","count"),
    seller_avg_score =("review_score","mean"),
    seller_n_orders =("order_id","count"),
    )
    .reset_index ()
    )
    logger .info ("Métricas de vendedor calculadas para %d vendedores.",len (seller_stats ))
    return seller_stats 

def split_data (df :pd .DataFrame ,target_col :str ="target"):
    # split estratificado para manter proporção do target em treino e teste
    X =df .drop (columns =[target_col ])
    y =df [target_col ]

    X_train ,X_test ,y_train ,y_test =train_test_split (
    X ,y ,
    test_size =TEST_SIZE ,
    random_state =RANDOM_STATE ,
    stratify =y ,
    )

    logger .info (
    "Split: treino=%d (%.0f%%) | teste=%d (%.0f%%)",
    len (X_train ),100 *(1 -TEST_SIZE ),
    len (X_test ),100 *TEST_SIZE ,
    )
    logger .info (
    "Distribuição target — treino: %.2f%% positivos | teste: %.2f%% positivos",
    100 *y_train .mean (),
    100 *y_test .mean (),
    )

    return X_train ,X_test ,y_train ,y_test 

if __name__ =="__main__":
    ensure_dirs ()

    clean_path =DATA_PROCESSED_DIR /"olist_clean.csv"
    if not clean_path .exists ():
        raise FileNotFoundError (
        f"Arquivo não encontrado: {clean_path}\n"
        "Execute primeiro: python src/data_cleaning.py"
        )

    logger .info ("Carregando dataset limpo de %s...",clean_path )
    df =pd .read_csv (clean_path )

    nan_count =df .isna ().sum ().sum ()
    if nan_count >0 :
        logger .warning ("Ainda há %d valores ausentes. Preenchendo com 0.",nan_count )
        df =df .fillna (0 )

    X_train ,X_test ,y_train ,y_test =split_data (df )

    train_df =X_train .copy ()
    train_df ["target"]=y_train .values 

    test_df =X_test .copy ()
    test_df ["target"]=y_test .values 

    train_df .to_csv (DATA_PROCESSED_DIR /"train.csv",index =False )
    test_df .to_csv (DATA_PROCESSED_DIR /"test.csv",index =False )

    logger .info ("Conjuntos salvos:")
    logger .info ("  → data/processed/train.csv (%d linhas)",len (train_df ))
    logger .info ("  → data/processed/test.csv  (%d linhas)",len (test_df ))
