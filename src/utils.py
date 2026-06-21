import logging 
import os 
from pathlib import Path 

# caminhos base do projeto
ROOT_DIR =Path (__file__ ).resolve ().parent .parent 
DATA_RAW_DIR =ROOT_DIR /"data"/"raw"
DATA_PROCESSED_DIR =ROOT_DIR /"data"/"processed"
MODELS_DIR =ROOT_DIR /"models"
REPORTS_DIR =ROOT_DIR /"report"

def get_logger (name :str )->logging .Logger :
    logging .basicConfig (
    level =logging .INFO ,
    format ="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt ="%Y-%m-%d %H:%M:%S",
    )
    return logging .getLogger (name )

def ensure_dirs ()->None :
    # cria as pastas necessárias se não existirem
    for directory in [DATA_RAW_DIR ,DATA_PROCESSED_DIR ,MODELS_DIR ]:
        directory .mkdir (parents =True ,exist_ok =True )

def check_raw_data ()->bool :
    expected_files =[
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_customers_dataset.csv",
    "olist_sellers_dataset.csv",
    "olist_products_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_geolocation_dataset.csv",
    "product_category_name_translation.csv",
    ]
    missing =[f for f in expected_files if not (DATA_RAW_DIR /f ).exists ()]
    if missing :
        logger =get_logger (__name__ )
        logger .warning (
        "Arquivos ausentes em data/raw/: %s\n"
        "Baixe o dataset em: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce",
        missing ,
        )
        return False 
    return True 
