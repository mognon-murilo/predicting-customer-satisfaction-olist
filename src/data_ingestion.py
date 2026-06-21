import pandas as pd 
from utils import DATA_RAW_DIR ,DATA_PROCESSED_DIR ,get_logger ,check_raw_data ,ensure_dirs 

logger =get_logger (__name__ )

def load_orders ()->pd .DataFrame :
    df =pd .read_csv (DATA_RAW_DIR /"olist_orders_dataset.csv",parse_dates =[
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    ])
    logger .info ("orders: %d linhas carregadas.",len (df ))
    return df 

def load_order_items ()->pd .DataFrame :
    df =pd .read_csv (DATA_RAW_DIR /"olist_order_items_dataset.csv")
    logger .info ("order_items: %d linhas carregadas.",len (df ))
    return df 

def load_reviews ()->pd .DataFrame :
    df =pd .read_csv (DATA_RAW_DIR /"olist_order_reviews_dataset.csv",parse_dates =[
    "review_creation_date",
    "review_answer_timestamp",
    ])
    logger .info ("reviews: %d linhas carregadas.",len (df ))
    return df 

def load_customers ()->pd .DataFrame :
    df =pd .read_csv (DATA_RAW_DIR /"olist_customers_dataset.csv")
    logger .info ("customers: %d linhas carregadas.",len (df ))
    return df 

def load_sellers ()->pd .DataFrame :
    df =pd .read_csv (DATA_RAW_DIR /"olist_sellers_dataset.csv")
    logger .info ("sellers: %d linhas carregadas.",len (df ))
    return df 

def load_products ()->pd .DataFrame :
    products =pd .read_csv (DATA_RAW_DIR /"olist_products_dataset.csv")
    translation =pd .read_csv (DATA_RAW_DIR /"product_category_name_translation.csv")
    df =products .merge (translation ,on ="product_category_name",how ="left")
    logger .info ("products: %d linhas carregadas.",len (df ))
    return df 

def load_payments ()->pd .DataFrame :
    df =pd .read_csv (DATA_RAW_DIR /"olist_order_payments_dataset.csv")
    # agrega por pedido: total pago, parcelas e tipo dominante
    agg =df .groupby ("order_id").agg (
    payment_value =("payment_value","sum"),
    payment_installments =("payment_installments","max"),
    payment_type =("payment_type","first"),
    ).reset_index ()
    logger .info ("payments agregados: %d pedidos.",len (agg ))
    return agg 

def build_analytical_table ()->pd .DataFrame :
    logger .info ("Iniciando construção da tabela analítica...")

    orders =load_orders ()
    items =load_order_items ()
    reviews =load_reviews ()
    customers =load_customers ()
    sellers =load_sellers ()
    products =load_products ()
    payments =load_payments ()

    # agrega itens por pedido
    items_agg =items .groupby ("order_id").agg (
    price_total =("price","sum"),
    freight_total =("freight_value","sum"),
    n_items =("order_item_id","count"),
    seller_id =("seller_id","first"),
    product_id =("product_id","first"),
    ).reset_index ()

    # mantém apenas o review mais recente por pedido
    reviews_dedup =(
    reviews .sort_values ("review_creation_date",ascending =False )
    .drop_duplicates (subset ="order_id",keep ="first")
    )

    df =(
    orders 
    .merge (items_agg ,on ="order_id",how ="left")
    .merge (reviews_dedup [["order_id","review_score","review_creation_date",
    "review_answer_timestamp"]],on ="order_id",how ="left")
    .merge (customers ,on ="customer_id",how ="left")
    .merge (sellers ,on ="seller_id",how ="left")
    .merge (products [["product_id","product_category_name_english",
    "product_weight_g","product_photos_qty"]],on ="product_id",how ="left")
    .merge (payments ,on ="order_id",how ="left")
    )

    logger .info ("Tabela analítica construída: %d linhas × %d colunas.",*df .shape )
    return df 

if __name__ =="__main__":
    ensure_dirs ()

    if not check_raw_data ():
        raise FileNotFoundError (
        "Dataset não encontrado. Baixe em: "
        "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce "
        "e extraia os CSVs em data/raw/"
        )

    df =build_analytical_table ()

    output_path =DATA_PROCESSED_DIR /"olist_analytical.parquet"
    df .to_parquet (output_path ,index =False )
    logger .info ("Tabela salva em: %s",output_path )
