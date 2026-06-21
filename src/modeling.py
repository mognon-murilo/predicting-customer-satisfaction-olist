import numpy as np 
import pandas as pd 
import json 
import time 

from utils import DATA_PROCESSED_DIR ,MODELS_DIR ,REPORTS_DIR ,get_logger ,ensure_dirs 

logger =get_logger (__name__ )
np .random .seed (42 )

# hiperparâmetros dos modelos
LR_PARAMS =dict (lr =0.05 ,epochs =200 ,C =0.5 )
GB_PARAMS =dict (n =100 ,lr =0.15 ,ss =0.5 ,mf =12 ,nt =5 )

def roc_auc (y :np .ndarray ,scores :np .ndarray )->float :
    # AUC via ordenação, O(n log n)
    order =np .argsort (scores )[::-1 ]
    ys =y [order ]
    P =int (y .sum ());N =len (y )-P 
    if P ==0 or N ==0 :
        return 0.5 
    tpr =np .cumsum (ys )/P 
    fpr =np .cumsum (1 -ys )/N 
    return float (np .trapezoid (np .r_ [0 ,tpr ,1 ],np .r_ [0 ,fpr ,1 ]))

def evaluate (y :np .ndarray ,yhat :np .ndarray ,scores :np .ndarray )->dict :
    y =y .astype (int )
    tp =((yhat ==1 )&(y ==1 )).sum ();fp =((yhat ==1 )&(y ==0 )).sum ()
    fn =((yhat ==0 )&(y ==1 )).sum ();tn =((yhat ==0 )&(y ==0 )).sum ()
    p1 =tp /(tp +fp +1e-12 );r1 =tp /(tp +fn +1e-12 )
    f1 =2 *p1 *r1 /(p1 +r1 +1e-12 )
    p0 =tn /(tn +fn +1e-12 );r0 =tn /(tn +fp +1e-12 )
    f0 =2 *p0 *r0 /(p0 +r0 +1e-12 )
    return {
    "Accuracy":round (float ((y ==yhat ).mean ()),4 ),
    "F1-Score (macro)":round (float ((f1 +f0 )/2 ),4 ),
    "Precision (macro)":round (float ((p1 +p0 )/2 ),4 ),
    "Recall (macro)":round (float ((r1 +r0 )/2 ),4 ),
    "ROC-AUC":round (roc_auc (y ,scores ),4 ),
    }

def stratified_kfold (y :np .ndarray ,k :int =3 ,seed :int =42 ):
    # divide os índices mantendo proporção das classes em cada fold
    rng =np .random .RandomState (seed )
    i0 =np .where (y ==0 )[0 ];rng .shuffle (i0 )
    i1 =np .where (y ==1 )[0 ];rng .shuffle (i1 )
    return [
    (np .setdiff1d (np .arange (len (y )),np .r_ [i0 [i ::k ],i1 [i ::k ]]),
    np .r_ [i0 [i ::k ],i1 [i ::k ]])
    for i in range (k )
    ]

class LogisticRegression :
    # gradient descent com regularização L2 e class_weight balanceado
    def __init__ (self ,lr :float =0.05 ,epochs :int =200 ,C :float =0.5 ):
        self .lr =lr ;self .epochs =epochs ;self .C =C 

    def fit (self ,X :np .ndarray ,y :np .ndarray )->"LogisticRegression":
        n ,d =X .shape 
        n0 ,n1 =(y ==0 ).sum (),(y ==1 ).sum ()
        sw =np .where (y ==1 ,n /(2 *n1 ),n /(2 *n0 )).astype (np .float32 )
        self .w =np .zeros (d ,np .float32 );self .b =np .float32 (0 )
        lam =np .float32 (1 /self .C )
        for _ in range (self .epochs ):
            p =1 /(1 +np .exp (-(X @self .w +self .b ).clip (-20 ,20 )))
            err =(p -y )*sw 
            self .w -=self .lr *(X .T @err /n +lam *self .w )
            self .b -=self .lr *err .mean ()
        return self 

    def predict_proba (self ,X :np .ndarray )->np .ndarray :
        return 1 /(1 +np .exp (-(X @self .w +self .b ).clip (-20 ,20 )))

    def predict (self ,X :np .ndarray ,threshold :float =0.5 )->np .ndarray :
        return (self .predict_proba (X )>=threshold ).astype (int )

class GaussianNaiveBayes :
    # likelihood gaussiana por feature, log-posteriors para estabilidade
    def fit (self ,X :np .ndarray ,y :np .ndarray )->"GaussianNaiveBayes":
        self .priors_ ={c :float ((y ==c ).mean ())for c in [0 ,1 ]}
        self .mu_ ={c :X [y ==c ].mean (axis =0 )for c in [0 ,1 ]}
        self .var_ ={c :X [y ==c ].var (axis =0 )+1e-9 for c in [0 ,1 ]}
        return self 

    def _log_likelihood (self ,X :np .ndarray ,c :int )->np .ndarray :
        mu ,var =self .mu_ [c ],self .var_ [c ]
        return -0.5 *(np .log (2 *np .pi *var )+(X -mu )**2 /var ).sum (axis =1 )

    def predict_proba (self ,X :np .ndarray )->np .ndarray :
        l0 =np .log (self .priors_ [0 ])+self ._log_likelihood (X ,0 )
        l1 =np .log (self .priors_ [1 ])+self ._log_likelihood (X ,1 )
        m =np .maximum (l0 ,l1 )
        e0 ,e1 =np .exp (l0 -m ),np .exp (l1 -m )
        return e1 /(e0 +e1 )

    def predict (self ,X :np .ndarray ,threshold :float =0.5 )->np .ndarray :
        return (self .predict_proba (X )>=threshold ).astype (int )

class GradientBoostingStumps :
    # boosting com decision stumps, implementação do algoritmo de Friedman
    def __init__ (self ,n :int =100 ,lr :float =0.15 ,
    ss :float =0.5 ,mf :int =12 ,nt :int =5 ):
        self .n =n ;self .lr =lr ;self .ss =ss 
        self .mf =mf ;self .nt =nt 
        self .stumps_ =[]
        self .F0_ =0.0 

    def fit (self ,X :np .ndarray ,y :np .ndarray )->"GradientBoostingStumps":
        n =len (y )
        p_mean =y .mean ().clip (0.01 ,0.99 )
        self .F0_ =float (np .log (p_mean /(1 -p_mean )))
        F =np .full (n ,self .F0_ ,dtype =np .float64 )

        for _ in range (self .n ):
            # subsample e cálculo dos resíduos
            idx =np .random .choice (n ,int (n *self .ss ),replace =False )
            p =1 /(1 +np .exp (-F [idx ].clip (-15 ,15 )))
            residual =(y [idx ]-p ).astype (np .float64 )

            # busca o melhor stump por subconjunto aleatório de features
            best_mse =np .inf ;bf =0 ;bt =0.0 ;bL =0.0 ;bR =0.0 
            fs =np .random .choice (X .shape [1 ],min (self .mf ,X .shape [1 ]),replace =False )
            for f in fs :
                col =X [idx ,f ];lo ,hi =col .min (),col .max ()
                if lo ==hi :
                    continue 
                for t in np .linspace (lo +(hi -lo )*0.1 ,hi -(hi -lo )*0.1 ,self .nt ):
                    lm =col <=t ;nL ,nR =lm .sum (),(~lm ).sum ()
                    if nL <8 or nR <8 :
                        continue 
                    vL =residual [lm ].mean ();vR =residual [~lm ].mean ()
                    mse =(((residual [lm ]-vL )**2 ).sum ()+((residual [~lm ]-vR )**2 ).sum ())/len (idx )
                    if mse <best_mse :
                        best_mse =mse ;bf =f ;bt =t ;bL =vL ;bR =vR 

            F +=self .lr *np .where (X [:,bf ]<=bt ,bL ,bR )
            self .stumps_ .append ((bf ,bt ,float (bL ),float (bR )))

        return self 

    def predict_proba (self ,X :np .ndarray )->np .ndarray :
        F =np .full (len (X ),self .F0_ ,dtype =np .float64 )
        for f ,t ,L ,R in self .stumps_ :
            F +=self .lr *np .where (X [:,f ]<=t ,L ,R )
        return 1 /(1 +np .exp (-F .clip (-15 ,15 )))

    def predict (self ,X :np .ndarray ,threshold :float =0.5 )->np .ndarray :
        return (self .predict_proba (X )>=threshold ).astype (int )

    def feature_importances (self ,n_features :int )->np .ndarray :
        # importância = ganho absoluto acumulado por feature
        imp =np .zeros (n_features )
        for f ,_ ,L ,R in self .stumps_ :
            imp [f ]+=abs (L -R )
        return imp /(imp .sum ()+1e-12 )

MODEL_FACTORY ={
"Regressão Logística":lambda :LogisticRegression (**LR_PARAMS ),
"Naive Bayes Gaussiano":lambda :GaussianNaiveBayes (),
"Gradient Boosting":lambda :GradientBoostingStumps (**GB_PARAMS ),
}

def _load_xy (path ):
    df =pd .read_csv (path ).fillna (0 )
    y =df ["target"].astype (int ).values
    # descarta target e colunas de texto (ex.: cidades), igual ao dashboard
    drop_cols =df .select_dtypes ("object").columns .tolist ()+["target"]
    feats =[c for c in df .columns if c not in drop_cols ]
    X =df [feats ].astype (np .float32 ).values
    return X ,y

def _aggregate (cv_metrics :list )->tuple :
    keys =cv_metrics [0 ].keys ()
    mean ={k :round (float (np .mean ([m [k ]for m in cv_metrics ])),4 )for k in keys }
    std ={k :round (float (np .std ([m [k ]for m in cv_metrics ])),4 )for k in keys }
    return mean ,std

def run_modeling_pipeline ():
    ensure_dirs ()

    train_path =DATA_PROCESSED_DIR /"train.csv"
    test_path =DATA_PROCESSED_DIR /"test.csv"
    for p in (train_path ,test_path ):
        if not p .exists ():
            raise FileNotFoundError (
            f"Arquivo não encontrado: {p}\n"
            "Execute primeiro: python src/feature_engineering.py"
            )

    X_train ,y_train =_load_xy (train_path )
    X_test ,y_test =_load_xy (test_path )
    logger .info ("Treino: %d×%d | Teste: %d×%d",*X_train .shape ,*X_test .shape )

    folds =stratified_kfold (y_train ,k =3 ,seed =42 )
    results ={}

    for name ,make in MODEL_FACTORY .items ():
        logger .info ("=== %s ===",name )
        t0 =time .time ()

        # validação cruzada estratificada
        cv_metrics =[]
        for i ,(tr_idx ,va_idx )in enumerate (folds ,1 ):
            model =make ().fit (X_train [tr_idx ],y_train [tr_idx ])
            scores =model .predict_proba (X_train [va_idx ])
            yhat =(scores >=0.5 ).astype (int )
            cv_metrics .append (evaluate (y_train [va_idx ],yhat ,scores ))
            logger .info ("  fold %d/%d concluído.",i ,len (folds ))
        cv_mean ,cv_std =_aggregate (cv_metrics )

        # treino no conjunto completo e avaliação no teste
        model =make ().fit (X_train ,y_train )
        scores =model .predict_proba (X_test )
        yhat =(scores >=0.5 ).astype (int )
        test_metrics =evaluate (y_test ,yhat ,scores )

        results [name ]={"cv_mean":cv_mean ,"cv_std":cv_std ,"test":test_metrics }
        logger .info ("  %s finalizado em %.1fs | teste: Accuracy=%.4f ROC-AUC=%.4f",
        name ,time .time ()-t0 ,test_metrics ["Accuracy"],test_metrics ["ROC-AUC"])

    output_path =MODELS_DIR /"results_final.json"
    with open (output_path ,"w",encoding ="utf-8")as f :
        json .dump (results ,f ,ensure_ascii =False ,indent =2 )
    logger .info ("Resultados salvos em: %s",output_path )
    return results

if __name__ =="__main__":
    run_modeling_pipeline ()
