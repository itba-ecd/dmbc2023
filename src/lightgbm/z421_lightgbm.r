# LightGBM  cambiando algunos de los parametros

#limpio la memoria
rm( list=ls() )  #remove all objects
gc()             #garbage collection

require("data.table")
require("lightgbm")


PARAM <- list()
PARAM$experimento  <- "KA4210"
PARAM$semilla   <- 102191
PARAM$prob_corte  <- 0.023


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#Aqui se debe poner la carpeta de la computadora local
setwd( "X:\\gdrive\\ITBA2023dmbc\\" )  #Establezco el Working Directory

#cargo el dataset donde voy a entrenar
dataset  <- fread("./datasets/dataset_pequeno.csv", stringsAsFactors= TRUE)


dir.create( "./exp/",  showWarnings = FALSE ) 
dir.create( paste0("./exp/", PARAM$experimento, "/" ), showWarnings = FALSE )
setwd( paste0("./exp/", PARAM$experimento, "/" ) )   #Establezco el Working Directory DEL EXPERIMENTO



#paso la clase a binaria que tome valores {0,1}  enteros
dataset[ foto_mes==202107, clase01 := ifelse( clase_ternaria=="BAJA+2", 1L, 0L) ]


#los campos que se van a utilizar
campos_buenos  <- setdiff( colnames(dataset), c("clase_ternaria","clase01") )


#dejo los datos en el formato que necesita LightGBM
dtrain  <- lgb.Dataset( data= data.matrix(  dataset[ foto_mes==202107, campos_buenos, with=FALSE]),
                        label= dataset[ foto_mes==202107, clase01] )

#genero el modelo con los parametros por default
modelo  <- lgb.train( data= dtrain,
                      param= list( objective=        "binary",
                                   learning_rate=       0.01,
                                   num_iterations=    560,
                                   num_leaves=         45,
                                   feature_fraction=    0.5,
                                   min_data_in_leaf=  500,
                                   max_bin=            31,
                                   seed=  PARAM$semilla )
                    )


dapply  <- dataset[ foto_mes == 202109 ]

#aplico el modelo a los datos nuevos
prediccion  <- predict( modelo, 
                        data.matrix( dapply[ ,  campos_buenos, with=FALSE ]) )


#Genero la entrega para Kaggle
entrega  <- as.data.table( list( "numero_de_cliente"= dapply[  , numero_de_cliente],
                                 "Predicted"= prediccion >  PARAM$prob_corte)  ) #genero la salida


#genero el archivo para Kaggle
fwrite( entrega, 
        file= paste0( PARAM$experimento, "_001.csv" ), 
        sep= "," )


#ahora imprimo la importancia de variables
tb_importancia  <-  as.data.table( lgb.importance(modelo) ) 
archivo_importancia  <- paste0( PARAM$experimento, "_importancia.csv" )

fwrite( tb_importancia, 
        file= archivo_importancia, 
        sep= "\t" )

