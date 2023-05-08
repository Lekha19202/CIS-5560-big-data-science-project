# Databricks notebook source
# MAGIC %md
# MAGIC ## CIS 5560 final Project
# MAGIC ## 6 algorithms randomforestRegressor, GBT,FM,SVM, Decsion tree and logistic regression. 
# MAGIC ##### Professor - Jongwook Woo (jwoo5@calstatela.edu)

# COMMAND ----------

# Import Spark SQL and Spark ML libraries
from pyspark.sql.types import *
from pyspark.sql.functions import *

from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression, FMRegressor, RandomForestRegressor, GBTRegressionModel, GBTRegressor
from pyspark.ml.classification import DecisionTreeClassifier, LogisticRegression, LinearSVC
from pyspark.ml.feature import VectorAssembler, MinMaxScaler, StringIndexer,VectorIndexer
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit
from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator, MulticlassClassificationEvaluator

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from time import time



# COMMAND ----------

IS_DB = True # Run the code in Databricks

PYSPARK_CLI = False
if PYSPARK_CLI:
    sc = SparkContext.getOrCreate()
    spark = SparkSession(sc)

# COMMAND ----------

# DataFrame Schema, that should be a Table schema 
Schema = StructType([
  StructField("Timestamp", StringType(), False),
  StructField("From Bank", IntegerType(), False),
  StructField("Account 1", StringType(), False),
  StructField("To Bank", IntegerType(), False),
  StructField("Account 2", StringType(), False),
  StructField("Amount Received", FloatType(), False),
  StructField("Receiving Currency", StringType(), False),
  StructField("Amount Paid", FloatType(), False),
  StructField("Payment Currency", StringType(), False),
  StructField("Payment Format", StringType(), False),
  StructField("Is Laundering", IntegerType(), False),  
])

# COMMAND ----------

# File location and type
file_location = ["/FileStore/tables/money__Laundering.csv"]
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)
  
display(df)

# COMMAND ----------

# Converting hexa decimal to integer 
df = df.withColumn('Account2', conv(df['Account2'], 16, 10))
df = df.withColumn('Account4', conv(df['Account4'], 16, 10))
df.show()


# COMMAND ----------


# Create a view or table
temp_table_name = "money_csv"
df.createOrReplaceTempView(temp_table_name)


# COMMAND ----------

# for spark submit
if PYSPARK_CLI:
    csv = spark.read.csv('money__Laundering.csv', inferSchema=True, header=True)
else:
    csv = spark.sql("SELECT * FROM money_csv")

csv.show(5)

# COMMAND ----------

#creating dataframe
data = df.select("Timestamp", "From Bank", "Account2", "To Bank", "Account4", "Amount Received", "Receiving Currency", "Amount Paid", "Payment Currency", "Payment Format", ((col("Is Laundering")).cast("Double").alias("label")))
data.show()

# Split the data
splits = data.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1].withColumnRenamed("label", "trueLabel")

# COMMAND ----------

#Finding the count of training and testing rows
train_rows = train.count()
test_rows = test.count()
print("Training Rows:", train_rows, " Testing Rows:", test_rows)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define the Pipeline
# MAGIC Defineing a pipeline that creates a feature vector and trains a the models

# COMMAND ----------

# converting sting values into numeric values
strIdx1 = StringIndexer(inputCol = "Timestamp", outputCol = "TimestampIdx")

strIdx2 = StringIndexer(inputCol = "Account2", outputCol = "Account1Idx")
strIdx3 = StringIndexer(inputCol = "Account4", outputCol = "Account2Idx")
strIdx4 = StringIndexer(inputCol = "Receiving Currency", outputCol = "ReceivingCurrencyIdx")
strIdx5 = StringIndexer(inputCol = "Payment Currency", outputCol = "PaymentCurrencyIdx")
strIdx6 = StringIndexer(inputCol = "Payment Format", outputCol = "PaymentFormatIdx")



# COMMAND ----------

#placing the catgorical values in a catvect
catVect = VectorAssembler(inputCols = [ "TimestampIdx","Account1Idx", "Account2Idx","ReceivingCurrencyIdx",  "PaymentCurrencyIdx", "PaymentFormatIdx"], outputCol="catFeatures")
catIdx = VectorIndexer(inputCol = catVect.getOutputCol(), outputCol = "idxCatFeatures")

featVect = VectorAssembler(inputCols=["catFeatures", "normFeatures"], outputCol="features")  #="features1")


# COMMAND ----------

# Normalizing the numeric values 
assembler = VectorAssembler(inputCols =["From Bank","To Bank","Amount Received","Amount Paid"], outputCol="features")
# number vector is normalized
minMax = MinMaxScaler(inputCol = assembler.getOutputCol(), outputCol="normFeatures")

finalVect = VectorAssembler(inputCols=["TimestampIdx", "From Bank", "Account2", "To Bank","Amount Received","Amount Paid","Account4", "Receiving Currency", "Payment Currency", "Payment Format"], outputCol="featuresfin")



# COMMAND ----------

# MAGIC %md
# MAGIC ##RandomForestRegressor

# COMMAND ----------

# RandomForestRegressor

rf = RandomForestRegressor(labelCol="label", featuresCol="features")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Tune Parameters
# MAGIC tuning parameters to find the best model for your data. To do this we are using **CrossValidator anf trainValidationSplit** class to evaluate each combination of parameters defined in a **ParameterGrid** against multiple *folds* of the data split into training and validation datasets, in order to find the best performing parameters. 

# COMMAND ----------

paramGrid = ParamGridBuilder() \
  .addGrid(rf.maxDepth, [2, 3]) \
  .addGrid(rf.maxBins, [5, 10]) \
  .addGrid(rf.minInfoGain, [0.0]) \
  .build()

# COMMAND ----------

pipeline = Pipeline(stages=[strIdx1, strIdx2, strIdx3, strIdx4, strIdx5, strIdx6, assembler, minMax, rf])


start = time()

#tvs = TrainValidationSplit(estimator=pipeline, evaluator=BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName="areaUnderROC"), estimatorParamMaps=paramGrid, trainRatio=0.8)

model = pipeline.fit(train)
#model = tvs.fit(train)

end = time()
phrase = 'Random Forest tvs testing'
print('{} takes {} seconds'.format(phrase, (end - start))) #round(end - start, 2)))

time_rf_tvs = end - start

# COMMAND ----------

rfModel = model.stages[-1]
print(rfModel.toDebugString)

# COMMAND ----------

#feature importance
import pandas as pd
featureImp = pd.DataFrame(list(zip(finalVect.getInputCols(),rfModel.featureImportances)),columns=["feature", "importance"])
featureImp.sort_values(by="importance", ascending=False)

# COMMAND ----------

import pandas as pd
featureImp = pd.DataFrame(list(zip(assembler.getInputCols(),rfModel.featureImportances)),columns=["feature", "importance"])
featureImp.sort_values(by="importance", ascending=False)

# COMMAND ----------

# TODO: params refered to the reference above
paramGrid2 = (ParamGridBuilder() \
             .addGrid(rf.maxDepth, [3, 5]) \
             .addGrid(rf.maxBins, [10, 15]) \
             .addGrid(rf.minInfoGain, [0.0]) \
             .build())

# COMMAND ----------

#Randomforest in TrainValidationSplit
pipelinetvs = Pipeline(stages=[strIdx1, strIdx2, strIdx3, strIdx4, strIdx5, strIdx6,catVect,catIdx,assembler, minMax, rf])

start = time()

tvs2 = TrainValidationSplit(estimator=pipelinetvs, evaluator=BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName="areaUnderROC"), estimatorParamMaps=paramGrid2, trainRatio=0.8)

# the second best model
modeltvs = tvs2.fit(train)

end = time()
phrase = 'Random Forest tvs2 testing'
print('{} takes {} seconds'.format(phrase, (end - start))) #round(end - start, 2)))


time_rf_tvs2 = end - start

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cross Validator with parameters

# COMMAND ----------

# TODO: params refered to the reference above
paramGridCV = ParamGridBuilder() \
  .addGrid(rf.maxDepth, [2, 3]) \
  .addGrid(rf.maxBins, [5, 10]) \
  .addGrid(rf.minInfoGain, [0.0]) \
  .build()

# COMMAND ----------

#randomforest with CrossValidator
pipelineCV = Pipeline(stages=[strIdx1, strIdx2, strIdx3, strIdx4, strIdx5, strIdx6,catVect,catIdx,assembler, minMax, rf])

start = time()

# TODO: K = 3
# K=3, 5
K = 3
cv =  CrossValidator(estimator=pipelineCV, estimatorParamMaps=paramGridCV, evaluator=BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName="areaUnderROC"),numFolds=K)

# the third best model
modelCV = cv.fit(train)

end = time()
phrase = 'Random Forest testing'
print('{} takes {} seconds'.format(phrase, (end - start))) #round(end - start, 2)))

time_rf_cv = end - start

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test the Model

# COMMAND ----------

# list prediction
predictiontvs = modeltvs.transform(test)
prediction = model.transform(test)
predictionCV = modelCV.transform(test)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Examine the Predicted and Actual Values

# COMMAND ----------

predicted = prediction.select("features","prediction", "trueLabel")
predictedtvs = predictiontvs.select("features","prediction", "trueLabel")
predictedCV = predictionCV.select("features","prediction", "trueLabel")
predictedCV.show(20)
predicted.show(20)
predictedtvs.show(20)


# COMMAND ----------

#the auc of the three random forest models 
evaluator = BinaryClassificationEvaluator(labelCol="trueLabel", rawPredictionCol="prediction", metricName="areaUnderROC")
auc_tvs1_rf = evaluator.evaluate(prediction)
print("AUC = ", auc_tvs1_rf)

evaluator = BinaryClassificationEvaluator(labelCol="trueLabel", rawPredictionCol="prediction", metricName="areaUnderROC")
auc_tvs_rf = evaluator.evaluate(predictiontvs)
print("AUC = ", auc_tvs_rf)

evaluator = BinaryClassificationEvaluator(labelCol="trueLabel", rawPredictionCol="prediction", metricName="areaUnderROC")
auc_cv_rf = evaluator.evaluate(predictionCV)
print("AUC = ", auc_cv_rf)


# COMMAND ----------

# MAGIC %md
# MAGIC ##GBT

# COMMAND ----------

#GBT

gbt = GBTRegressor(labelCol="label", featuresCol="features")


# COMMAND ----------

paramGrid2 = (ParamGridBuilder() \
             .addGrid(gbt.maxDepth, [3, 5]) \
             .addGrid(gbt.maxBins, [10, 15]) \
             .addGrid(gbt.minInfoGain, [0.0]) \
             .build())


# COMMAND ----------

#GBT using TrainValidationSplit
pipelinegbt = Pipeline(stages=[strIdx1, strIdx2, strIdx3, strIdx4, strIdx5, strIdx6,catVect,catIdx,assembler, minMax, gbt])

start2 = time()

gbt_tvs = TrainValidationSplit(estimator=pipelinegbt, evaluator=BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName="areaUnderROC"), estimatorParamMaps=paramGrid2, trainRatio=0.8)

model = gbt_tvs.fit(train)

end2 = time()
phrase = 'GBT testing'
print('{} takes {} seconds'.format(phrase, (end2 - start2))) #round(end - start, 2)))

time_gbt_tvs= end - start


# COMMAND ----------

#GBT using CrossValidator

start2 = time()

gbt_cv = CrossValidator(estimator=pipelinegbt, evaluator=BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName="areaUnderROC"), estimatorParamMaps=paramGrid2, numFolds=K)

# the second best model

model2 = gbt_cv.fit(train)

end2 = time()
phrase = 'GBT testing'
print('{} takes {} seconds'.format(phrase, (end2 - start2))) #round(end - start, 2)))

time_gbt_cv= end - start

# COMMAND ----------

prediction_gbt_tvs = model.transform(test)
predicted_gbt_tvs = prediction_gbt_tvs.select("normFeatures", "prediction", "trueLabel")
predicted_gbt_tvs.show() 

prediction_gbt_cv = model2.transform(test)
predicted_gbt_cv = prediction_gbt_cv.select("normFeatures", "prediction", "trueLabel")
predicted_gbt_cv.show()

# COMMAND ----------

#AUC for the GBT
evaluator = BinaryClassificationEvaluator(labelCol="trueLabel", rawPredictionCol="prediction", metricName="areaUnderROC")
auc_tvs_gbt = evaluator.evaluate(prediction_gbt_tvs)
print("AUC = ", auc_tvs_gbt)

evaluator = BinaryClassificationEvaluator(labelCol="trueLabel", rawPredictionCol="prediction", metricName="areaUnderROC")
auc_cv_gbt = evaluator.evaluate(prediction_gbt_cv)
print("AUC = ", auc_cv_gbt)

# COMMAND ----------

# MAGIC %md
# MAGIC ##FM

# COMMAND ----------

#FM
fm = FMRegressor(labelCol="label", featuresCol="normFeatures")

# COMMAND ----------

paramGrid2 = (ParamGridBuilder() \
             .addGrid(fm.maxIter, [5, 10])\
             .build())


# COMMAND ----------

#FM using TrainValidationSplit
pipelinefm = Pipeline(stages=[strIdx1, strIdx2, strIdx3, strIdx4, strIdx5, strIdx6,catVect,catIdx,assembler, minMax, fm])

start3 = time()

fm_tvs = TrainValidationSplit(estimator=pipelinefm, evaluator=BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName="areaUnderROC"), estimatorParamMaps=paramGrid2, trainRatio=0.8)


# the second best model
model = fm_tvs.fit(train)

end3 = time()
phrase = 'FM testing'
print('{} takes {} seconds'.format(phrase, (end3 - start3))) #round(end - start, 2)))

time_fm_tvs= end - start

# COMMAND ----------

#FM using CrossValidator

start3 = time()

fm_cv = CrossValidator(estimator=pipelinefm, evaluator=BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName="areaUnderROC"), estimatorParamMaps=paramGrid2, numFolds=K)

model2 = fm_cv.fit(train)

end3 = time()
phrase = 'FM testing'
print('{} takes {} seconds'.format(phrase, (end3 - start3))) #round(end - start, 2)))

time_fm_cv= end - start

# COMMAND ----------

prediction_fm_tvs = model.transform(test)
predicted_fm_tvs = prediction_fm_tvs.select("normFeatures", "prediction", "trueLabel")
predicted_fm_tvs.show()

prediction_fm_cv = model2.transform(test)
predicted_fm_cv = prediction_fm_cv.select("normFeatures", "prediction", "trueLabel")
predicted_fm_cv.show()

# COMMAND ----------

#auc of FM
evaluator = BinaryClassificationEvaluator(labelCol="trueLabel", rawPredictionCol="prediction", metricName="areaUnderROC")
auc_tvs_fm = evaluator.evaluate(predicted_fm_tvs)
print("AUC = ", auc_tvs_fm)

evaluator = BinaryClassificationEvaluator(labelCol="trueLabel", rawPredictionCol="prediction", metricName="areaUnderROC")
auc_cv_fm = evaluator.evaluate(predicted_fm_cv)
print("AUC = ", auc_cv_fm)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Support Vector Machines

# COMMAND ----------

lsvc = LinearSVC(labelCol="label", featuresCol="features", maxIter=50)

pipelinesvc = Pipeline(stages=[strIdx1, strIdx2, strIdx3, strIdx4, strIdx5, strIdx6,catVect,catIdx,assembler, minMax, lsvc])

#SVM with CrossValidator

start4 = time()


svc_cv = CrossValidator(estimator=pipelinesvc, evaluator=BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName="areaUnderROC"), estimatorParamMaps=paramGrid2, numFolds=3)

modelsvc_cv = svc_cv.fit(train)


end4 = time()
phrase = 'SVM testing'
print('{} takes {} seconds'.format(phrase, (end4 - start4))) #round(end - start, 2)))

time_svm_cv= end - start

# svc_tvs = TrainValidationSplit(estimator=pipelinesvc, evaluator=RegressionEvaluator(), estimatorParamMaps=paramGrid2, trainRatio=0.8)
# svc_cv = CrossValidator(estimator=pipelinesvc, evaluator=RegressionEvaluator(), estimatorParamMaps=paramGrid2, numFolds=3)

# COMMAND ----------

#SVM with TrainValidationSplit
start4 = time()


svc_tvs = TrainValidationSplit(estimator=pipelinesvc, evaluator=BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName="areaUnderROC"), estimatorParamMaps=paramGrid2, trainRatio=0.8)

modelsvc_tvs = svc_tvs.fit(train)

end4 = time()
phrase = 'SVM testing'
print('{} takes {} seconds'.format(phrase, (end4 - start4))) #round(end - start, 2)))

time_svm_tvs= end - start


# COMMAND ----------


predictionSVM_tvs = modelsvc_tvs.transform(test)
predictedSVM_tvs = predictionSVM_tvs.select("normFeatures", "prediction", "trueLabel")
predictedSVM_tvs.show()

predictionSVM_cv = modelsvc_cv.transform(test)
predictedSVM_cv = predictionSVM_cv.select("normFeatures", "prediction", "trueLabel")
predictedSVM_cv.show()

# COMMAND ----------

evaluatorSVM_tvs = BinaryClassificationEvaluator(labelCol="trueLabel", rawPredictionCol="prediction", metricName="areaUnderROC")
auc_SVM_tvs = evaluatorSVM_tvs.evaluate(predictionSVM_tvs)
print("AUC = ", auc_SVM_tvs)

evaluatorSVM_cv = BinaryClassificationEvaluator(labelCol="trueLabel", rawPredictionCol="prediction", metricName="areaUnderROC")
auc_SVM_cv = evaluatorSVM_cv.evaluate(predictionSVM_cv)
print("AUC = ", auc_SVM_cv)

# COMMAND ----------

# MAGIC %md
# MAGIC ##LogisticRegression

# COMMAND ----------

lr = LogisticRegression(labelCol="label",featuresCol="normFeatures",maxIter=10,regParam=0.3)


# COMMAND ----------

#paramGrid2 = (ParamGridBuilder() \
#             .build())

paramGrid2=(ParamGridBuilder()
             .addGrid(lr.regParam, [0.01, 0.1, 0.5, 1.0, 2.0])
             .addGrid(lr.elasticNetParam, [0.0, 0.25, 0.5, 0.75, 1.0])
             .addGrid(lr.maxIter, [1, 5, 10, 20, 50])
             .build())

# COMMAND ----------

#LogisticRegression with TrainValidationSplit
pipelinelr = Pipeline(stages=[strIdx1, strIdx2, strIdx3, strIdx4, strIdx5, strIdx6,catVect,catIdx,assembler, minMax, lr])

start5 = time()

lr_tvs = TrainValidationSplit(estimator=pipelinelr, evaluator=RegressionEvaluator(), estimatorParamMaps=paramGrid2, trainRatio=0.8)

# the second best model
model = lr_tvs.fit(train)

end5 = time()
phrase = 'Logistic Regression testing'
print('{} takes {} seconds'.format(phrase, (end5 - start5))) #round(end - start, 2)))

time_lr_tvs= end - start


# COMMAND ----------

##LogisticRegression with CrossValidator

start5 = time()

lr_cv = CrossValidator(estimator=pipelinelr, evaluator=RegressionEvaluator(), estimatorParamMaps=paramGrid2, numFolds=3)

model2 = lr_cv.fit(train)

end5 = time()
phrase = 'Logistic Regression testing'
print('{} takes {} seconds'.format(phrase, (end5 - start5))) #round(end - start, 2)))

time_lr_cv= end - start



# COMMAND ----------

#logistic regression 
pipeline = Pipeline(stages=[strIdx1, strIdx2, strIdx3, strIdx4, strIdx5, strIdx6,catVect,catIdx,assembler, minMax, lr])

start5 = time()


Model3 = pipeline.fit(train)
prediction = Model3.transform(test)

end5 = time()
phrase = 'Logistic Regression testing'
print('{} takes {} seconds'.format(phrase, (end5 - start5))) #round(end - start, 2)))

time_lr= end - start



# COMMAND ----------

predicted = prediction.select("features", "prediction", "trueLabel")
predicted.show(100, truncate=False)

# COMMAND ----------

prediction_lr_tvs = model.transform(test)
predicted_lr_tvs = prediction_fm_tvs.select("normFeatures", "prediction", "trueLabel")
predicted_lr_tvs.show()

prediction_lr_cv = model2.transform(test)
predicted_lr_cv = prediction_fm_cv.select("normFeatures", "prediction", "trueLabel")
predicted_lr_cv.show()

prediction_lr = Model3.transform(test)
predicted_lr = prediction.select("normFeatures", "prediction", "trueLabel")
predicted_lr.show()

# COMMAND ----------

# Precision and Recall
tp = float(prediction_lr_tvs.filter("prediction == 1.0 AND truelabel == 1").count())
fp = float(prediction_lr_tvs.filter("prediction == 1.0 AND truelabel == 0").count())
tn = float(prediction_lr_tvs.filter("prediction == 0.0 AND truelabel == 0").count())
fn = float(prediction_lr_tvs.filter("prediction == 0.0 AND truelabel == 1").count())
metrics2 = spark.createDataFrame([
 ("TP", tp),
 ("FP", fp),
 ("TN", tn),
 ("FN", fn),
 ("Precision", tp / (tp + fp)),
 ("Recall", tp / (tp + fn))],["metric", "value"])
metrics2.show()

# COMMAND ----------

evaluator = BinaryClassificationEvaluator(labelCol="trueLabel", rawPredictionCol="prediction", metricName="areaUnderROC")
auc_tvs_lr = evaluator.evaluate(prediction_lr_tvs)
print("AUC = ", auc_tvs_lr)

evaluator = BinaryClassificationEvaluator(labelCol="trueLabel", rawPredictionCol="prediction", metricName="areaUnderROC")
auc_cv_lr = evaluator.evaluate(prediction_lr_cv)
print("AUC = ", auc_cv_lr)

evaluator = BinaryClassificationEvaluator(labelCol="trueLabel", rawPredictionCol="prediction", metricName="areaUnderROC")
auc_lr = evaluator.evaluate(predicted)
print("AUC = ", auc_lr)

# COMMAND ----------

# MAGIC %md
# MAGIC ##DecisionTreeClassifier

# COMMAND ----------

dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")


# COMMAND ----------

paramGrid2=(ParamGridBuilder()
             .addGrid(dt.maxDepth, [2, 5, 10, 20, 30])
             .addGrid(dt.maxBins, [10, 20, 40, 80, 100])
             .build())

# COMMAND ----------

#DecisionTreeClassifier with TrainValidationSplit
pipelinedt = Pipeline(stages=[strIdx1, strIdx2, strIdx3, strIdx4, strIdx5, strIdx6,catVect,catIdx,assembler, minMax, dt])

start6 = time()

dt_tvs = TrainValidationSplit(estimator=pipelinedt, evaluator=RegressionEvaluator(), estimatorParamMaps=paramGrid2, trainRatio=0.8)

# the second best model
model = dt_tvs.fit(train)

end6 = time()
phrase = 'Decision Tree testing'
print('{} takes {} seconds'.format(phrase, (end6 - start6))) #round(end - start, 2)))

time_dt_tvs= end - start


# COMMAND ----------

#DecisionTreeClassifier with CrossValidator
start6 = time()

dt_cv = CrossValidator(estimator=pipelinedt, evaluator=RegressionEvaluator(), estimatorParamMaps=paramGrid2, numFolds=K)
model2 = dt_cv.fit(train)

end6 = time()
phrase = 'Decision Tree testing'
print('{} takes {} seconds'.format(phrase, (end6 - start6))) #round(end - start, 2)))

time_dt_cv= end - start



# COMMAND ----------

prediction_dt_tvs = model.transform(test)
predicted_dt_tvs = prediction_dt_tvs.select("normFeatures", "prediction", "trueLabel")
predicted_dt_tvs.show()

prediction_dt_cv = model2.transform(test)
predicted_dt_cv = prediction_dt_cv.select("normFeatures", "prediction", "trueLabel")
predicted_dt_cv.show()

# COMMAND ----------

#AUC of Decision Tree
evaluator = BinaryClassificationEvaluator(labelCol="trueLabel", rawPredictionCol="prediction", metricName="areaUnderROC")
auc_tvs_dt = evaluator.evaluate(predicted_dt_tvs)
print("AUC = ", auc_tvs_dt)

evaluator = BinaryClassificationEvaluator(labelCol="trueLabel", rawPredictionCol="prediction", metricName="areaUnderROC")
auc_cv_dt = evaluator.evaluate(predicted_dt_cv)
print("AUC = ", auc_cv_dt)

# COMMAND ----------

#all metrics in a tabular format
metrics = spark.createDataFrame([
 ("auc_tvs1_rf", auc_tvs1_rf),
 ("auc_tvs_rf", auc_tvs_rf),
 ("auc_cv_rf", auc_cv_rf),
 ("auc_tvs_gbt", auc_tvs_gbt),
 ("auc_cv_gbt", auc_cv_gbt),
 ("auc_tvs_fm", auc_tvs_fm),
 ("auc_cv_fm",auc_cv_fm),
 ("auc_tvs_svm", auc_SVM_tvs),
 ("auc_cv_svm", auc_SVM_cv),
 ("auc_tvs_lr",auc_tvs_lr),
 ("auc_cv_lr",auc_cv_lr),
 ("auc_lr",auc_lr),
 ("auc_tvs_dt",auc_tvs_dt),
 ("auc_cv_dt",auc_cv_dt),
  
 ],["metric", "value"])

metrics.show() 

# COMMAND ----------


