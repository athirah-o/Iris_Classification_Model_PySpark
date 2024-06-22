# Load libraries and packages
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier, LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
import pandas as pd

# Load iris dataset from scikit-learn package
iris = load_iris()
iris

# Convert iris data into dataframe
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = pd.Series(iris.target)
df.head()

# Create SparkSession 
spark = SparkSession.builder.appName("Classification_Iris_Dataset").getOrCreate()

# Transform from pandas dataframe into spark dataframe
df_iris = spark.createDataFrame(df)
df_iris.printSchema()
df_iris.show(5)

# Transform feature columns using Vector Assembler
features = iris.feature_names

va = VectorAssembler(inputCols = features, outputCol='features')

va_df = va.transform(df_iris)
va_df = va_df.select(['features', 'label'])
print(va_df.show(5))

# Split data into training and testing data
(train, test) = va_df.randomSplit([0.7, 0.3], seed=1)

# Define random forest classifier model using RandomForestClassifier()
rfc = RandomForestClassifier(featuresCol="features", labelCol="label")

# Hyperparameter tuning
# Setting up a grid search with cross-validation to tune hyperparameters for a RandomForestClassifier in Apache Spark's MLlib
paramGrid_rfc = ParamGridBuilder()\
    .addGrid(rfc.numTrees, [10, 20, 30]) \
    .addGrid(rfc.maxDepth, [5, 10, 15])\
    .addGrid(rfc.impurity, ['gini', 'entropy'])\
    .build()

cv_rfc = CrossValidator(estimator=rfc,
                    estimatorParamMaps=paramGrid_rfc,
                    evaluator=MulticlassClassificationEvaluator(labelCol="label"),
                    numFolds=5)

# Fit training dataset
model_rfc = cv_rfc.fit(train)

# RANDOM FOREST MODEL
# Retrieves the best performing RandomForestClassifier model based on the evaluation metric specified (MulticlassClassificationEvaluator).
bestModel_rfc = model_rfc.bestModel

# Extract the best parameters resulted in the highest evaluation metric
bestParams_rfc = bestModel_rfc.extractParamMap()
print("Best Parameters:")
for param, value in bestParams_rfc.items():
    print(f"{param.name}: {value}")

# Predict test data based on the best RFC model
pred_rfc = bestModel_rfc.transform(test)
print("Prediction Table")
pred_rfc.show(3)

evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")

# Evaluation Metrics
acc_rfc = evaluator.evaluate(pred_rfc)
precision_rfc = evaluator.evaluate(pred_rfc, {evaluator.metricName: "weightedPrecision"})
recall_rfc = evaluator.evaluate(pred_rfc, {evaluator.metricName: "weightedRecall"})
f1_rfc = evaluator.evaluate(pred_rfc, {evaluator.metricName: "f1"})
 
print("Accuracy (Random Forest Model): ", acc_rfc)
print("Precision (Random Forest Model): ", precision_rfc)
print("Recall (Random Forest Model): ", recall_rfc)
print("F1 (Random Forest Model): ", f1_rfc)

# Confusion Matrix
y_pred_rfc = pred_rfc.select("prediction").collect()
y_orig_rfc = pred_rfc.select("label").collect()

cm_rfc = confusion_matrix(y_orig_rfc, y_pred_rfc)
print("\nConfusion Matrix (Random Forest Model):")
print(cm_rfc)

# DECISION TREE CLASSIFICATION MODEL
# Define decision tree classifier model using DecisionTreeClassifier()
dtc = DecisionTreeClassifier(featuresCol="features", labelCol="label")

# Hyperparameter tuning
# Setting up a grid search with cross-validation to tune hyperparameters for a RandomForestClassifier in Apache Spark's MLlib
paramGrid_dtc = ParamGridBuilder()\
    .addGrid(dtc.minInstancesPerNode, [1, 3, 5]) \
    .addGrid(dtc.maxDepth, [5, 10, 15])\
    .addGrid(dtc.impurity, ['gini', 'entropy'])\
    .build()

cv_dtc = CrossValidator(estimator=dtc,
                    estimatorParamMaps=paramGrid_dtc,
                    evaluator=MulticlassClassificationEvaluator(labelCol="label"),
                    numFolds=5)

# Fit training dataset
model_dtc = cv_dtc.fit(train)

# Retrieves the best performing DecisionTreeClassifier model based on the evaluation metric specified (MulticlassClassificationEvaluator).
bestModel_dtc = model_dtc.bestModel

# Extract the best parameters resulted in the highest evaluation metric
bestParams_dtc = bestModel_dtc.extractParamMap()
print("Best Parameters:")
for param, value in bestParams_dtc.items():
    print(f"{param.name}: {value}")

# Predict test data based on the best DTC model
pred_dtc = bestModel_dtc.transform(test)
print("Prediction Table")
pred_dtc.show(3)

evaluator=MulticlassClassificationEvaluator(predictionCol="prediction")

#Evaluation metrics
acc_dtc = evaluator.evaluate(pred_dtc)
precision_dtc = evaluator.evaluate(pred_dtc, {evaluator.metricName: "weightedPrecision"})
recall_dtc = evaluator.evaluate(pred_dtc, {evaluator.metricName: "weightedRecall"})
f1_dtc = evaluator.evaluate(pred_dtc, {evaluator.metricName: "f1"})
 
print("Accuracy (Decision Tree Classifier Model): ", acc_dtc)
print("Precision (Decision Tree Classifier Model): ", precision_dtc)
print("Recall (Decision Tree Classifier Model): ", recall_dtc)
print("F1 (Decision Tree Classifier Model): ", f1_dtc)

y_pred_dtc = pred_dtc.select("prediction").collect()
y_orig_dtc = pred_dtc.select("label").collect()

# Confusion matrix
cm_dtc = confusion_matrix(y_orig_dtc, y_pred_dtc)
print("\nConfusion Matrix (Decision Tree Model):")
print(cm_dtc)

# LOGISTIC REGRESSION MODEL
# Define logistic regression model using LogisticRegression()
logr = LogisticRegression(featuresCol="features", labelCol="label")

# Hyperparameter tuning
# Setting up a grid search with cross-validation to tune hyperparameters for a RandomForestClassifier in Apache Spark's MLlib
paramGrid_logr = ParamGridBuilder()\
    .addGrid(logr.regParam, [0.01, 0.1, 1.0]) \
    .addGrid(logr.elasticNetParam, [0.0, 0.5, 1.0])\
    .build()

cv_logr = CrossValidator(estimator=logr,
                    estimatorParamMaps=paramGrid_logr,
                    evaluator=MulticlassClassificationEvaluator(labelCol="label"),
                    numFolds=5)

# Define logistic regression model using LogisticRegression()
logr = LogisticRegression(featuresCol="features", labelCol="label")

# Hyperparameter tuning
# Setting up a grid search with cross-validation to tune hyperparameters for a RandomForestClassifier in Apache Spark's MLlib
paramGrid_logr = ParamGridBuilder()\
    .addGrid(logr.regParam, [0.01, 0.1, 1.0]) \
    .addGrid(logr.elasticNetParam, [0.0, 0.5, 1.0])\
    .build()

cv_logr = CrossValidator(estimator=logr,
                    estimatorParamMaps=paramGrid_logr,
                    evaluator=MulticlassClassificationEvaluator(labelCol="label"),
                    numFolds=5)

# Predict test data based on the best logr model
pred_logr = bestModel_logr.transform(test)
print("Prediction Table")
pred_logr.show(3)

evaluator=MulticlassClassificationEvaluator(predictionCol="prediction")

# Evaluate accuracy
acc_logr = evaluator.evaluate(pred_logr)
precision_logr = evaluator.evaluate(pred_logr, {evaluator.metricName: "weightedPrecision"})
recall_logr = evaluator.evaluate(pred_logr, {evaluator.metricName: "weightedRecall"})
f1_logr = evaluator.evaluate(pred_logr, {evaluator.metricName: "f1"})
 
print("Accuracy (Logistic Regression Model): ", acc_logr)
print("Precision (Logistic Regression Model): ", precision_logr)
print("Recall (Logistic Regression Model): ", recall_logr)
print("F1 (Logistic Regression Model): ", f1_logr)
 
y_pred_logr = pred_logr.select("prediction").collect()
y_orig_logr = pred_logr.select("label").collect()

# Confusion matrix
cm_logr = confusion_matrix(y_orig_logr, y_pred_logr)
print("\nConfusion Matrix (Logistic Regression Model):")
print(cm_logr)

# Stop spark session
spark.stop()