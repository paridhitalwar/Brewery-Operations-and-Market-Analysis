import os
import sys
import pandas as pd
from pandas import DataFrame
import numpy as np
from IPython.core.interactiveshell import InteractiveShell
from datetime import *
import statistics as stats
from scipy import stats
import random
import pyspark
from pyspark.rdd import RDD
from pyspark.sql import Row
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql import functions
from pyspark.sql.types import IntegerType, DoubleType, StringType
from pyspark.sql.functions import split
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from pyspark.sql.functions import corr
from pyspark.sql.functions import to_date, trunc, mean
from pyspark.sql.functions import desc, col
from pyspark.ml import Pipeline
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql.types import *

os.environ["SPARK_DRIVER_EXTRA_OPTS"] = "--illegal-access=warn"
os.environ["SPARK_EXECUTOR_EXTRA_OPTS"] = "--illegal-access=warn"

# This helps auto print out the items without explicitly using 'print'
InteractiveShell.ast_node_interactivity = "all"
# %matplotlib inline


################ Initiate a Spark Session #################

# Create a Spark session with appropriate memory allocation
spark = SparkSession.builder \
    .appName("Craft Beer Dataset") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .getOrCreate()

input= sys.argv[1]
output = sys.argv[2]
output_data = []

df = spark.read.csv(input, header=True, inferSchema=True)
print('Data frame type: ' + str(type(df)))


################ Define the Schema #################

# Define the schema explicitly
schema = StructType([
    StructField("Batch_ID", IntegerType(), True),
    StructField("Brew_Date", StringType(), True),
    StructField("Beer_Style", StringType(), True),
    StructField("SKU", StringType(), True),
    StructField("Location", StringType(), True),
    StructField("Fermentation_Time", IntegerType(), True),
    StructField("Temperature", DoubleType(), True),
    StructField("pH_Level", DoubleType(), True),
    StructField("Gravity", DoubleType(), True),
    StructField("Alcohol_Content", DoubleType(), True),
    StructField("Bitterness", IntegerType(), True),
    StructField("Color", IntegerType(), True),
    StructField("Ingredient_Ratio", StringType(), True),
    StructField("Volume_Produced", IntegerType(), True),
    StructField("Total_Sales", DoubleType(), True),
    StructField("Quality_Score", DoubleType(), True),
    StructField("Brewhouse_Efficiency", DoubleType(), True),
    StructField("Loss_During_Brewing", DoubleType(), True),
    StructField("Loss_During_Fermentation", DoubleType(), True),
    StructField("Loss_During_Bottling_Kegging", DoubleType(), True)
])

# Here's how you would rename columns if necessary
new_column_names = [
    "Batch_ID",
    "Brew_Date",
    "Beer_Style",
    "SKU",
    "Location",
    "Fermentation_Time",
    "Temperature",
    "pH_Level",
    "Gravity",
    "Alcohol_Content",
    "Bitterness",
    "Color",
    "Ingredient_Ratio",
    "Volume_Produced",
    "Total_Sales",
    "Quality_Score",
    "Brewhouse_Efficiency",
    "Loss_During_Brewing",
    "Loss_During_Fermentation",
    "Loss_During_Bottling_Kegging"
]

# Print the schema
df.printSchema()


################ Dropping Missing Values #################

# Drop rows with missing values if needed
df_clean = df.dropna()


################ Checking for Duplicates #################

# Check for duplicates
duplicates = df_clean.count() - df_clean.dropDuplicates().count()
print(f"Number of duplicate rows: {duplicates}")


################ Data Transformation #################

# Convert columns to appropriate types
df_clean = df_clean.withColumn("fermentation_time", col("fermentation_time").cast(IntegerType())) \
       .withColumn("temperature", col("temperature").cast(DoubleType())) \
       .withColumn("pH_Level", col("pH_Level").cast(DoubleType())) \
       .withColumn("gravity", col("gravity").cast(DoubleType())) \
       .withColumn("quality_score", col("quality_score").cast(IntegerType())) \
       .withColumn("total_sales", col("total_sales").cast(DoubleType()))


################ Feature Engineering #################

# Split 'Ingredient_Ratio' into separate columns
split_col = split(df_clean["Ingredient_Ratio"], ":")
df_clean = df_clean.withColumn("ingredient1", split_col.getItem(0).cast(DoubleType())) \
       .withColumn("ingredient2", split_col.getItem(1).cast(DoubleType())) \
       .withColumn("ingredient3", split_col.getItem(2).cast(DoubleType())) \
       .withColumn("brew_ratio", col("ingredient1") / (col("ingredient2") + col("ingredient3"))) \
       .withColumn("sales_efficiency", col("total_sales") / col("Volume_Produced")) \
       .drop("Ingredient_Ratio")


################ Exploratory Data Analysis #################

################ Calculating Frequency Distribution #################

# Count the number of unique values in categorical columns
categorical_columns = ["beer_style"]

# Calculate distinct value counts
for col in categorical_columns:
    distinct_count = df_clean.select(col).distinct().count()
    print(f"Number of distinct values in '{col}': {distinct_count}")

# Frequency distribution for categorical columns
for col in categorical_columns:
    value_counts = df_clean.groupBy(col).count().orderBy("count", ascending=False)
    print(f"Frequency distribution for '{col}':")
    value_counts.show(truncate=False)


################ Summary Statistics #################

df.describe(["fermentation_time", "temperature", "pH_Level", "gravity"]).show()


################ Mean Statistics #################

# Calculate mean statistics by beer style
grouped_stats = df.groupBy("beer_style") \
                  .agg({
                      "fermentation_time": "mean",
                      "temperature": "mean",
                      "pH_Level": "mean",
                      "gravity": "mean",
                      "quality_score": "mean",
                      "total_sales": "mean"
                  })

# Use the imported col function here
sorted_stats = grouped_stats.orderBy("avg(quality_score)", ascending=False)

# Show only the top 5 beer styles
top_5_stats = sorted_stats.limit(5)

# Show grouped statistics
print("Top 5 beer styles by mean quality score:")
top_5_stats.show(truncate=False)


################ Trend Analysis#################

sales_trends = df.groupBy("Brew_Date").agg({"total_sales": "sum"}).orderBy("Brew_Date")
sales_trends.show()


################ Rolling Mean and Standard Deviation #################

# Calculate rolling mean and standard deviation for total_sales
window_spec = Window.orderBy("Brew_Date").rowsBetween(-7, 0)  # 7-day rolling window

df_rolling_stats = df_clean.withColumn("rolling_mean_sales", F.avg("total_sales").over(window_spec)) \
                           .withColumn("rolling_std_sales", F.stddev("total_sales").over(window_spec))

df_rolling_stats.select("Brew_Date", "total_sales", "rolling_mean_sales", "rolling_std_sales").show(5, truncate=False)


################ Correlation Analysis #################

# Calculate correlations
corr_ing2_ing3 = df_clean.select(corr("ingredient2", "ingredient3")).collect()[0][0]
corr_ing2_brew = df_clean.select(corr("ingredient2", "Brewhouse_Efficiency")).collect()[0][0]
corr_ing3_brew = df_clean.select(corr("ingredient3", "Brewhouse_Efficiency")).collect()[0][0]

# Print correlations
print("Correlation between ingredient2 and ingredient3:", corr_ing2_ing3)
print("Correlation between ingredient2 and Brewhouse_Efficiency:", corr_ing2_brew)
print("Correlation between ingredient3 and Brewhouse_Efficiency:", corr_ing3_brew)


################ Mean Quality score and Brewhouse Efficiency #################

# Assuming you have a SparkSession created as 'spark'
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

# Convert 'Brew_Date' to date type in PySpark DataFrame
df_clean = df_clean.withColumn("Brew_Date", to_date(df_clean["Brew_Date"]))

# Calculate mean Quality Score and Brewhouse Efficiency per month
mean_quality_score = df_clean.groupBy(trunc("Brew_Date", "month").alias("month")).agg(mean("quality_score").alias("mean_quality"))
mean_brewhouse_efficiency = df_clean.groupBy(trunc("Brew_Date", "month").alias("month")).agg(mean("Brewhouse_Efficiency").alias("mean_efficiency"))

# Show the results
mean_quality_score.show(5)
mean_brewhouse_efficiency.show(5)


################ Calculating Control Limits #################

# Calculate mean Quality Score for all months
quality_score_mean = mean_quality_score.select(mean("mean_quality")).collect()[0][0]

# Calculate mean Brewhouse Efficiency for all months
brewhouse_efficiency_mean = mean_brewhouse_efficiency.select(mean("mean_efficiency")).collect()[0][0]

# Calculate standard deviation of Quality Score for all months
quality_score_std = mean_quality_score.select(F.stddev("mean_quality")).collect()[0][0]

# Calculate standard deviation of Brewhouse Efficiency for all months
brewhouse_efficiency_std = mean_brewhouse_efficiency.select(F.stddev("mean_efficiency")).collect()[0][0]

# Calculate control limits for Quality Score
quality_score_control_limit_upper = quality_score_mean + 3 * quality_score_std
quality_score_control_limit_lower = quality_score_mean - 3 * quality_score_std

# Calculate control limits for Brewhouse Efficiency
brewhouse_efficiency_control_limit_upper = brewhouse_efficiency_mean + 3 * brewhouse_efficiency_std
brewhouse_efficiency_control_limit_lower = brewhouse_efficiency_mean - 3 * brewhouse_efficiency_std

# Summary statistics
print("Quality Score Control Limits:")
print(f"Upper Control Limit: {quality_score_control_limit_upper}")
print(f"Lower Control Limit: {quality_score_control_limit_lower}")

print("\nBrewhouse Efficiency Control Limits:")
print(f"Upper Control Limit: {brewhouse_efficiency_control_limit_upper}")
print(f"Lower Control Limit: {brewhouse_efficiency_control_limit_lower}")


# Out-of-Control Points for Quality Score
out_of_control_quality = mean_quality_score.filter(
    (F.col("mean_quality").cast("float") > quality_score_control_limit_upper) |
    (F.col("mean_quality").cast("float") < quality_score_control_limit_lower)
)
out_of_control_quality.show()

# Out-of-Control Points for Brewhouse Efficiency
out_of_control_efficiency = mean_brewhouse_efficiency.filter(
    (F.col("mean_efficiency").cast("float") > brewhouse_efficiency_control_limit_upper) |
    (F.col("mean_efficiency").cast("float") < brewhouse_efficiency_control_limit_lower)
)

if out_of_control_efficiency.count() > 0:
    out_of_control_efficiency.show()
else:
    print("No out-of-control points found for Brewhouse Efficiency.")


################ Splitting the Data #################

# Split the data
train_data, test_data = df_clean.randomSplit([0.8, 0.2], seed=42)

# Print the count of records in each dataset
print(f"Training Data Count: {train_data.count()}")
# print(f"Validation Data Count: {validation_data.count()}")
print(f"Test Data Count: {test_data.count()}")


################ Defining Features #################

# Define all feature columns
feature_columns = [
    "fermentation_time",
    "temperature",
    "pH_Level",
    "gravity",
    "brew_ratio",
    "sales_efficiency",
    "Alcohol_Content",
    "Bitterness",
    "Color",
    "Volume_Produced",
    "Loss_During_Brewing",
    "Loss_During_Fermentation",
    "Loss_During_Bottling_Kegging",
    "ingredient2",
    "ingredient3",
    "Brewhouse_Efficiency"
]
print(feature_columns)
# Assemble features
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
df_features = assembler.transform(df_clean)


################ Model Fitting #################

# Define the RandomForestRegressor to use scaled features
rf = RandomForestRegressor(featuresCol="features", labelCol="quality_score", seed=42)

# Create a pipeline with the correct stages
pipeline = Pipeline(stages=[assembler, rf])


################ Cross Validation #################

# Set up a parameter grid
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 20, 30]) \
    .addGrid(rf.maxDepth, [5, 10]) \
    .build()

# Set up cross-validation
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator(labelCol="quality_score", metricName="rmse"),
                          numFolds=3)  # 5-fold cross-validation

# Train the model with cross-validation
cvModel = crossval.fit(train_data)
print("Crossvaalidation fitting done")
# Make predictions on the test data
predictions_test = cvModel.transform(test_data)


################ Model Evaluation #################

print("Evaluation:")
# Evaluate the model on the test data
evaluator = RegressionEvaluator(labelCol="quality_score", predictionCol="prediction")
rmse_test = evaluator.evaluate(predictions_test, {evaluator.metricName: "rmse"})
mae_test = evaluator.evaluate(predictions_test, {evaluator.metricName: "mae"})

print(f"Test RMSE after cross-validation: {rmse_test:.4f}")
print(f"Test MAE after cross-validation: {mae_test:.4f}")

# Show a sample of actual vs. predicted values
predictions_test.select("quality_score", "prediction").show(10)

# Write to output file
output_data.append("Model Output\n")
output_data.append(f'Test RMSE: {rmse_test:.4f}\n')
output_data.append(f'Test MAE: {mae_test:.4f}\n')


################ Feature Importance #################

# Extract feature importances
best_model = cvModel.bestModel.stages[-1]  # Access the RandomForest model in the pipeline
feature_importances = best_model.featureImportances

# Print feature importances
print("Feature Importances:")
for feature, importance in zip(feature_columns, feature_importances):
    print(f"{feature}: {importance:.4f}")
    

################ Clustering #################

# Set up KMeans with a specified number of clusters (k=3) and a random seed for reproducibility
kmeans = KMeans(featuresCol='features', k=3, seed=42)

# Fit the KMeans model to the scaled data
kmeans_model = kmeans.fit(df_features)

# Make predictions (assign clusters to each data point)
cluster_predictions = kmeans_model.transform(df_features)

# Evaluate the clustering performance using the Silhouette score
evaluator = ClusteringEvaluator(featuresCol='features')
silhouette = evaluator.evaluate(cluster_predictions)
print(f"Silhouette with squared euclidean distance = {silhouette:.4f}")

# Show the cluster centers
centers = kmeans_model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)

output_data.append("Cluster Centers:\n")
for idx, center in enumerate(centers):
  center_str = ", ".join([f"{val:.4f}" for val in center])
  output_data.append(f"Cluster {idx + 1} Center: [{center_str}]\n")

output_rdd = spark.sparkContext.parallelize(output_data)
output_rdd.saveAsTextFile(output)

spark.stop()