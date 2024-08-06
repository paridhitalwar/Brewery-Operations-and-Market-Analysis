
| CS-688   | Web Minning and Graph Analysis            |
|----------|-------------------------------------------|
| Name     | Paridhi Talwar                            |
| Date     | 04/19/2024                                |
| Course   | Summer'2024                               |
| Project  | Brewery-Operations-and-Market-Analysis    |

# Brewery-Operations-and-Market-Analysis
## Project Overview
This project focuses on analyzing brewing parameters, sales trends, and quality metrics in craft beer production. Using Apache Spark and PySpark on Google Cloud Platform (GCP), we explore data from 2020 to 2024 to gain insights into the brewing process, identify factors influencing quality scores, and predict future outcomes using machine learning models.

The project involves data preprocessing, feature engineering, exploratory data analysis (EDA), and model development using Random Forest Regression and KMeans clustering. The ultimate goal is to enhance understanding of the brewing process and optimize production for better quality and efficiency.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Installation](#installation)
4. [Data Preprocessing](#data-preprocessing)
5. [Feature Engineering](#feature-engineering)
6. [Exploratory Data Analysis](#exploratory-data-analysis)
7. [Model Implementation](#model-implementation)
8. [Results](#results)
9. [Conclusion](#conclusion)
10. [References](#references)
11. [License](#license)

## Dataset Description
### Brewing Parameters:
•	Fermentation Time: The duration of fermentation for each batch.
•	Temperature: The temperature maintained during brewing.
•	pH Level: The acidity level of the brew.
•	Gravity: The density of the beer relative to water.
•	Ingredient Ratios: The proportions of different ingredients used.
### Beer Styles and Packaging:
•	Beer Styles: Categorized into IPA, Stout, Lager, etc.
•	Packaging Types: Includes kegs, bottles, cans, and pints.
### Quality Scores:
•	Ratings on a scale indicating the success and consistency of brewing methods.
Sales Data (USD):
•	Detailed sales figures across different locations in Bangalore.
### Supply Chain and Efficiency Metrics:
•	Volume Produced: The total volume of beer produced per batch.
•	Total Sales: Sales revenue generated per batch.

## Installation and Setup
To run this project locally, follow these steps:
1. Clone the Repository:
 ```bash
git clone https://github.com/your-username/brewing-analysis.git
cd brewing-analysis
  ```
2. Set Up a Virtual Environment:
 ```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
  ```
3. Run the Notebook:
Launch Jupyter Notebook and open brewing_analysis.ipynb to explore the analysis.

4. Configure GCP:
Ensure you have access to Google Cloud Platform with appropriate permissions to run Spark jobs.

## Data Preprocessing
Data preprocessing is a crucial step in preparing the dataset for analysis and modeling. Key tasks include:
• Schema Definition: Explicitly defined the schema to ensure accurate data types, aiding in consistency and reliability during processing.
• Handling Missing Values: Dropped rows with missing values to maintain data integrity, ensuring the dataset is complete for analysis.
• Duplicate Check: Identified and removed duplicate rows to avoid redundancy and ensure data accuracy.
• Data Transformation: Converted columns to appropriate data types, facilitating smooth analysis and computation.

## Feature Engineering
• Feature engineering involves creating new features from existing data to improve model performance. Key steps include:
• Ingredient Ratio Splitting: Split the Ingredient_Ratio column into separate columns for detailed analysis of individual ingredients.
• Brew Ratio Calculation: Calculated the brew ratio to analyze the balance of ingredients, providing insights into brewing efficiency.
• Sales Efficiency: Computed sales efficiency as the ratio of total sales to volume produced, offering a metric for assessing performance.

## Exploratory Data Analysis
Exploratory Data Analysis (EDA) helps uncover patterns and relationships within the data:
• Frequency Distribution: Analyzed the distribution of categorical variables like beer styles to identify popular categories.
• Summary Statistics: Generated summary statistics for brewing parameters, offering a comprehensive overview of data characteristics.
• Trend Analysis: Examined sales trends over time to identify patterns in consumer demand and production cycles.
• Rolling Statistics: Calculated rolling mean and standard deviation for total sales, smoothing out short-term fluctuations.
• Correlation Analysis: Evaluated correlations between brewing parameters and efficiency metrics to identify potential quality drivers.

## Model Implementation
The project implements machine learning models to predict quality scores and identify patterns in the data:
• Data Splitting: Split the data into training and test sets to ensure reliable model evaluation.
• Feature Definition: Selected relevant features for modeling based on EDA insights.
• Random Forest Regression: Implemented a Random Forest Regressor to predict quality scores, optimizing parameters using cross-validation.
• Clustering: Applied KMeans clustering to segment data into distinct groups, evaluating performance with the Silhouette score.

## Results and Evaluation
Model evaluation metrics and feature importance provide insights into model performance:
• Evaluation Metrics: The Random Forest model achieved a Test RMSE of 1.1182 and a Test MAE of 1.0002, indicating accurate predictions of quality scores.
• Feature Importance: Analyzed feature importances to identify key drivers of quality scores. Notable features include pH Level, Bitterness, and Loss During Brewing.
• Clustering Evaluation: The KMeans model effectively segmented the data, highlighting distinct groups within the dataset.

## References
• PySpark Documentation: https://spark.apache.org/docs/latest/api/python/
• Google Cloud Platform: https://cloud.google.com/
• Random Forest Regression: https://scikit-learn.org/stable/modules/ensemble.html#forest
• KMeans Clustering: https://scikit-learn.org/stable/modules/clustering.html#k-means
• Kaggle: https://www.kaggle.com/
• Stack Overflow: https://stackoverflow.com/

## License
This project is licensed under the [MIT License](LICENSE).
