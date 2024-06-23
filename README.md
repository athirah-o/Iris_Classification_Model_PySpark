# Classification Model using PySpark with Iris Dataset
### Project Objective:
To perform classification models (**Random Forest, Decision Tree and Logistic Regression**) on Iris dataset by leveraging PySpark MLlib library.
### About Iris Dataset:
The Iris dataset consists of 150 samples (flowers), each with 4 features (sepal length, sepal width, petal length, and petal width).<br>
The target variable (y) has three classes: 0 for setosa, 1 for versicolor, and 2 for virginica.<br>
The Iris dataset is often used as a starting point for learning and practicing various machine learning algorithms and techniques due to its simplicity and clarity.<br>
![51518iris img1](https://github.com/athirah-o/STQD6324_DataManagement_Assignment3/assets/152348953/023f8892-4055-4058-a3b7-091a96fd5d0e)
### Setup Guide:
For this project, we will setup Spark in local machine where it will run in local process, utilizing only a single JVM (Java Virtual Machine) on the local machine.<br>
In windows:
1. Install Java
3. Install Apache Spark
4. Install winutils.exe<br>

In conda environment (since we will use Jupyter Notebook to run PySpark):
1. Install PySpark 
2. Install FindSpark 
### Results and Discussions:
Include in the python notebook together with the code.
### Conclusion:
Based on the evaluation metrics (accuracy, precision, recall and F1), all three models exhibits great model performance albeit logistic regression perform slightly better than random forest and decision tree model. The similar performance shown by random forest and decision tree model might suggest that the dataset may not exhibit complexities that benefit from the ensemble approach of Random Forest. It could be relatively straightforward with well-separated features that are easily classified by both tree-based models.
