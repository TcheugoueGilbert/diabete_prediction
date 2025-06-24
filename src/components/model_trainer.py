import os
import sys
from dataclasses import dataclass
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
               "Logistic Regression": LogisticRegression(max_iter=1000),
               "Random Forest": RandomForestClassifier(),
               "KNN": KNeighborsClassifier(),
               "SVM (Linear)": svm.SVC(kernel='linear'),
               "SVM (RBF)": svm.SVC(kernel='rbf')
            }
            params={
                "Logistic Regression":{ 
                    'C': [0.01, 0.1, 1, 10, 100],
                    'solver': ['liblinear', 'lbfgs']
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    'n_estimators': [90, 100, 200], 
                    'max_depth': [None, 5, 10, 20],
                    'min_samples_split': [2, 5, 10]
                    
                      },
                "KNN":{
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance']
                },
               "SVM (Linear)":{
                    'C': [0.01, 0.1, 1, 10, 100]
                   
                    },
                 "SVM (RBF)":{
                   'C': [0.01, 0.1, 1, 10, 100],
                   'gamma': ['scale', 0.01, 0.1, 1]
                }
                
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            test_acc = accuracy_score(y_test, predicted)
            return test_acc
            



            
        except Exception as e:
            raise CustomException(e,sys)