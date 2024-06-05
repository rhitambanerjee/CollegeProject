from flask import Flask, render_template, request, redirect, url_for
from sklearn.impute import SimpleImputer
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif, VarianceThreshold, SelectFromModel, RFE
from skrebate import ReliefF
from skfeature.function.information_theoretical_based import MRMR
from skfeature.function.statistical_based import CFS
from skfeature.function.similarity_based import fisher_score
from skfeature.function.sparse_learning_based import ls_l21

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'datasets'

def preprocess_data(dataset):
    # Ensure dataset is not empty
    if dataset.empty:
        raise ValueError("The dataset is empty. Please provide a valid CSV file.")


    imputer = SimpleImputer(strategy='mean')
    dataset = pd.DataFrame(imputer.fit_transform(dataset), columns=dataset.columns)
    
    target_column = dataset.columns[-1]
    X = dataset.drop(target_column, axis=1)
    y = dataset[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return  X_train_scaled, X_test_scaled, y_train, y_test

# Feature Selection Methods
def chi_square_feature_selection(X_train, X_test, y_train, k=10):
    X_train_non_negative = X_train - X_train.min(axis=0)
    X_test_non_negative = X_test - X_test.min(axis=0)
    
    skb = SelectKBest(chi2, k=k)
    X_train_selected = skb.fit_transform(X_train_non_negative, y_train)
    X_test_selected = skb.transform(X_test_non_negative)
    return X_train_selected, X_test_selected

def f_classif_feature_selection(X_train, X_test, y_train, k=10):
    skb = SelectKBest(f_classif, k=k)
    X_train_selected = skb.fit_transform(X_train, y_train)
    X_test_selected = skb.transform(X_test)
    return X_train_selected, X_test_selected

def variance_threshold_feature_selection(X_train, X_test, y_train, threshold=0.0):
    vt = VarianceThreshold(threshold=threshold)
    X_train_selected = vt.fit_transform(X_train)
    X_test_selected = vt.transform(X_test)
    return X_train_selected, X_test_selected

def select_from_model_feature_selection(X_train, X_test, y_train, model=LogisticRegression(), threshold=None):
    sfm = SelectFromModel(model, threshold=threshold)
    X_train_selected = sfm.fit_transform(X_train, y_train)
    X_test_selected = sfm.transform(X_test)
    return X_train_selected, X_test_selected

def rfe_feature_selection(X_train, X_test, y_train, model=LogisticRegression(), k=10):
    rfe = RFE(model, n_features_to_select=k)
    X_train_selected = rfe.fit_transform(X_train, y_train)
    X_test_selected = rfe.transform(X_test)
    return X_train_selected, X_test_selected

def relief_feature_selection(X_train, X_test, y_train, k=10):
    fs = ReliefF(n_neighbors=10)
    X_train_selected = fs.fit_transform(X_train, y_train)
    X_test_selected = fs.transform(X_test)
    return X_train_selected, X_test_selected

def mrmr_feature_selection(X_train, X_test, y_train, k=10):
    selected_features = MRMR.mrmr(X_train, y_train, n_selected_features=k)
    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]
    return X_train_selected, X_test_selected

def cfs_feature_selection(X_train, X_test, y_train):
    selected_features = CFS.cfs(X_train, y_train)
    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]
    return X_train_selected, X_test_selected

def fisher_score_feature_selection(X_train, X_test, y_train, k=10):
    selected_features = fisher_score.fisher_score(X_train, y_train)
    selected_features = selected_features[:k]
    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]
    return X_train_selected, X_test_selected

def l21_feature_selection(X_train, X_test, y_train, k=10):
    selected_features = ls_l21.proximal_gradient_descent(X_train, y_train, n_selected_features=k)
    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]
    return X_train_selected, X_test_selected

def forward_feature_selection(X_train, X_test, y_train, k=10):
    estimator = RandomForestClassifier(n_estimators=10)
    selector = RFE(estimator, n_features_to_select=k, step=1)
    selector = selector.fit(X_train, y_train)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    return X_train_selected, X_test_selected

def backward_feature_elimination(X_train, X_test, y_train, k=10):
    estimator = RandomForestClassifier(n_estimators=10)
    selector = RFE(estimator, n_features_to_select=k, step=1)
    selector = selector.fit(X_train, y_train)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    return X_train_selected, X_test_selected

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
        
        # Read the CSV file with multiple possible delimiters
        delimiters = [',', ';', '\t']
        for delimiter in delimiters:
            try:
                dataset = pd.read_csv(filename, delimiter=delimiter)
                if not dataset.empty:
                    break
            except pd.errors.ParserError:
                continue
        else:
            return "Failed to parse CSV file. Please check the file format and delimiter.", 400
        
        try:
            X_train, X_test, y_train, y_test = preprocess_data(dataset)
        except ValueError as e:
            return str(e), 400
        
        feature_selection_method = request.form.get('feature')
        k = int(request.form.get('k_features', 10))
        
        if feature_selection_method == 'chi2':
            X_train_selected, X_test_selected = chi_square_feature_selection(X_train, X_test, y_train, k)
        elif feature_selection_method == 'f_classif':
            X_train_selected, X_test_selected = f_classif_feature_selection(X_train, X_test, y_train, k)
        elif feature_selection_method == 'variance_threshold':
            threshold = float(request.form.get('threshold', 0.0))
            X_train_selected, X_test_selected = variance_threshold_feature_selection(X_train, X_test, y_train, threshold)
        elif feature_selection_method == 'select_from_model':
            model = LogisticRegression()
            X_train_selected, X_test_selected = select_from_model_feature_selection(X_train, X_test, y_train, model, threshold=None)
        elif feature_selection_method == 'rfe':
            model = LogisticRegression()
            X_train_selected, X_test_selected = rfe_feature_selection(X_train, X_test, y_train, model, k)
        elif feature_selection_method == 'relief':
            X_train_selected, X_test_selected = relief_feature_selection(X_train, X_test, y_train, k)
        elif feature_selection_method == 'mrmr':
            X_train_selected, X_test_selected = mrmr_feature_selection(X_train, X_test, y_train, k)
        elif feature_selection_method == 'cfs':
            X_train_selected, X_test_selected = cfs_feature_selection(X_train, X_test, y_train)
        elif feature_selection_method == 'fisher_score':
            X_train_selected, X_test_selected = fisher_score_feature_selection(X_train, X_test, y_train, k)
        elif feature_selection_method == 'l21':
            X_train_selected, X_test_selected = l21_feature_selection(X_train, X_test, y_train, k)
        elif feature_selection_method == 'forward_selection':
            X_train_selected, X_test_selected = forward_feature_selection(X_train, X_test, y_train, k)
        elif feature_selection_method == 'backward_elimination':
            X_train_selected, X_test_selected = backward_feature_elimination(X_train, X_test, y_train, k)
        else:
            return "Invalid feature selection method", 400
        
        print("Shape of X_train_selected:", X_train_selected.shape)
        print("Shape of X_test_selected:", X_test_selected.shape)

        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100),
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'SVM': SVC(),
            'KNN': KNeighborsClassifier()
        }
        
        accuracies = {}
        
        for model_name, model in models.items():
            model.fit(X_train_selected, y_train)
            y_pred = model.predict(X_test_selected)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies[model_name] = accuracy * 100  # Convert to percentage
        
        return render_template('index.html', accuracies=accuracies)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
