from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, chi2
from mlxtend.feature_selection import SequentialFeatureSelector, ExhaustiveFeatureSelector

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'datasets'

def preprocess_data(dataset, feature_selection_method='chi2', exhaustive_selection=False):
    target_column = dataset.columns[-1]
    
    X = dataset.drop(target_column, axis=1)
    y = dataset[target_column]
    
    selected_feature_indices = None
    
    if feature_selection_method == 'chi2':
        chi2_selector = SelectKBest(chi2, k=5)  # You can adjust 'k' as needed
        X = chi2_selector.fit_transform(X, y)
        selected_feature_indices = chi2_selector.get_support(indices=True)
    elif feature_selection_method == 'forward_selection':
        sfs_selector = SequentialFeatureSelector(
            GradientBoostingClassifier(random_state=42),  # Change the classifier as needed
            k_features=5,  # You can adjust the number of features to select
            forward=True,
            scoring='accuracy',
            cv=3
        )
        X = sfs_selector.fit_transform(X, y)
        selected_feature_indices = list(sfs_selector.k_feature_idx_)
    elif feature_selection_method == 'backward_elimination':
        sbs_selector = SequentialFeatureSelector(
            GradientBoostingClassifier(random_state=42),  # Change the classifier as needed
            k_features=5,  # You can adjust the number of features to select
            forward=False,  # Backward elimination
            scoring='accuracy',
            cv=3
        )
        X = sbs_selector.fit_transform(X, y)
        selected_feature_indices = list(sbs_selector.k_feature_idx_)
    
    if exhaustive_selection:
        efs_selector = ExhaustiveFeatureSelector(
            GradientBoostingClassifier(random_state=42),  # Change the classifier as needed
            min_features=1,
            max_features=X.shape[1],
            scoring='accuracy',
            cv=3,
            n_jobs=-1
        )
        efs_selector = efs_selector.fit(X, y)
        X = X[:, efs_selector.best_idx_]
        selected_feature_indices = list(efs_selector.best_idx_)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    if selected_feature_indices is not None:
        selected_feature_indices = list(selected_feature_indices)
    
    return X_train, X_test, y_train, y_test, selected_feature_indices

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
        
        dataset = pd.read_csv(filename)
        
        feature_selection_method = request.form.get('feature', 'chi2')
        exhaustive_selection = request.form.get('exhaustive', False)
        
        X_train, X_test, y_train, y_test, selected_feature_indices = preprocess_data(dataset, feature_selection_method, exhaustive_selection)
        
        models = {
            'RandomForestClassifier': RandomForestClassifier(random_state=42),
            'GradientBoostingClassifier': GradientBoostingClassifier(random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42),
            'SVC': SVC(kernel='linear', random_state=42),
            'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=5)
        }
        selected_models = request.form.getlist('model')
        model_results = ""
        
        for model_name, model in models.items():
            if model_name in selected_models:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                model_results += f"<p>{model_name}: Accuracy = {accuracy:.2%}</p>"
                print(f"{model_name}: Accuracy = {accuracy:.2%}")

        selected_feature_names = [dataset.columns[i] for i in selected_feature_indices] if selected_feature_indices is not None else []
        
        return render_template('index.html', model_results=model_results, selected_features=selected_feature_names)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
