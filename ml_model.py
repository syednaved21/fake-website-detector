import argparse
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump
from features import extract_url_features

def build_features(urls: pd.Series) -> pd.DataFrame:
    rows = [extract_url_features(u) for u in urls.tolist()]
    return pd.DataFrame(rows)

def train(data_path: str, model_dir: str = 'model'):
    df = pd.read_csv(data_path)
    if not {'url','label'}.issubset(df.columns):
        raise ValueError("Dataset must have columns: url,label (label: 0=benign, 1=phishing)")
    X = build_features(df['url'])
    y = df['label'].astype(int).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred, digits=4))

    os.makedirs(model_dir, exist_ok=True)
    dump(clf, os.path.join(model_dir, 'model.pkl'))
    dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    print(f"Saved model & scaler to {model_dir}/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to dataset CSV with columns url,label')
    args = parser.parse_args()
    train(args.data)
