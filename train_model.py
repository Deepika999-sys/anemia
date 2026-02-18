import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def train_and_save(csv_path='anemia.csv', model_dir='models'):
    os.makedirs(model_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    # drop duplicates if any
    df = df.drop_duplicates()
    if 'Result' not in df.columns:
        raise RuntimeError('CSV missing Result column')
    X = df.drop('Result', axis=1)
    y = df['Result']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train)

    # Evaluate
    from sklearn.metrics import accuracy_score
    y_pred = lr.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model accuracy on test set: {accuracy:.4f}')

    # Save scaler and model
    with open(os.path.join(model_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(model_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(lr, f)

    return os.path.join(model_dir, 'model.pkl'), os.path.join(model_dir, 'scaler.pkl')

if __name__ == '__main__':
    print('Training model...')
    model_path, scaler_path = train_and_save()
    print('Saved model to', model_path)
