from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import random
import joblib
import os
import time

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main():

    start_time = time.perf_counter()

    if request.method == "POST":
        Pclass = request.form.get("Pclass")
        Sex = request.form.get("sex")
        Age = request.form.get("age")
        SibSp = request.form.get("SibSp")
        Parch = request.form.get("Parch")
        Fare = request.form.get("Fare")
        Embarked = request.form.get("Embarked")

        model_path="model.pkl"

        input_data = {
            "PassengerId": random.randint(900, 15000),
            "Pclass": [int(Pclass)],
            "Sex": [Sex],
            "Age": [float(Age)],
            "SibSp": [int(SibSp)],
            "Parch": [int(Parch)],
            "Fare": [float(Fare)],
            "Embarked": [Embarked]
        }
        input_df = pd.DataFrame(input_data)
        input_df["Sex"] = input_df["Sex"].map({"male": 0, "female": 1})
        input_df["Embarked"] = input_df["Embarked"].map({"C": 0, "S": 1, "Q": 2})

        if os.path.exists(model_path):
            model=joblib.load(model_path)
            output_pred = model.predict(input_df)
            if(output_pred==0):
                return render_template("result.html", prediction="Died")
            else:
                return render_template("result.html", prediction="Survived")
        else:
            train_data = pd.read_csv('train.csv')
            X_train = train_data.drop(columns=["Survived", "Name", "Cabin", "Ticket"])
            Y_train = train_data["Survived"]
            X_train["Sex"] = X_train["Sex"].map({"male": 0, "female": 1})
            X_train["Embarked"] = X_train["Embarked"].map({"C": 0, "S": 1, "Q": 2})
            model = DecisionTreeClassifier()
            model.fit(X_train, Y_train)
            output_pred = model.predict(input_df)
            joblib.dump(model, model_path)
            if(output_pred==0):
                return render_template("result.html", prediction="Died")
            else:
                return render_template("result.html", prediction="Survived")
    end_time = time.perf_counter()
    time_taken = (end_time - start_time) * 1000
    print(f"Time taken for prediction: {time_taken:.15f} milliseconds")

    return render_template("titanic.html")

if __name__ == '__main__':
    app.secret_key = '12345789'
    app.run(debug=True)