import numpy as np
import boto3
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
import sqlite3


def fetch_from_dynamodb():
    print("Fetching dataset from dynamodb")
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('car_table')
    response = table.scan()
    data = response['Items']

    while 'LastEvaluatedKey' in response:
        response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
        data.extend(response['Items'])

    return data


def format_dataset(data):
    print("Formatting dataset.")
    raw_dataset = np.array([list(d.values())[1:] for d in data])
    enc = OrdinalEncoder()
    enc.fit(raw_dataset)
    dataset = enc.transform(raw_dataset)
    attributes = list(data[0].keys())[1:]
    features = attributes[:5] + attributes[6:]
    x = dataset[:, [0, 1, 2, 3, 4, 6]]
    y = dataset[:, 5]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    return [features, x_train, x_test, y_train, y_test]


def fit_model(features, x_train, x_test, y_train, y_test):
    print("Fitting model.")
    clf = tree.DecisionTreeClassifier(criterion="entropy")
    clf = clf.fit(x_train, y_train)
    model_score = clf.score(x_test, y_test)
    print("Model score is: " + str(model_score))

    # Export tree
    print("Exporting model visualization.")
    tree.export_graphviz(clf, out_file="tree.dot",
                         feature_names=features,
                         class_names='class',
                         filled=True, rounded=True,
                         special_characters=True)

    return clf


def random_sample(x, features):
    return [np.random.choice(np.unique(x[:, i])) for i in range(len(features))]


def insert_statement(prediction):
    prefix = "INSERT INTO Predictions VALUES ("
    pred_list = prediction.tolist()
    pred_str = ",".join(map(str, pred_list))
    post_fix = ")"
    return prefix + pred_str + post_fix


def predict_random(clf, x_train, features):
    # Generate random samples and classify them.
    print("Generating random predictions.")
    synthetic_data = np.array([random_sample(x_train, features) for i in range(100)], dtype=np.uint)
    y_predictions = clf.predict(synthetic_data)
    predictions = np.append(synthetic_data, np.array([y_predictions], dtype=np.uint).T, axis=1)

    # Store results in SQLite db.
    print("Saving results into SQLite database.")
    con = sqlite3.connect("predictions.db")
    cur = con.cursor()
    cur.execute('''CREATE TABLE Predictions
        (buying INT,maint INT,safety INT,lug_boot INT,persons INT,doors INT,class INT)''')

    for i in range(len(predictions)):
        cur.execute(insert_statement(predictions[i]))

    con.commit()
    con.close()


def demo_routine():
    dynamodb_dataset = fetch_from_dynamodb()
    [features, x_train, x_test, y_train, y_test] = format_dataset(dynamodb_dataset)
    model = fit_model(features, x_train, x_test, y_train, y_test)

    predict_random(model, x_train, features)
