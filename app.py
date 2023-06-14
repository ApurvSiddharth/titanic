from flask import Flask, request, jsonify
import pickle
import pandas as pd
import json

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    model = pickle.load(open('my_rfmodel.pickle', 'rb'))
    train_data = pd.read_csv('./train.csv')
    train_data['data_source'] = 'train'
    data_req = request.json  # Get the JSON data from the request
    if(type(data_req) != dict):
        data_req = json.loads(data_req)
    
    # Process the input data and make predictions using the loaded model
    # Return the predictions as a JSON response
    data = pd.DataFrame.from_dict(data_req, orient='index') 
    data = data.transpose()
    data['data_source'] = 'input'
    numerical_columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for column in numerical_columns:
        data[column] = data[column].astype(float)
        data[column].fillna(train_data[column].median(), inplace=True)

    

    categorical_columns = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination','VIP']
    for column in categorical_columns:
        data[column].fillna(train_data[column].mode().iloc[0], inplace=True)

    data['CryoSleep'] = data['CryoSleep'].map({'True':1,'False':0})
    data['VIP'] = data['VIP'].map({'True':1,'False':0})

    temp_df = train_data.append(data)

    categorical_columns = ['HomePlanet', 'Destination']
    df_encoded_test = pd.get_dummies(temp_df, columns=categorical_columns)
    df_encoded_test = df_encoded_test[df_encoded_test['data_source'] == 'input']
    
    print(df_encoded_test)

    pred = model.predict(df_encoded_test[[x for x in df_encoded_test.columns if x not in ['Transported', 'Cabin','PassengerId','Name','data_source']]])
    sub = pd.DataFrame()
    sub['PassengerId'] = data['PassengerId']
    sub['Transported'] = pred[0]
    sub['Transported'] = sub['Transported'].map({1:True,0:False})
    print(pred)
    print(sub)
    sub = sub.to_dict(orient='records')
    print(sub)
    return jsonify(sub)

if __name__ == '__main__':
    app.run()