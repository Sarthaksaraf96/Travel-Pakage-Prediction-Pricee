import pickle
import numpy as np
import pandas as pd
from flask_ngrok import run_with_ngrok
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA



# Specify the path to the saved model file
model_file_path = 'Model\decision_tree_reg_latest.pkl'

# Load the model from the file
with open(model_file_path, 'rb') as model_file:
    loaded_decision_tree_reg = pickle.load(model_file)


# Define the PCA and StandardScaler objects
n_components = 23  # Adjust the number of components as needed
pca = PCA(n_components=n_components)
scaler = StandardScaler()

# Load the Sentence Transformer model
model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v4_MiniLM-L6') 


# Create a function for prediction
def predict_price(input_data, lr_model, pca, scaler):
    # Prepare the input data
    text_columns = [
        'Package Name', 'Destination', 'Itinerary', 'Places Covered',
        'Hotel Details', 'Airline', 'Sightseeing Places Covered', 'Cancellation Rules'
    ]

    # Initialize an empty DataFrame
    df = pd.DataFrame([input_data])

    # Encode text-based columns and create embeddings
    for column in text_columns:
        df[column + '_embedding'] = df[column].apply(lambda text: model.encode(text))

    # Apply PCA separately to each text embedding column
    n_components = 23  # Adjust the number of components as needed
    text_embeddings_pca = np.empty((len(df), n_components * len(text_columns)))

    for i, column in enumerate(text_columns):
        embeddings = df[column + '_embedding'].values.tolist()
        embeddings_pca = pca.transform(embeddings)
        text_embeddings_pca[:, i * n_components:(i + 1) * n_components] = embeddings_pca

    # Combine text embeddings with other numerical features if available
    numerical_features = [
        'Package Type_Standard', 'Package Type_Premium', 'Package Type_Luxury',
        'Travel_Month', 'Package Type_Budget', 'Package Type_Deluxe',
        'Hotel Ratings', 'Start City_New Delhi', 'Start City_Mumbai',
        'Travel_DayOfWeek', 'Travel_Year'
    ]

    X_numerical = df[numerical_features].values

    # Combine PCA-transformed text embeddings and numerical features
    X = np.hstack((text_embeddings_pca, X_numerical))

    # Scale the data using the same scaler used during training
    X = scaler.transform(X)

    # Make predictions using the trained Linear Regression model
    y_pred = lr_model.predict(X)

    return y_pred[0]

df_input = pd.read_csv('Data\marketing_sample_for_makemytrip_com-travel__20190901_20190930__30k_data.csv',on_bad_lines='skip')


# Filling missing values for Hotel Details with 'Not Available'
df_input['Hotel Details'].fillna('Not Available', inplace=True)

# Filling missing values for Airline with 'Not Available'
df_input['Airline'].fillna('Not Available', inplace=True)

# Filling missing values for Onwards Return Flight Time with 'Not Available'
df_input['Onwards Return Flight Time'].fillna('Not Available', inplace=True)

# Filling missing values for Sightseeing Places Covered with 'Not Available'
df_input['Sightseeing Places Covered'].fillna('Not Available', inplace=True)

# Filling missing values for Initial Payment For Booking with 0 (assuming no initial payment)
df_input['Initial Payment For Booking'].fillna(0, inplace=True)

# Filling missing values for Cancellation Rules with 'Not Available'
df_input['Cancellation Rules'].fillna('Not Available', inplace=True)

# Dropping columns with all missing values (Flight Stops, Date Change Rules, Unnamed: 22, Unnamed: 23)
df_input.drop(columns=["Flight Stops", "Meals", "Initial Payment For Booking", "Date Change Rules"], inplace=True)
df_input['Travel Date'] = pd.to_datetime(df_input['Travel Date'], format='%d-%m-%Y', errors='coerce')
allowed_package_types = ['Deluxe', 'Standard', 'Premium', 'Luxury', 'Budget']

# Filter the DataFrame to keep only the rows with allowed package types
df_input = df_input[df_input['Package Type'].isin(allowed_package_types)]
df_input.drop('Company', axis=1, inplace=True)
df_input.drop('Crawl Timestamp', axis=1, inplace=True)

# Extracting hotel ratings from Hotel Details
df_input['Hotel Ratings'] = df_input['Hotel Details'].str.extract(r'(\d+\.\d+)')

# Convert 'Hotel Ratings' to numeric (float)
df_input['Hotel Ratings'] = pd.to_numeric(df_input['Hotel Ratings'], errors='coerce')

# Calculate the mode of the 'Hotel Ratings' column
mode_rating = df_input['Hotel Ratings'].mode()[0]

# Replace NaN values with the mode
df_input['Hotel Ratings'].fillna(mode_rating, inplace=True)

# Assuming your data is stored in a DataFrame called 'df_input'
df_input['Travel Date'] = pd.to_datetime(df_input['Travel Date'])
df_input['Travel_Year'] = df_input['Travel Date'].dt.year
df_input['Travel_Month'] = df_input['Travel Date'].dt.month
df_input['Travel_DayOfWeek'] = df_input['Travel Date'].dt.dayofweek

# Example of one-hot encoding
df_input = pd.get_dummies(df_input, columns=['Package Type','Start City'])

Q1 = df_input['Per Person Price'].quantile(0.25)
Q3 = df_input['Per Person Price'].quantile(0.75)
IQR = Q3 - Q1

# Step 2: Define lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Step 3: Filter the DataFrame to remove outliers
df_input = df_input[(df_input['Per Person Price'] >= lower_bound) & (df_input['Per Person Price'] <= upper_bound)]


row_index = 100  # Replace with the desired row index

# Access the row from df_input using iloc
input_row = df_input.iloc[row_index]

# Create an input dictionary from the selected row
input_data = {
    'Package Name': input_row['Package Name'],
    'Destination': input_row['Destination'],
    'Itinerary': input_row['Itinerary'],
    'Places Covered': input_row['Places Covered'],
    'Hotel Details': input_row['Hotel Details'],
    'Airline': input_row['Airline'],
    'Sightseeing Places Covered': input_row['Sightseeing Places Covered'],
    'Cancellation Rules': input_row['Cancellation Rules'],
    'Package Type_Standard': input_row['Package Type_Standard'],
    'Package Type_Premium': input_row['Package Type_Premium'],
    'Package Type_Luxury': input_row['Package Type_Luxury'],
    'Travel_Month': input_row['Travel_Month'],
    'Package Type_Budget': input_row['Package Type_Budget'],
    'Package Type_Deluxe': input_row['Package Type_Deluxe'],
    'Hotel Ratings': input_row['Hotel Ratings'],
    'Start City_New Delhi': input_row['Start City_New Delhi'],
    'Start City_Mumbai': input_row['Start City_Mumbai'],
    'Travel_DayOfWeek': input_row['Travel_DayOfWeek'],
    'Travel_Year': input_row['Travel_Year']
}
import joblib
# Load the models from the files
loaded_lr_model = joblib.load('Preprocessing_models\lr_model.joblib')
loaded_pca = joblib.load('Preprocessing_models\pca.joblib')
loaded_scaler = joblib.load('Preprocessing_models\scaler.joblib')

# predicted_price = str(predict_price(input_data, loaded_lr_model, loaded_pca, loaded_scaler))[-10:]
# print(f'Predicted Per Person Price: ${predicted_price}')
 

app = Flask(__name__ , template_folder='.')
run_with_ngrok(app)

@app.route('/', methods=['GET', 'POST'])
def predict():
    return render_template('template.html')



@app.route('/predict', methods=['POST'])
def index():
    if request.method == 'POST':
        # Get input data from the form
        package_name = request.form.get('Package Name')
        destination = request.form.get('Destination')
        itinerary = request.form.get('Itinerary')
        places_covered = request.form.get('Places Covered')
        hotel_details = request.form.get('Hotel Details')
        airline = request.form.get('Airline')
        sightseeing_places = request.form.get('Sightseeing Places Covered')
        cancellation_rules = request.form.get('Cancellation Rules')
        package_standard = int(request.form.get('Package Type_Standard', 0))
        package_premium = int(request.form.get('Package Type_Premium', 0))
        package_luxury = int(request.form.get('Package Type_Luxury', 0))
        travel_month = int(request.form.get('Travel_Month'))
        package_budget = int(request.form.get('Package Type_Budget', 0))
        package_deluxe = int(request.form.get('Package Type_Deluxe', 0))
        hotel_ratings = float(request.form.get('Hotel Ratings'))
        start_city_delhi = int(request.form.get('Start City_New Delhi', 0))
        start_city_mumbai = int(request.form.get('Start City_Mumbai', 0))
        travel_day_of_week = int(request.form.get('Travel_DayOfWeek'))
        travel_year = int(request.form.get('Travel_Year'))

        # Create a dictionary to store the input data
        data = {
            'Package Name': package_name,
            'Destination': destination,
            'Itinerary': itinerary,
            'Places Covered': places_covered,
            'Hotel Details': hotel_details,
            'Airline': airline,
            'Sightseeing Places Covered': sightseeing_places,
            'Cancellation Rules': cancellation_rules,
            'Package Type_Standard': package_standard,
            'Package Type_Premium': package_premium,
            'Package Type_Luxury': package_luxury,
            'Travel_Month': travel_month,
            'Package Type_Budget': package_budget,
            'Package Type_Deluxe': package_deluxe,
            'Hotel Ratings': hotel_ratings,
            'Start City_New Delhi': start_city_delhi,
            'Start City_Mumbai': start_city_mumbai,
            'Travel_DayOfWeek': travel_day_of_week,
            'Travel_Year': travel_year
        }

        # Perform prediction using the custom_input dictionary
        prediction = predict_price(data, loaded_lr_model, loaded_pca, loaded_scaler)//10
        prediction = str(prediction)[-7:]

        return jsonify({'prediction': prediction})


if __name__ == "__main__":
    app.run()