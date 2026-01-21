from flask import Flask, render_template, request
import joblib
import numpy as np

# Create the Flask app
app = Flask(__name__)

# Load the model and encoder we created in Part A
model = joblib.load('model/house_price_model.pkl')
label_encoder = joblib.load('model/label_encoder.pkl')

# Home page
@app.route('/')
def home():
    # Get all neighborhood names for the dropdown menu
    neighborhoods = label_encoder.classes_.tolist()
    return render_template('index.html', neighborhoods=neighborhoods)

# Prediction page
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the values the user entered in the form
        overall_qual = float(request.form['overall_qual'])
        gr_liv_area = float(request.form['gr_liv_area'])
        total_bsmt_sf = float(request.form['total_bsmt_sf'])
        garage_cars = float(request.form['garage_cars'])
        year_built = float(request.form['year_built'])
        neighborhood = request.form['neighborhood']
        
        # Convert neighborhood name to a number
        neighborhood_encoded = label_encoder.transform([neighborhood])[0]
        
        # Put all features together in the right order
        features = np.array([[overall_qual, gr_liv_area, total_bsmt_sf, 
                             garage_cars, year_built, neighborhood_encoded]])
        
        # Use the model to predict the price
        prediction = model.predict(features)[0]
        
        # Show the result
        neighborhoods = label_encoder.classes_.tolist()
        return render_template('index.html', 
                             prediction=f'${prediction:,.2f}',
                             neighborhoods=neighborhoods)
    except Exception as e:
        # If something goes wrong, show an error
        neighborhoods = label_encoder.classes_.tolist()
        return render_template('index.html', 
                             prediction=f'Error: {str(e)}',
                             neighborhoods=neighborhoods)

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)