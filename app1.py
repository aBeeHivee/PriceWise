from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import math

app = Flask(__name__)

# Load the trained machine learning model
with open("LinearRegression.pkl", "rb") as f:
    model1 = pickle.load(f)

with open("laptop_price_mod.pkl", "rb") as g:
    model2 = pickle.load(g)

model3 = pickle.load(open("Mobile_price_model.pkl", "rb"))

# Load the DataFrame containing the car data
car = pd.read_csv("Final_dataset.csv")
laptop = pd.read_csv("Final_dataset_laptop.csv")

# Assuming 'car' is the DataFrame containing your dataset
# Replace 'car' with your actual DataFrame if it has a different name
def get_unique_values():
    names = car["name"].unique().tolist()
    companies = car["company"].unique().tolist()
    fuel_types = car["fuel_type"].unique().tolist()

    return {"names": names, "companies": companies, "fuel_types": fuel_types}


def get_unique_values2():
    companies = laptop["Company"].unique().tolist()
    type_names = laptop["TypeName"].unique().tolist()
    cpu_brands = laptop["Cpu brand"].unique().tolist()
    gpu_brands = laptop["Gpu brand"].unique().tolist()

    return {"companies": companies, "type_names": type_names, "cpu_brands": cpu_brands, "gpu_brands": gpu_brands}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/category")
def category():
    return render_template("category.html")

@app.route("/carf")
def carf():
    return render_template("car_form.html")

@app.route("/lapf")
def lapf():
    return render_template("laptop_form.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/home")
def home():
    return render_template("index.html")

@app.route("/phone")
def phone():
    return render_template("phone_form.html")

@app.route("/get_dropdown_options")
def get_dropdown_options():
    options = get_unique_values()
    return jsonify(options)

@app.route("/get_dropdown_options2")
def get_dropdown_options2():
    options = get_unique_values2()
    return jsonify(options)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Convert the input data into a DataFrame
    input_data = pd.DataFrame(data, index=[0])

    # Use the model to make predictions
    predicted_price = model1.predict(input_data)

    return jsonify({"price": round(predicted_price[0])})

@app.route("/pred", methods=["POST"])
def pred():
    data = request.get_json()

    # Convert the input data into a DataFrame
    input_data = pd.DataFrame(data, index=[0])

    # Use the model to make predictions
    predicted_price = model2.predict(input_data)

    return jsonify({"price": round(predicted_price[0])})

@app.route('/pre', methods=['POST'])
def pre():
    data = request.form.to_dict()
    features = np.array(list(data.values())).reshape(1, -1).astype(float)
    prediction = model3.predict(features)
    return jsonify(math.exp(prediction[0]))

if __name__ == "__main__":
    app.run(debug=True)
