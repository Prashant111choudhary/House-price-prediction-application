from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model, scaler, and columns
model = pickle.load(open("ridge_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# Load new location dataset
df = pd.read_csv("location.csv")

# Clean location names
df["Location"] = df["Location"].astype(str).str.strip().str.title()

# Create multiplier dictionary (default = 1.0 for all locations)
location_map = {loc: 1.0 for loc in df["Location"].unique()}

@app.route("/")
def index():
    return render_template("index.html", locations=sorted(location_map.keys()))

@app.route("/predict", methods=["POST"])
def predict():
    try:
        form_data = request.form

        def yesno(name):
            return 1 if form_data.get(name) == "yes" else 0

        mainroad_val = 1 if form_data.get("mainroad", "no") == "yes" else 0
        furnishing = form_data.get("furnishingstatus", "furnished")
        location = form_data.get("location")

        input_dict = {
            "area": float(form_data.get("area", 0)),
            "bedrooms": int(form_data.get("bedrooms", 0)),
            "bathrooms": int(form_data.get("bathrooms", 0)),
            "stories": int(form_data.get("stories", 0)),
            "mainroad": mainroad_val,
            "guestroom": yesno("guestroom"),
            "basement": yesno("basement"),
            "hotwaterheating": yesno("hotwaterheating"),
            "airconditioning": yesno("airconditioning"),
            "parking": int(form_data.get("parking", 0)),
            "prefarea": yesno("prefarea"),
            "furnishingstatus_semi-furnished": 1 if furnishing == "semi-furnished" else 0,
            "furnishingstatus_unfurnished": 1 if furnishing == "unfurnished" else 0,
        }

        # Match training columns
        input_df = pd.DataFrame([input_dict]).reindex(columns=columns, fill_value=0)
        scaled_input = scaler.transform(input_df)
        base_price = float(model.predict(scaled_input)[0])

        # Get multiplier (currently 1.0 for all)
        multiplier = location_map.get(location, 1.0)
        adjusted_price = base_price * multiplier

        pretty_inputs = {
            "Area (sq ft)": form_data.get("area", ""),
            "Bedrooms": form_data.get("bedrooms", ""),
            "Bathrooms": form_data.get("bathrooms", ""),
            "Stories": form_data.get("stories", ""),
            "Main Road": "Yes" if mainroad_val else "No",
            "Guest Room": "Yes" if input_dict["guestroom"] else "No",
            "Basement": "Yes" if input_dict["basement"] else "No",
            "Hot Water Heating": "Yes" if input_dict["hotwaterheating"] else "No",
            "Air Conditioning": "Yes" if input_dict["airconditioning"] else "No",
            "Parking (spots)": form_data.get("parking", ""),
            "Preferred Area": "Yes" if input_dict["prefarea"] else "No",
            "Furnishing": furnishing.replace("-", " ").title(),
            "Location": location,
        }

        return render_template(
            "result.html",
            price=round(adjusted_price, 2),
            base_price=round(base_price, 2),
            multiplier=round(multiplier, 2),
            details=pretty_inputs,
        )

    except Exception as e:
        return render_template("result.html", error=str(e), details={})

if __name__ == "__main__":
    app.run(debug=True)
