from flask import Flask, jsonify
import pandas as pd
import folium
from geopy.distance import geodesic
from sklearn.preprocessing import LabelEncoder
from haversine import haversine, Unit
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


def depthWellAdvanced(user_lat, user_lon, thresh):

    def create_well_recommendation_map(
        user_latitude, user_longitude, threshold_distance=float(thresh)
    ):

        file_path = "dugwell.csv"
        df = pd.read_csv(file_path)

        df = df.dropna(subset=["Y", "X", "Depth (m.bgl)", "Well Type"])

        X = df[["Y", "X"]]
        y_depth = df["Depth (m.bgl)"]
        y_well_type = df["Well Type"]

        depth_model = KNeighborsRegressor(n_neighbors=3)
        depth_model.fit(X, y_depth)

        well_type_model = KNeighborsClassifier(n_neighbors=3)
        well_type_model.fit(X, y_well_type)

        def recommend_well_and_nearest(user_lat, user_lon):
            user_location = [[user_lat, user_lon]]

            nearest_dugwell_index = depth_model.kneighbors(user_location)[1][0][0]
            nearest_dugwell_coordinates = X.iloc[nearest_dugwell_index]
            nearest_dugwell_depth = y_depth.iloc[nearest_dugwell_index]
            nearest_dugwell_well_type = y_well_type.iloc[nearest_dugwell_index]

            distance_to_nearest_dugwell = geodesic(
                user_location[0], nearest_dugwell_coordinates
            ).kilometers
            if distance_to_nearest_dugwell > threshold_distance:
                return f"No suitable well within {threshold_distance} km.", None, None

            return (
                nearest_dugwell_depth,
                nearest_dugwell_well_type,
                nearest_dugwell_coordinates,
            )

        map_center = [user_latitude, user_longitude]
        map_object = folium.Map(location=map_center, zoom_start=12)

        recommendation_result = recommend_well_and_nearest(
            user_latitude, user_longitude
        )
        recommended_depth, recommended_well_type, recommended_coordinates = (
            recommendation_result
        )

        if (
            isinstance(recommended_well_type, str)
            and recommended_well_type != "No suitable well within 3 km."
        ):

            folium.Marker(
                location=[user_latitude, user_longitude],
                popup=f"Recommended Depth: {recommended_depth} meters, Recommended Well Type: {recommended_well_type}",
                icon=folium.Icon(color="red"),
            ).add_to(map_object)

            folium.Marker(
                location=[recommended_coordinates["Y"], recommended_coordinates["X"]],
                popup=f"Nearest Well - Depth: {recommended_depth} meters, Well Type: {recommended_well_type}",
                icon=folium.Icon(color="green"),
            ).add_to(map_object)
        else:

            folium.Marker(
                location=[user_latitude, user_longitude],
                popup="No suitable well within 3 km.",
                icon=folium.Icon(color="gray"),
            ).add_to(map_object)

        return map_object, recommended_depth, recommended_well_type

    user_latitude = user_lat
    user_longitude = user_lon

    map_object, recommended_depth, recommended_well_type = (
        create_well_recommendation_map(user_latitude, user_longitude)
    )

    print(f"Recommended Depth: {recommended_depth} meters")
    print(f"Recommended Well Type: {recommended_well_type}")

    map_object.save("well_recommendation_map.html")

    with open("well_recommendation_map.html", "r") as file:

        html_content = file.read()

    result = {
        "depth": f"{recommended_depth}",
        "well_type": f"{recommended_well_type}",
        "html_content": f"{html_content}",
    }

    return result


def drillingTechnic(user_lat, user_lon):

    file_path = "Aquifer_data_Cuddalore.xlsx"
    aquifer_df = pd.read_excel(file_path)

    aquifer_df = aquifer_df.dropna(subset=["FORMATION", "Y_IN_DEC", "X_IN_DEC"])

    label_encoder = LabelEncoder()
    aquifer_df["FORMATION"] = label_encoder.fit_transform(aquifer_df["FORMATION"])

    X = aquifer_df[["Y_IN_DEC", "X_IN_DEC"]]
    y = aquifer_df["FORMATION"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    knn_classifier.fit(X_train, y_train)

    def predict_formation(user_latitude, user_longitude):
        user_location = [[user_latitude, user_longitude]]

        nearest_aquifer_index = knn_classifier.kneighbors(user_location)[1][0][0]
        nearest_aquifer_formation = label_encoder.inverse_transform(
            [y.iloc[nearest_aquifer_index]]
        )[0]

        return nearest_aquifer_formation

    def suggest_drilling_technique(formation):
        if formation == "SR":
            return "SoftRock formation suggests Rotary drilling technique."
        elif formation == "HR":
            return "HardRock formation suggests Down the hole drilling technique."

    user_latitude = user_lat
    user_longitude = user_lon

    predicted_formation = predict_formation(user_latitude, user_longitude)

    print(
        f"The predicted 'Aquifer type for the given coordinates is: {predicted_formation}"
    )

    if predicted_formation == "SR":
        print(suggest_drilling_technique(predicted_formation))
        pf = suggest_drilling_technique(predicted_formation)
    elif predicted_formation == "HR":
        print(suggest_drilling_technique(predicted_formation))
        pf = suggest_drilling_technique(predicted_formation)
    else:
        pf = "Not found"
        print("Not found")

    result = {"formation": f"{predicted_formation}", "drilling_technic": f"{pf}"}

    return result


def waterQualityChloride(user_lat, user_lon, thresh):

    csv_file_path = "Modified_Water_Quality_Shuffled.csv"
    df_water_quality = pd.read_csv(csv_file_path)

    user_latitude = user_lat
    user_longitude = user_lon
    test_location = (user_latitude, user_longitude)

    threshold_distance_km = float(thresh)

    distances = []
    for index, row in df_water_quality.iterrows():
        well_location = (row["Latitude"], row["Longitude"])
        distance = haversine(test_location, well_location, unit=Unit.KILOMETERS)
        distances.append(distance)

    nearby_wells_indices = np.where(np.array(distances) <= threshold_distance_km)[0]

    if len(nearby_wells_indices) > 0:

        mean_chloride = df_water_quality.iloc[nearby_wells_indices][
            "Chloride_mg_per_l"
        ].mean()
        print(f"The predicted chloride level is: {mean_chloride:.2f} mg/l")
        chloride_level = f"{mean_chloride:.2f} mg/l"
    else:
        print(f"Not able to predict the chloride level")
        chloride_level = f"NaN"

    result = {"chloride_level": f"{chloride_level}"}

    return result


def waterQualityChlorideAll(user_late, user_lon, thresh):

    file_path = "UpdatedWaterQuality.csv"
    df_water_quality = pd.read_csv(file_path)

    user_lat = user_late
    user_long = user_lon

    user_location = (user_lat, user_long)

    columns_of_interest = ["EC_1", "F_1", "EC_2", "F_2", "EC_3", "F_3", "EC_4", "F_4"]

    threshold_distance_km = float(thresh)

    nearby_values = {}
    tds_values = {}
    means = {}

    for column in columns_of_interest:
        nearby_column_values = []

        for index, row in df_water_quality.iterrows():
            well_location = (row["Y"], row["X"])
            distance = haversine(user_location, well_location, unit=Unit.KILOMETERS)

            if distance <= threshold_distance_km:
                nearby_column_values.append(row[column])

        mean_value = np.mean(nearby_column_values)
        print(f"Mean {column} value of nearby wells: {mean_value:.2f}")
        means.update({f"{column}": f"{mean_value:.2f}"})

        nearby_values[column] = nearby_column_values

        if column.startswith("EC"):
            aquifer_number = int(column.split("_")[1])
            tds_column = f"TDS_{aquifer_number}"
            tds_values[tds_column] = mean_value * 0.67

    tds = {}
    print("\nTDS values for Aquifers 1, 2, 3, and 4:")
    for aquifer_number in range(1, 5):
        tds_column = f"TDS_{aquifer_number}"
        print(
            f"TDS value for Aquifer {aquifer_number}: {tds_values.get(tds_column, 0):.2f}"
        )
        tds.update(
            {
                f"TDS_Value_of_Aquifer_{aquifer_number}": f"{tds_values.get(tds_column, 0):.2f}"
            }
        )

    result = {"means": means, "tds": tds}

    return result


def depthOfWaterBearing(user_lat, user_lon):

    def find_three_nearest_aquifers_and_average_depth(
        file_path,
        formation_column,
        top_column,
        bottom_column,
        user_latitude,
        user_longitude,
    ):

        aquifer_df = pd.read_excel(file_path)

        aquifer_df = aquifer_df.dropna(
            subset=["FORMATION", "Y_IN_DEC", "X_IN_DEC", top_column, bottom_column]
        )

        label_encoder = LabelEncoder()
        aquifer_df["FORMATION"] = label_encoder.fit_transform(aquifer_df["FORMATION"])

        X = aquifer_df[["Y_IN_DEC", "X_IN_DEC"]]
        y = aquifer_df["FORMATION"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        knn_classifier = KNeighborsClassifier(n_neighbors=3)
        knn_classifier.fit(X_train, y_train)

        def find_three_nearest_aquifers():
            user_location = [[user_latitude, user_longitude]]

            nearest_aquifer_indices = knn_classifier.kneighbors(
                user_location, n_neighbors=3
            )[1][0]
            nearest_aquifer_data = aquifer_df.iloc[nearest_aquifer_indices]

            return nearest_aquifer_data

        def calculate_average_aquifer_depth(nearest_aquifer_data, column_name):
            average_depth = nearest_aquifer_data[column_name].mean()
            return average_depth

        nearest_aquifer_data = find_three_nearest_aquifers()

        average_top_depth = calculate_average_aquifer_depth(
            nearest_aquifer_data, top_column
        )
        average_bottom_depth = calculate_average_aquifer_depth(
            nearest_aquifer_data, bottom_column
        )

        print(
            f"{formation_column} water bearing zone is: {average_top_depth} meters - {average_bottom_depth} meters"
        )

        return f"{average_top_depth:.2f},{average_bottom_depth:.2f}"

    user_latitude = user_lat
    user_longitude = user_lon

    result = {
        "first": find_three_nearest_aquifers_and_average_depth(
            "Aquifer_data_Cuddalore.xlsx",
            "First",
            "Aq_I_top_Rl (m.amsl)",
            "Aq_I_Bottom_RL (m.amsl)",
            user_latitude,
            user_longitude,
        ),
        "second": find_three_nearest_aquifers_and_average_depth(
            "Aquifer_data_Cuddalore.xlsx",
            "Second",
            "Aq_II_top_Rl (m.amsl)",
            "Aq_II_Bottom_RL (m.amsl)",
            user_latitude,
            user_longitude,
        ),
        "third": find_three_nearest_aquifers_and_average_depth(
            "Aquifer_data_Cuddalore.xlsx",
            "Third",
            "Aq_III_top_Rl (m.amsl)",
            "Aq_III_Bottom_RL (m.amsl)",
            user_latitude,
            user_longitude,
        ),
        "forth": find_three_nearest_aquifers_and_average_depth(
            "Aquifer_data_Cuddalore.xlsx",
            "Fourth",
            "Aq_IV_top_Rl  (m.amsl)",
            "Aq_IV_top_Rl  (m.amsl)",
            user_latitude,
            user_longitude,
        ),
    }

    return result


def waterDischarge(user_lat, user_lon):
    ls = []

    def train_and_predict(file_path, target_columns, user_latitude, user_longitude):

        df = pd.read_excel(file_path)

        predictions = {}

        for target_column in target_columns:

            selected_columns = ["Y_IN_DEC", "X_IN_DEC", target_column]
            df_selected = df[selected_columns]

            df_cleaned = df_selected.dropna(subset=[target_column])

            X = df_cleaned[["Y_IN_DEC", "X_IN_DEC"]]
            y = df_cleaned[target_column]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("regressor", LinearRegression()),
                ]
            )

            model.fit(X_train, y_train)

            user_data = pd.DataFrame(
                [[user_latitude, user_longitude]], columns=["Y_IN_DEC", "X_IN_DEC"]
            )

            prediction = model.predict(user_data)

            print(f"Predicted {target_column}:", prediction[0])

            predictions[target_column] = prediction[0]

            ls.append(f"{prediction[0]:.2f}")

        return predictions

    file_path = "Transmisivitty.xlsx"

    user_latitude = user_lat
    user_longitude = user_lon

    target_columns = [
        "aq1_yield (lps)",
        "aq2_yield (lps)",
        "AQ3_yield (lps)",
        "AQ4_yield (lps)",
    ]

    result = {"ans": ls}

    all_predictions = train_and_predict(
        file_path, target_columns, user_latitude, user_longitude
    )

    return result


@app.route("/analyze_location/<float:user_lat>/<float:user_lon>/<int:thresh>")
def analyze_location(user_lat, user_lon, thresh):

    depth_well_result = depthWellAdvanced(user_lat, user_lon, thresh)
    drilling_technic_result = drillingTechnic(user_lat, user_lon)
    water_quality_chloride_result = waterQualityChloride(user_lat, user_lon, thresh)
    water_quality_all_result = waterQualityChlorideAll(user_lat, user_lon, thresh)
    depth_water_bearing_result = depthOfWaterBearing(user_lat, user_lon)
    water_discharge_result = waterDischarge(user_lat, user_lon)

    final_result = {
        "depth_well_result": depth_well_result,
        "drilling_technic_result": drilling_technic_result,
        "water_quality_chloride_result": water_quality_chloride_result,
        "water_quality_all_result": water_quality_all_result,
        "depth_water_bearing_result": depth_water_bearing_result,
        "water_discharge_result": water_discharge_result,
    }

    return jsonify(final_result)


if __name__ == "__main__":
    app.run(debug=True, port=8080)
