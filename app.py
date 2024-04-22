from flask import Flask, request, jsonify, render_template
import asyncio
from test_pre_url import create_dictionary_words, process_urls_chunk
from test_pre_content import create_dictionary_words, process_contents_chunk, enrich_content_chunk
import requests
from requests.exceptions import Timeout
from bs4 import BeautifulSoup
import csv
import joblib
import ray
import pandas as pd
from more_itertools import chunked, flatten
from parallel_compute import execute_with_ray
import os
import numpy as np


from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)



dictionary_words = create_dictionary_words()

# ไฟล์ Model ที่ใช้ในการตรวจสอบ URL และ Content
random_for_url = joblib.load("R_URL.joblib")
randon_for_tag = joblib.load("random60000.pkl")


# route server HTML Page
@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/display_csv')
def display_csv():
    file_path = 'predictions.csv' #ไฟล์ที่ใช้แสดงผลในหน้าเว็บ View Details

    if os.path.exists(file_path):
        df = pd.read_csv(file_path, encoding='utf-8')

        # column ที่แสดงผลในหน้าเว็บ View Details
        selected_columns = ['tag', 'predictions'] 

        if all(col in df.columns for col in selected_columns):
            selected_data = df[selected_columns]

            
            columns = selected_data.columns.tolist()
            data = selected_data.to_dict(orient='list')

            return render_template('display_csv.html', columns=columns, data=data)
        else:
            return render_template('invalid_columns_template.html') 
    else:
        return render_template('default_template.html') 



@app.route('/process_urls', methods=['POST'])
def process_urls():
    data = request.get_json()
    urls = data.get("urls", []) 

    urlt = urls

    urlt = str(urlt[0])
    
    try:
        os.remove("urls_results.csv")
        os.remove("tags.csv")
        os.remove("testPre2.csv")
        os.remove("predictions.csv")
        print("Old CSV file deleted")
    except FileNotFoundError:
        print("No existing CSV file found")

    print("Received URLs:", urls)

    # ส่ง URL ไป Preprocess 
    results = asyncio.run(process_urls_chunk(urls, dictionary_words))
    

    enriched_results = []
    enriched_results_content = []

    for url, result in zip(urls, results):
        result_list = result[0].split(',')
        result_numeric_list = [float(value) for value in result_list]
        

        # ให้โมเดลทำนายผล URL     
        predictions = random_for_url.predict([result_numeric_list])

        
        enriched_result = {"url": url, "url_result": results, "predictions": predictions.tolist()}
        enriched_results.append(enriched_result)

        
    # บันทึกผลลัพธ์ลงไฟล์ CSV
    csv_filename = "urls_results.csv"
    with open(csv_filename, "w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["URL", "URL Result", "Predictions"])
        for enriched_result in enriched_results:
            csv_writer.writerow([enriched_result["url"], enriched_result["url_result"], enriched_result["predictions"]])
            
            
    

    # ดึง content จากเว็บไซต์
    try:
        response = requests.get(urlt,timeout=10)

        if response.status_code == 200:
            html_content = response.content

            # แยก content HTML ด้วย BeautifulSoup
            soup = BeautifulSoup(html_content, "html.parser")

            # หา HTML และ JS Tag
            html_tags = soup.find_all()
            javascript_tags = soup.find_all("script")

            
            csv_file = "tags.csv"

            
            with open(csv_file, "w", newline="", encoding="utf-8") as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(["Tag"])

                for tag in html_tags:
                    tag_content = tag.prettify()[:20000]  
                    csv_writer.writerow([tag_content])

                for tag in javascript_tags:
                    tag_content = tag.prettify()[:20000]
                    csv_writer.writerow([tag_content])
                    

            print("Data collected and saved to", csv_file)
            print("Predic", predictions)
            results_content = asyncio.run(process_contents_chunk(urls, dictionary_words)) #ส่ง Content ไปทำการ Preprocess
            

            rows = (list(row) for row in pd.read_csv("tags.csv", encoding="ISO-8859-1").itertuples(index=False))
            row_chunks = [(chunk,) for chunk in chunked(rows, 100)]

            ray.shutdown()
            ray.init(include_dashboard=False)

            # ใช้ Ray
            data = [["tag", "total_chars", "num_spaces", "num_parenthesis", "num_slash", "num_plus", "num_point", "num_comma", "num_semicolon", "num_alpha", "num_numeric", "ratio_spaces", "ratio_alpha", "ratio_numeric", "ratio_parenthesis", "ratio_slash", "ratio_plus", "ratio_point", "ratio_comma", "ratio_semicolon", "ent", "word_count"]] + list(
                flatten(execute_with_ray(enrich_content_chunk, row_chunks, object_store={"dictionary_words": dictionary_words}))
            )

            # นำแถวที่ว่างออก
            data_without_empty_rows = [row for row in data if any(row)]

            with open("testPre2.csv", "w", encoding="utf-8", newline="") as enriched_csv:
                csv_writer = csv.writer(enriched_csv)
                csv_writer.writerows(data_without_empty_rows)

            ray.shutdown()

            csv_file_path = "testPre2.csv" 
            data_to_predict = pd.read_csv(csv_file_path)

            # ใช้ column ที่ 1-22 ในการทำนาย Content
            selected_data = data_to_predict.iloc[:, 1:22]

            # ใช้ความน่าจะเป็นในการทำนายผล
            prediction_probabilities = randon_for_tag.predict_proba(selected_data)

            
            predictions_confidence = prediction_probabilities[:, 1]

            
            categorized_predictions = []

            for confidence in predictions_confidence:
                if 0.7 <= confidence <= 1:
                    categorized_predictions.append(confidence)
                elif 0.4 <= confidence < 0.7:
                    categorized_predictions.append(confidence)
                elif 0 <= confidence <= 0.3:
                    categorized_predictions.append(confidence)
                else:
                    categorized_predictions.append(confidence) 

            data_to_predict['predictions'] = categorized_predictions

            output_csv_file = "predictions.csv" 
            data_to_predict.to_csv(output_csv_file, index=False)

            print("Predictions categorized and saved to", output_csv_file)

            # ในส่วนของ Content ถ้ามีค่าที่มากกว่า 0.2 จะให้เป็นค่า มีความเสี่ยง
            if all(prediction <= 0.2 for prediction in categorized_predictions):
                content_result = "Low Risk"
            else:
                content_result = "Risk"

            print("Content result:", content_result)

        else:
            print("Error: Received non-200 status code:", response.status_code)
            content_result = "Error: Received non-200 status code"
            
    except Timeout as e:
        print("Request timed out:", e)
        content_result = "Request timed out"
    except requests.exceptions.ConnectionError as e:
        print("Connection error:", e)
        content_result = "Connection error"
    except requests.exceptions.RequestException as e:
        print("Request error:", e)
        content_result = "Request error"

    enriched_results[0]["content_result"] = content_result

    return jsonify(enriched_results)

if __name__ == '__main__':
    app.run(debug=True)
