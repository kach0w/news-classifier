import streamlit as st
import pandas as pd
import requests 
import json
import pickle
from datetime import datetime, timezone
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('API_KEY')
print(API_KEY)
st.write("""
# News Classifier
Using a Multinomial Naive Bayes Model to Classify News Headlines
""")

link = "https://newsapi.org/v2/top-headlines?country=us&apiKey=" + API_KEY
res = requests.get(link)
res = json.loads(res.text)["articles"]
# st.write(res)

model = pickle.load(open('model.pkl', 'rb'))
count = 0
counts = []
texts = []
for q in range(len(res)):
    if(res[q]["title"] != "[Removed]" and res[q]["description"] != "[Removed]" and res[q]["urlToImage"] is not None and count<=20):
        counts.append(q)
        # print(res[q]["urlToImage"]);
        # print(res[q]["title"][:5])
        # print("-----------------------------------")
        count+=1
    title = ""
    for i in range(len(res[q]["title"].split("-")) - 1):
        title +=  " " + res[q]["title"].split("-")[i]
    descript = str(res[q]["description"])        
    text = title + " " + descript
    texts.append(text)
pred = model.predict(texts)
for i in counts:
    utc_datetime = datetime.strptime(res[i]["publishedAt"], "%Y-%m-%dT%H:%M:%SZ")
    local_datetime = utc_datetime.replace(tzinfo=timezone.utc).astimezone()
    local_time_str = local_datetime.strftime("%I:%M:%S %p")
    title = ""
    for q in range(len(res[i]["title"].split("-")) - 1):
        title +=  " " + res[i]["title"].split("-")[q]
    
    string = f"""
        <a href={res[i]["url"]}>
            <div style="display: flex; justify-content: space-between;">
                <div style="width: 48%; background-color: #f0f0f0; padding: 10px;">
                    <img style="border-radius: 8px;" width="350px" src={res[i]["urlToImage"]}>
                </div>
                <div style="line-height: 0px; color: black; text-decoration: none; width: 48%; background-color: #f0f0f0; padding: 10px;">
                    <h5><b><a href={res[i]["url"]}>{title.replace('"', '').replace("'", "")}</a></b></h5>
                    <br>
                    <h6>{res[i]["source"]["name"]} | {local_time_str}</h6>
                    <br>
                    <i style="color:red;">{pred[i]}</i>
                </div>
            </div>
        </a>
    """
    st.write(string, unsafe_allow_html=True)
# print(pred)