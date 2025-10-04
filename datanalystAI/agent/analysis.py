from agent.visualization import generate_insights, \
    generate_report_and_get_images, create_visualization
from predictive_modelling.training import train_model
from preprocessing.process_data import preprocess_data, transform_data
import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
import io
import base64
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=UserWarning, module="pandas.core.arrays.masked")

def analyze_data(file_path, data_description, target_column=None, n=2, predictive_model=False, return_code=False):
    df = pd.read_csv(file_path)
    processed_data = preprocess_data(df, target_column)
    # print("target data type", processed_data[target_column].dtype)
    insights = generate_insights(processed_data.info(), data_description, n)
    visualizations = create_visualization(processed_data, insights)
    # visualizations = {'Explore the trend of delays over days of the week.': 'exports\\charts\\temp_chart_16d83871-2bf0-4514-801f-c31aeb584648.png',
    #                 'Find weather conditions associated with the most significant delays.': 'exports\\charts\\temp_chart_95fdae04-2d22-47a3-99a6-2b9be8b478c7.png'}
    html_path, zip_path = generate_report_and_get_images(visualizations)
    if predictive_model:
        model_fit, accuracy, importance = train_model(processed_data, target_column)
        # print(model_fit,'\n', accuracy, '\n', importance)
        importance = model_fit.feature_importances_
        feature_names = df.columns  # or model_fit.feature_names_in_ if available
        importance_dict = dict(zip(feature_names, importance))
        top_10 = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)[:10])
        colors = plt.cm.tab10(range(len(top_10)))
        plt.figure(figsize=(10, 6))
        bars = plt.barh(list(top_10.keys()), list(top_10.values()), color=colors)
        plt.gca().invert_yaxis()
        plt.title("Top 10 Feature Importances")
        plt.xlabel("Features")
        plt.ylabel("Importance Score")
        for bar in bars:
            width = bar.get_width()
            plt.text(width - 0.01, bar.get_y() + bar.get_height()/2,
                    f'{width * 100:.1f}%', va='center', ha='right', color='black', fontsize=10)
        plt.tight_layout()
        # plt.show()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        with open(html_path, "r+", encoding="utf-8") as f:
            html = f.read()

            # Insert image before </body> or at a placeholder
            heading = "<h2>Top 10 Feature Importances</h2>\n"
            img_tag = f'<img src="data:image/png;base64,{img_base64}" alt="Feature Importance Chart">\n'
            injection = heading + img_tag
            if "</body>" in html:
                html = html.replace("</body>", injection + "</body>")
            else:
                html += injection  # Append at the end if no </body> tag

            # Write back to file
            f.seek(0)
            f.write(html)
            f.truncate()
    return html_path, zip_path

# file_path = r"C:\Users\mamta\Desktop\code vita\agents\AI-Agents\archive\Stress_Dataset.csv"
# description = '''this is a survey data on young people replying to various reasons of stress using a number scale. higher number shows high agreement with the reason and vice versa.'''
# analyze_data(file_path = file_path, data_description = description, target_column = 'Which type of stress do you primarily experience?', predictive_model=True, n=3)