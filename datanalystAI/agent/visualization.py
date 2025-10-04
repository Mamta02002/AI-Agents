##use llm to understand the data and generate some point which need to be visualized and insights to be drawn in the data
##llm will generate those points
##use pandas AI and give it those points to generate visualizations
##based on visualization and analysis done by pandas AI charts data, insights will be written down

from pandasai import Agent
import pandasai as pai
import pandas as pd
import matplotlib.pyplot as plt
import os
import requests
from openai import OpenAI
from io import StringIO
from pandasai_litellm import LiteLLM
from pandasai import SmartDataframe
import ast
import time
from pathlib import Path
import shutil
import os
import zipfile


def generate_insights(data_summary, description, n):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="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"
    )
    
    completion = client.chat.completions.create(
        model="qwen/qwen-2.5-72b-instruct:free",
        messages=[
            {
                "role": "user",
                "content": f"""
                        You are a professional data analyst.

                        You are given a dataset with the following:
                        - Description: {description}
                        - Summary: {data_summary}

                        Your task is to identify the {n} most important analytical points in the dataset. Focus on:
                        - Trends
                        - Outliers
                        - Correlations
                        - Performance metrics

                        Instructions:
                        1. Return a Python dictionary.
                        2. Each key should be a clear title of the analysis.
                        3. Each value should be:
                        - A one-liner prompt suitable for PandasAI's df.chat method.
                        - If the analysis is broad, return a list of short, specific prompts instead of vague ones.
                        4. Do NOT include any code, formulas, or explanations.
                        5. Do NOT use SmartDataframe.
                        6. Keep prompts concise and actionable.

                        Only return the dictionary. No extra text.
                        """
            }
        ],
    )
    return completion.choices[0].message.content

def create_visualization(df, insights: str, verbose=False):
    llm = LiteLLM(
            model="openrouter/qwen/qwen-2.5-72b-instruct:free", # Or another Qwen model available on OpenRouter
            api_key="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", # Replace with your actual OpenRouter API key
            base_url="https://openrouter.ai/api/v1"
            )
    pai.config.set({
            "llm": llm,
            'verbose': verbose, 
            'enable_cache': False, 
            'max_retries': 2
        })
    data = pai.DataFrame(df)
    visualizations = {}
    insights_dict = ast.literal_eval(insights)
    for point in insights_dict.values():
        if type(point)==str:
            prompt_viz = f"Please analyze {point.strip()} and create a suitable visualization"
            try:
                fig = data.chat(prompt_viz)
                visualizations[point] = fig
            except Exception as e:
                if verbose:
                    print(f"⚠️ Could not generate visualization for: {point.strip()}")
            time.sleep(5)
        elif type(point)==list:
            for sub_point in point:
                prompt_viz = f"Please analyze {sub_point.strip()} and create a suitable visualization"
                try:
                    fig = data.chat(prompt_viz)
                    visualizations[sub_point] = fig
                except Exception as e:
                    if verbose:
                        print(f"⚠️ Could not generate visualization for: {point.strip()}")
                time.sleep(5)
    return visualizations

def generate_report_and_get_images(visualizations):
    output_dir = Path("output")
    images_dir = output_dir / "renamed_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Start HTML content
    html = """
    <html>
    <head>
        <title>Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; padding: 0px; }
            h2 { margin-top: 40px; }
            img { max-width: 800px; height: auto; margin-bottom: 20px; }
            .error { color: red; font-style: italic; }
        </style>
    </head>
    <body>
    <h1>Stress Analysis Report</h1>
    """

    # Process each item
    for title, response in visualizations.items():
        html += f"<h2>{title}</h2>"
        if str(response).endswith('.png'):
            # Clean filename
            safe_filename = "".join(c if c.isalnum() else "_" for c in title) + ".png"
            src_path = Path(response.value)
            dst_path = images_dir / safe_filename

            # Copy and rename image
            shutil.copy(src_path, dst_path)

            # Add image to HTML
            html += f'<img src="output/renamed_images/{safe_filename}" alt="{title}"><br>'
        else:
            html += f'<p class="error">{response}</p>'

    html += "</body></html>"

    # Save HTML file
    html_path = output_dir / "analysis_report.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    # Create ZIP file of renamed images
    zip_path = output_dir / "plots.zip"
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for img_file in images_dir.glob("*.png"):
            zipf.write(img_file, arcname=img_file.name)

    print(f"✅ HTML report saved to: {html_path}")
    print(f"✅ ZIP file created: {zip_path}")
    return html_path, zip_path

