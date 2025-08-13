# AI-Powered Business Insights Dashboard

An interactive AI-powered dashboard for retail/business analytics built with **Streamlit**.  
It enables users to upload datasets, generate insights, visualize trends, forecast sales, and export reports in PDF format — all without needing advanced technical skills.

---

##  Features

-  **Interactive Dashboard** – Visualize and analyze business performance instantly.
-  **Sales Forecasting** – Predict sales trends for the next 30 days using **Prophet**.
-  **Customer Segmentation** – Group customers with **RFM + KMeans**.
-  **Churn Prediction** – Estimate customer churn risk with **XGBoost**.
-  **Product Insights** – Identify top and low-performing products.
-  **PDF Export** – Generate professional reports with **ReportLab**.

---

##  Deployment

Deployed on **Streamlit Cloud**.

**Live Demo:**
[https://ai-business-insights-bemegbrtmjioermx6apgpr.streamlit.app/](https://ai-business-insights-bemegbrtmjioermx6apgpr.streamlit.app/)

---

##  Mandatory CSV File Format

Your uploaded CSV **must contain** these columns:

| Column Name    | Description                    |
| -------------- | ------------------------------ |
| `order_id`     | Unique order identifier        |
| `date`         | Order date (YYYY-MM-DD format) |
| `product_name` | Name of the product            |
| `category`     | Product category               |
| `price`        | Selling price per unit         |
| `cost_price`   | Cost price per unit            |
| `quantity`     | Number of units sold           |

⚠ **Note:** The app will not run if these columns are missing or misnamed.

---

##  Tech Stack

- **Frontend/UI**: Streamlit  
- **Backend/Language**: Python 3.x  
- **Data Handling**: Pandas, NumPy  
- **Visualization**: Matplotlib, Plotly  
- **Machine Learning**: Prophet, Scikit-learn, XGBoost, KMeans  
- **Report Generation**: ReportLab  
- **Development**: Jupyter Notebook  
- **Deployment**: Streamlit Cloud / Render  

---

##  Dependencies

**Python Version**: 3.7+  

- **streamlit** — For building the interactive web app  
- **pandas** — Data manipulation and analysis  
- **matplotlib** — Data visualization  
- **reportlab** — PDF report generation  
- **prophet** — Time series forecasting (optional)  
- **textblob** — Text analysis (optional)  
- **scikit-learn** — Machine learning utilities (optional)  

**Install all dependencies at once:**
```bash
pip install streamlit pandas matplotlib reportlab prophet textblob scikit-learn
````

---

##  Running Locally

```bash
git clone https://github.com/HIMANSHIWANJARI/AI-Business-Insights.git
cd AI-Business-Insights/dashboard
pip install -r requirements.txt
streamlit run app.py
```

---


##  Project Summary

**Purpose:**
A web-based dashboard for small retailers to gain AI-driven insights from sales data — including forecasting, segmentation, churn prediction, and product analysis.

**Workflow:**

1. Upload CSV → validate columns.
2. Analyze sales trends, customer groups, churn risk.
3. Visualize insights with charts and tables.
4. Export results as PDF/CSV.

**Performance Constraints:**

* Process data & return results in **≤5 seconds**.
* Works on any modern browser.

---

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests for bug fixes, new features, or improvements.

Thank you for using AI Retail Dashboard! We hope it helps you unlock valuable business insights with ease.
