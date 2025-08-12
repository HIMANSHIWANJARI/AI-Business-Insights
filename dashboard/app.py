import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from prophet import Prophet
from io import StringIO, BytesIO
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import tempfile
import os  # Added missing import

st.set_page_config(
    page_title="AI Business Insights Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add this right after the page config
st.markdown("""
<style>
    /* Main container */
    .stApp {
        background-color: #f5f5f5;
    }
    
    /* Titles */
    h1 {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
    }
    
    /* Cards */
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    /* Metrics */
    [data-testid="stMetric"] {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Tabs */
    .stTabs [role="tablist"] {
        gap: 10px;
    }
    .stTabs [role="tab"] {
        border-radius: 8px 8px 0 0 !important;
        padding: 10px 20px !important;
    }
    
    /* Dataframes */
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Custom header - add this right after the CSS
st.markdown("""
<div style="background-color:#3498db;padding:20px;border-radius:10px;margin-bottom:30px">
    <h1 style="color:white;text-align:center;">AI Business Insights Dashboard</h1>
    <p style="color:white;text-align:center;">Powerful analytics for smarter decisions</p>
</div>
""", unsafe_allow_html=True)
st.title("🔹 AI Business Insights Dashboard")

# Upload files
# Replace the existing file upload section with this
upload_col1, upload_col2 = st.columns(2)
with upload_col1:
    sales_file = st.file_uploader(
        "📤 Upload Sales Data (CSV)", 
        type=["csv"],
        help="Upload your sales data in CSV format"
    )
with upload_col2:
    review_file = st.file_uploader(
        "📤 Upload Reviews Data (CSV)", 
        type=["csv"],
        help="Upload your customer reviews in CSV format"
    )

if sales_file and review_file:
    st.success("✅ Both files uploaded successfully!")
    st.balloons()

# Utility function to read CSV safely
COLUMN_MAP = {
    'transactionid': 'order_id',
    'order id': 'order_id',
    'orderid': 'order_id',
    
    'timestamp': 'date',
    'datetime': 'date',

    'userid': 'user_id',
    'user id': 'user_id',
    'customerid': 'user_id',
    'customer id': 'user_id',

    'itemname': 'product_name',
    'item name': 'product_name',
    'product': 'product_name',
    
    'productid': 'product_id',
    'product id': 'product_id',
    
    'categoryname': 'category',
    'productcategory': 'category',
    
    'qty': 'quantity',
    'units': 'quantity',

    'price': 'price',
    'retailprice': 'price',
    'mrp': 'price',

    'costprice': 'cost_price',
    'basecost': 'cost_price'
}

REQUIRED_SALES_COLUMNS = {'order_id', 'date', 'product_name', 'quantity', 'price', 'cost_price'}

def read_sales_csv(file):
    try:
        content = file.getvalue().decode("utf-8")
        if content.strip() == "":
            return None
        df = pd.read_csv(StringIO(content))

        # Normalize column names
        df.columns = df.columns.str.lower().str.strip()
        df.rename(columns=lambda col: COLUMN_MAP.get(col, col), inplace=True)

        # Check for required sales columns
        missing = REQUIRED_SALES_COLUMNS - set(df.columns)
        if missing:
            st.warning(f"⚠️ Sales CSV is missing required columns: {', '.join(missing)}")
            return None

        return df
    except Exception as e:
        st.error(f"❌ Error loading sales CSV: {e}")
        return None
    
def read_review_csv(file):
    try:
        content = file.getvalue().decode("utf-8")
        if content.strip() == "":
            return None
        df = pd.read_csv(StringIO(content))

        # Normalize column names
        df.columns = df.columns.str.lower().str.strip()

        # Check for review-specific columns
        required_review_cols = {'product_name', 'review'}
        missing = required_review_cols - set(df.columns)
        if missing:
            st.warning(f"⚠️ Review CSV is missing required columns: {', '.join(missing)}")
            return None

        return df
    except Exception as e:
        st.error(f"❌ Error loading review CSV: {e}")
        return None

sales_df = read_sales_csv(sales_file) if sales_file else None
review_df = read_review_csv(review_file) if review_file else None

# Tabs
tabs = st.tabs([
    "📊 Sales Insights", 
    "📊 EDA & Visualizations",
    "💬 Review Sentiment", 
    "📈 Forecasting", 
    "🎯 Product Recommendations", 
    "🤖 AI Business Advisor", 
    "📥 Download Report",
])

# ========== 1. Sales Insights ==========
with tabs[0]:
    st.subheader("📊 Sales Overview")

    if sales_df is not None:
        try:
            # Clean and prepare data
            df = sales_df.copy()
            df.columns = df.columns.str.lower().str.strip()

            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df.dropna(subset=['date', 'price', 'cost_price', 'quantity'], inplace=True)

            df['revenue'] = df['price'] * df['quantity']
            df['profit'] = (df['price'] - df['cost_price']) * df['quantity']

            # 📈 Key Metrics
            col1, col2 = st.columns(2)
            col1.metric("Total Revenue", f"₹ {df['revenue'].sum():,.2f}")
            col2.metric("Total Profit", f"₹ {df['profit'].sum():,.2f}")

            # 📅 Add time-based columns
            df['week'] = df['date'].dt.to_period('W').astype(str)
            df['month'] = df['date'].dt.to_period('M').astype(str)
            df['year'] = df['date'].dt.year

            # 📊 Monthly Revenue Trend Chart
            monthly = df.groupby('month')['revenue'].sum().reset_index()

            fig, ax = plt.subplots()
            ax.plot(monthly['month'], monthly['revenue'], marker='o')
            ax.set_title("Monthly Revenue Trend")
            ax.set_xlabel("Month")
            ax.set_ylabel("Revenue (₹)")
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # 🔍 Preview of the processed data
            st.dataframe(df.head())

            # ========================
            # 📆 Sales Summary Reports
            # ========================
            st.subheader("📆 Sales Summary Reports (Weekly / Monthly / Yearly)")

            # Order ID handling
            if 'order_id' not in df.columns:
                df['order_id'] = df.index  # fallback if no order_id

            # Weekly Summary
            weekly_summary = df.groupby('week').agg({
                'revenue': 'sum',
                'profit': 'sum',
                'order_id': 'nunique',
                'quantity': 'sum'
            }).reset_index().rename(columns={'order_id': 'orders'})

            # Monthly Summary
            monthly_summary = df.groupby('month').agg({
                'revenue': 'sum',
                'profit': 'sum',
                'order_id': 'nunique',
                'quantity': 'sum'
            }).reset_index().rename(columns={'order_id': 'orders'})

            # Yearly Summary
            yearly_summary = df.groupby('year').agg({
                'revenue': 'sum',
                'profit': 'sum',
                'order_id': 'nunique',
                'quantity': 'sum'
            }).reset_index().rename(columns={'order_id': 'orders'})

            # 📋 User selection
            report_type = st.selectbox("Select Report Type", ["Weekly", "Monthly", "Yearly"])

            if report_type == "Weekly":
                st.write("📅 **Weekly Sales Summary**")
                st.dataframe(weekly_summary)
                st.line_chart(weekly_summary.set_index('week')['revenue'])

            elif report_type == "Monthly":
                st.write("📆 **Monthly Sales Summary**")
                st.dataframe(monthly_summary)
                st.line_chart(monthly_summary.set_index('month')['revenue'])

            elif report_type == "Yearly":
                st.write("🗓️ **Yearly Sales Summary**")
                st.dataframe(yearly_summary)
                st.bar_chart(yearly_summary.set_index('year')['revenue'])

        except Exception as e:
            st.error(f"❌ Sales analysis failed: {e}")
    else:
        st.info("📎 Please upload a valid sales CSV.")

# ========== 2. EDA & Visualizations ==========
with tabs[1]:
    st.subheader("📊 Exploratory Data Analysis & Visualizations")

    if sales_df is not None:
        try:
            df = sales_df.copy()
            df.columns = df.columns.str.lower().str.strip()

            # Prepare columns
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df.dropna(subset=['date', 'price', 'cost_price', 'quantity'], inplace=True)

            df['revenue'] = df['price'] * df['quantity']
            df['profit'] = (df['price'] - df['cost_price']) * df['quantity']
            df['month'] = df['date'].dt.to_period('M')
            df['year'] = df['date'].dt.year
            df['week'] = df['date'].dt.isocalendar().week  # For weekly trend

            # === Summary ===
            with st.expander("📌 Dataset Summary"):
                st.write("#### Descriptive Statistics")
                st.dataframe(df.describe())

                st.write("#### Null Values")
                nulls = df.isnull().sum()
                nulls = nulls[nulls > 0]
                if not nulls.empty:
                    st.dataframe(nulls.reset_index().rename(columns={0: "Null Count", "index": "Column"}))
                else:
                    st.success("✅ No missing values detected.")

            # === 1. Daily Revenue ===
            st.markdown("### 📈 Daily Revenue Over Time")
            try:
                daily_sales = df.groupby('date')['revenue'].sum()
                fig1 = plt.figure()
                daily_sales.plot(marker='o')
                plt.title('Daily Revenue')
                plt.xlabel('Date')
                plt.ylabel('Revenue')
                plt.grid()
                st.pyplot(fig1)
            except Exception as e:
                st.warning(f"Could not plot daily revenue: {e}")

            # === 2. Monthly Sales Trend ===
            st.markdown("### 🗓️ Monthly Sales Trend")
            try:
                monthly_sales = df.groupby('month')['revenue'].sum()
                fig2 = plt.figure()
                monthly_sales.plot(kind='bar')
                plt.title('Monthly Revenue')
                plt.xlabel('Month')
                plt.ylabel('Revenue')
                plt.grid()
                st.pyplot(fig2)
            except Exception as e:
                st.warning(f"Could not plot monthly sales: {e}")

            # === 2.1 Yearly Sales Trend ===
            st.markdown("### 📅 Yearly Sales Trend")
            try:
                yearly_sales = df.groupby('year')['revenue'].sum()
                fig2_1 = plt.figure()
                yearly_sales.plot(kind='bar', color='orange')
                plt.title('Yearly Revenue')
                plt.xlabel('Year')
                plt.ylabel('Revenue')
                plt.grid()
                st.pyplot(fig2_1)
            except Exception as e:
                st.warning(f"Could not plot yearly sales: {e}")

            # === 3. Top Products ===
            st.markdown("### 🏆 Top 10 Selling Products")
            try:
                top_products = df.groupby('product_name')['quantity'].sum().sort_values(ascending=False).head(10)
                fig3 = plt.figure()
                top_products.plot(kind='bar')
                plt.title('Top Selling Products')
                plt.ylabel('Units Sold')
                plt.xlabel('Product')
                plt.xticks(rotation=45, ha='right')
                plt.grid(axis='y')
                st.pyplot(fig3)
            except Exception as e:
                st.warning(f"Could not plot top products: {e}")
            
            # === 4 Weekly Profit Trend ===
            st.markdown("### 📆 Weekly Profit Trend")
            try:
                weekly_profit = df.groupby('week')['profit'].sum()
                fig6 = plt.figure()
                weekly_profit.plot(marker='o', color='purple')
                plt.title("Weekly Profit Trend")
                plt.xlabel("Week Number")
                plt.ylabel("Profit")
                plt.grid()
                st.pyplot(fig6)
            except Exception as e:
                st.warning(f"Could not plot weekly profit trend: {e}")

            # === 5 Monthly Profit Trend ===
            st.markdown("### 💰 Monthly Profit Trend")
            try:
                monthly_profit = df.groupby('month')['profit'].sum()
                fig5 = plt.figure()
                monthly_profit.plot(marker='o', color='green')
                plt.title("Monthly Profit")
                plt.xlabel("Month")
                plt.ylabel("Profit")
                plt.grid()
                st.pyplot(fig5)
            except Exception as e:
                st.warning(f"Could not plot profit trend: {e}")

            # === 6. Yearly Profit Trend ===
            st.markdown("### 📅 Yearly Profit Trend")
            try:
                yearly_profit = df.groupby('year')['profit'].sum()
                fig7 = plt.figure()
                yearly_profit.plot(kind='bar', color='teal')
                plt.title("Yearly Profit Trend")
                plt.xlabel("Year")
                plt.ylabel("Profit")
                plt.grid()
                st.pyplot(fig7)
            except Exception as e:
                st.warning(f"Could not plot yearly profit trend: {e}")

        except Exception as e:
            st.error(f"❌ EDA failed: {e}")
    else:
        st.info("📎 Upload your sales CSV to view EDA.")

# ========== 3. Review Sentiment ==========
with tabs[2]:
    st.subheader("💬 Customer Review Sentiment")

    if review_df is not None:
        try:
            import nltk
            from nltk.sentiment import SentimentIntensityAnalyzer
            nltk.download('vader_lexicon')

            sia = SentimentIntensityAnalyzer()

            rev = review_df.copy()
            rev.columns = rev.columns.str.lower().str.strip()

            # Validate required columns
            if 'review' not in rev.columns or 'product_name' not in rev.columns:
                st.error("❌ 'review' and 'product_name' columns are required in your CSV.")
            elif 'date' not in rev.columns:
                st.error("❌ 'date' column is required to plot monthly/yearly trends.")
            else:
                # Convert date column to datetime
                rev['date'] = pd.to_datetime(rev['date'], errors='coerce')
                rev.dropna(subset=['date'], inplace=True)

                # Sentiment categorization using VADER
                def categorize_sentiment(comment):
                    scores = sia.polarity_scores(str(comment))
                    compound = scores['compound']
                    if compound >= 0.05:
                        return "Positive"
                    elif compound <= -0.05:
                        return "Negative"
                    else:
                        return "Neutral"

                # Apply classification
                rev['label'] = rev['review'].apply(categorize_sentiment)

                # Ensure all sentiment types appear in counts
                sentiment_counts = rev['label'].value_counts().reindex(
                    ['Positive', 'Negative', 'Neutral'], fill_value=0
                )

                # 📊 Pie chart
                fig, ax = plt.subplots()
                colors = ['green', 'red', 'gold']
                wedges, texts, autotexts = ax.pie(
                    sentiment_counts,
                    labels=sentiment_counts.index,
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=colors,
                    wedgeprops={'edgecolor': 'white'}
                )
                plt.setp(autotexts, size=10, weight="bold")
                ax.set_title("Customer Review Sentiment Distribution", fontsize=14, fontweight='bold')
                st.pyplot(fig)

                # 📊 Bar chart
                st.markdown("### Sentiment Count")
                st.bar_chart(sentiment_counts)

                # 📈 Monthly sentiment trend
                st.markdown("### 📅 Monthly Sentiment Trend")
                monthly_counts = rev.groupby([rev['date'].dt.to_period('M'), 'label']).size().unstack(fill_value=0)
                monthly_counts.index = monthly_counts.index.to_timestamp()
                st.line_chart(monthly_counts)

                # 📆 Yearly sentiment trend
                st.markdown("### 📅 Yearly Sentiment Trend")
                yearly_counts = rev.groupby([rev['date'].dt.year, 'label']).size().unstack(fill_value=0)
                st.bar_chart(yearly_counts)

        except Exception as e:
            st.error(f"❌ Sentiment analysis failed: {e}")
    else:
        st.info("📎 Please upload a valid review CSV.")

# ========== 4. Forecasting ==========
with tabs[3]:
    st.subheader("📈 Revenue Forecast")
    if sales_df is not None:
        try:
            from prophet import Prophet
            import matplotlib.pyplot as plt
            import matplotlib.ticker as ticker

            # Prepare data
            df = sales_df.copy()
            df.columns = df.columns.str.lower().str.strip()
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df.dropna(subset=['date', 'price', 'quantity'], inplace=True)
            df['revenue'] = df['price'] * df['quantity']

            # Prepare data for Prophet
            prophet_df = df[['date', 'revenue']].rename(columns={'date': 'ds', 'revenue': 'y'})

            # Train model
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            model.fit(prophet_df)

            # Create future dataframe for 1 year ahead
            future = model.make_future_dataframe(periods=365)
            forecast = model.predict(future)

            # Rename columns for readability
            forecast = forecast.rename(columns={
                'yhat': 'Expected Revenue',
                'yhat_lower': 'Minimum Expected Revenue',
                'yhat_upper': 'Maximum Expected Revenue'
            })

            # -------- YEARLY FORECAST --------
            st.markdown("## 📅 Yearly Forecast")
            forecast['Year'] = forecast['ds'].dt.year
            yearly_forecast = forecast.groupby('Year').agg({
                'Expected Revenue': 'sum',
                'Minimum Expected Revenue': 'sum',
                'Maximum Expected Revenue': 'sum'
            }).reset_index()

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(yearly_forecast['Year'], yearly_forecast['Expected Revenue'], color='blue', marker='o', label='Expected Revenue')
            ax.fill_between(yearly_forecast['Year'], yearly_forecast['Minimum Expected Revenue'], yearly_forecast['Maximum Expected Revenue'],
                            color='skyblue', alpha=0.3, label='Confidence Interval')
            ax.set_title("Yearly Revenue Forecast", fontsize=14, fontweight='bold')
            ax.set_ylabel("Revenue")
            ax.set_xlabel("Year")
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax.legend()
            st.pyplot(fig)

            st.dataframe(yearly_forecast.style.format({
                'Expected Revenue': '₹{:.2f}',
                'Minimum Expected Revenue': '₹{:.2f}',
                'Maximum Expected Revenue': '₹{:.2f}'
            }).set_properties(**{'background-color': '#f9f9f9'}))

            # -------- MONTHLY FORECAST --------
            st.markdown("## 📅 Monthly Forecast")
            forecast['Month'] = forecast['ds'].dt.to_period('M').apply(lambda r: r.start_time)
            monthly_forecast = forecast.groupby('Month').agg({
                'Expected Revenue': 'sum',
                'Minimum Expected Revenue': 'sum',
                'Maximum Expected Revenue': 'sum'
            }).reset_index()

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(monthly_forecast['Month'], monthly_forecast['Expected Revenue'], color='green', marker='o', label='Expected Revenue')
            ax.fill_between(monthly_forecast['Month'], monthly_forecast['Minimum Expected Revenue'], monthly_forecast['Maximum Expected Revenue'],
                            color='lightgreen', alpha=0.3, label='Confidence Interval')
            ax.set_title("Monthly Revenue Forecast", fontsize=14, fontweight='bold')
            ax.set_ylabel("Revenue")
            ax.set_xlabel("Month")
            ax.xaxis.set_major_locator(ticker.MaxNLocator(12))
            plt.xticks(rotation=45)
            ax.legend()
            st.pyplot(fig)

            st.dataframe(monthly_forecast.style.format({
                'Expected Revenue': '₹{:.2f}',
                'Minimum Expected Revenue': '₹{:.2f}',
                'Maximum Expected Revenue': '₹{:.2f}'
            }).set_properties(**{'background-color': '#f4fff4'}))

            # -------- WEEKLY FORECAST --------
            st.markdown("## 📅 Weekly Forecast")
            forecast['Week'] = forecast['ds'].dt.to_period('W').apply(lambda r: r.start_time)
            weekly_forecast = forecast.groupby('Week').agg({
                'Expected Revenue': 'sum',
                'Minimum Expected Revenue': 'sum',
                'Maximum Expected Revenue': 'sum'
            }).reset_index()

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(weekly_forecast['Week'], weekly_forecast['Expected Revenue'], color='orange', marker='o', label='Expected Revenue')
            ax.fill_between(weekly_forecast['Week'], weekly_forecast['Minimum Expected Revenue'], weekly_forecast['Maximum Expected Revenue'],
                            color='navajowhite', alpha=0.3, label='Confidence Interval')
            ax.set_title("Weekly Revenue Forecast", fontsize=14, fontweight='bold')
            ax.set_ylabel("Revenue")
            ax.set_xlabel("Week")
            ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
            plt.xticks(rotation=45)
            ax.legend()
            st.pyplot(fig)

            st.dataframe(weekly_forecast.style.format({
                'Expected Revenue': '₹{:.2f}',
                'Minimum Expected Revenue': '₹{:.2f}',
                'Maximum Expected Revenue': '₹{:.2f}'
            }).set_properties(**{'background-color': '#fffaf4'}))

        except Exception as e:
            st.error(f"❌ Forecasting failed: {e}")
    else:
        st.info("📎 Upload sales CSV for forecasting.")

# ========== 5. Product Recommendations ==========
with tabs[4]:
    st.subheader("🎯 Top Products by Quantity Sold & Profit")
    if sales_df is not None:
        try:
            import plotly.express as px

            # ===== Data Cleaning =====
            df = sales_df.copy()
            df.columns = df.columns.str.lower().str.strip()
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['profit'] = (df['price'] - df['cost_price']) * df['quantity']

            # ===== Top Products Overall =====
            top_products = (
                df.groupby('product_name')
                .agg({'quantity': 'sum', 'profit': 'sum'})
                .sort_values(by='quantity', ascending=False)
                .head(5)
                .reset_index()
            )
            top_products.columns = ['Product Name', 'Quantity Sold', 'Total Profit']

            st.subheader("🏆 Overall Top Products")
            st.table(
                top_products.style.format({
                    'Quantity Sold': '{:.0f}',
                    'Total Profit': '₹{:.2f}'
                })
            )

            # ===== Weekly Trend =====
            df['Week'] = df['date'].dt.strftime('%Y-%U')  # Year-Week format
            weekly_best = (
                df.groupby(['Week', 'product_name'])
                .agg({'quantity': 'sum', 'profit': 'sum'})
                .reset_index()
            )

            # Best selling per week
            weekly_best_selling = weekly_best.loc[
                weekly_best.groupby('Week')['quantity'].idxmax()
            ].rename(columns={
                'product_name': 'Best Selling Product',
                'quantity': 'Quantity Sold'
            })

            # Most profitable per week
            weekly_most_profitable = weekly_best.loc[
                weekly_best.groupby('Week')['profit'].idxmax()
            ].rename(columns={
                'product_name': 'Most Profitable Product',
                'profit': 'Total Profit'
            })

            # Merge results
            weekly_summary = pd.merge(weekly_best_selling[['Week', 'Best Selling Product', 'Quantity Sold']],
                                      weekly_most_profitable[['Week', 'Most Profitable Product', 'Total Profit']],
                                      on='Week')

            st.subheader("📅 Weekly Best Products")
            st.dataframe(
                weekly_summary.style.format({
                    'Quantity Sold': '{:.0f}',
                    'Total Profit': '₹{:.2f}'
                })
            )

            # Weekly chart
            fig_weekly_qty = px.bar(
                weekly_best_selling,
                x='Week', y='Quantity Sold', color='Best Selling Product',
                title="Weekly Best Selling Products",
                text='Quantity Sold'
            )
            st.plotly_chart(fig_weekly_qty, use_container_width=True)

            fig_weekly_profit = px.bar(
                weekly_most_profitable,
                x='Week', y='Total Profit', color='Most Profitable Product',
                title="Weekly Most Profitable Products",
                text='Total Profit'
            )
            st.plotly_chart(fig_weekly_profit, use_container_width=True)

            # ===== Monthly Trend =====
            df['Month'] = df['date'].dt.strftime('%Y-%m')  # Year-Month format
            monthly_best = (
                df.groupby(['Month', 'product_name'])
                .agg({'quantity': 'sum', 'profit': 'sum'})
                .reset_index()
            )

            # Best selling per month
            monthly_best_selling = monthly_best.loc[
                monthly_best.groupby('Month')['quantity'].idxmax()
            ].rename(columns={
                'product_name': 'Best Selling Product',
                'quantity': 'Quantity Sold'
            })

            # Most profitable per month
            monthly_most_profitable = monthly_best.loc[
                monthly_best.groupby('Month')['profit'].idxmax()
            ].rename(columns={
                'product_name': 'Most Profitable Product',
                'profit': 'Total Profit'
            })

            # Merge results
            monthly_summary = pd.merge(monthly_best_selling[['Month', 'Best Selling Product', 'Quantity Sold']],
                                       monthly_most_profitable[['Month', 'Most Profitable Product', 'Total Profit']],
                                       on='Month')

            st.subheader("📆 Monthly Best Products")
            st.dataframe(
                monthly_summary.style.format({
                    'Quantity Sold': '{:.0f}',
                    'Total Profit': '₹{:.2f}'
                })
            )

            # Monthly chart
            fig_monthly_qty = px.bar(
                monthly_best_selling,
                x='Month', y='Quantity Sold', color='Best Selling Product',
                title="Monthly Best Selling Products",
                text='Quantity Sold'
            )
            st.plotly_chart(fig_monthly_qty, use_container_width=True)

            fig_monthly_profit = px.bar(
                monthly_most_profitable,
                x='Month', y='Total Profit', color='Most Profitable Product',
                title="Monthly Most Profitable Products",
                text='Total Profit'
            )
            st.plotly_chart(fig_monthly_profit, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Recommendation failed: {e}")
    else:
        st.info("📎 Upload sales CSV to generate recommendations.")

# ========== 6. AI Business Advisor ==========
with tabs[5]:
    st.subheader("🤖 AI Business Advisor — Strategic Profit & Sales Insights")

    if sales_df is not None and review_df is not None:
        try:
            import plotly.express as px
            from sklearn.preprocessing import MinMaxScaler
            from textblob import TextBlob

            # --- Data Prep ---
            sales = sales_df.copy()
            reviews = review_df.copy()

            sales.columns = sales.columns.str.lower().str.strip()
            reviews.columns = reviews.columns.str.lower().str.strip()

            required_sales_cols = {'product_name', 'quantity', 'price', 'cost_price', 'date'}
            required_review_cols = {'product_name', 'review'}
            if not required_sales_cols.issubset(sales.columns) or not required_review_cols.issubset(reviews.columns):
                st.error("❌ Missing required columns in one of the files.")
            else:
                sales['date'] = pd.to_datetime(sales['date'], errors='coerce')
                sales['profit'] = (sales['price'] - sales['cost_price']) * sales['quantity']

                reviews['sentiment'] = reviews['review'].astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)

                sentiment_df = reviews.groupby('product_name')['sentiment'].mean().reset_index()
                sales_agg = sales.groupby('product_name').agg({
                    'quantity': 'sum',
                    'profit': 'sum'
                }).reset_index()

                merged = pd.merge(sales_agg, sentiment_df, on='product_name', how='inner')
                scaler = MinMaxScaler()
                merged[['quantity', 'profit', 'sentiment']] = scaler.fit_transform(merged[['quantity', 'profit', 'sentiment']])
                merged['score'] = 0.4 * merged['quantity'] + 0.4 * merged['profit'] + 0.2 * merged['sentiment']
                merged = merged.sort_values(by='score', ascending=False)

                # --- KPI Cards ---
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("💰 Total Revenue", f"₹{(sales['price']*sales['quantity']).sum():,.2f}")
                col2.metric("📈 Total Profit", f"₹{sales['profit'].sum():,.2f}")
                col3.metric("🏆 Best Seller", sales.groupby('product_name')['quantity'].sum().idxmax())
                col4.metric("💹 Most Profitable", sales.groupby('product_name')['profit'].sum().idxmax())

                # --- Recommendations ---
                st.markdown("### ✅ Recommended Products to Promote")
                st.dataframe(
                    merged.head(3)[['product_name', 'quantity', 'profit', 'sentiment', 'score']]
                    .style.format({'quantity': '{:.0f}', 'profit': '₹{:.2f}', 'sentiment': '{:.2f}', 'score': '{:.2f}'})
                )

                st.markdown("### ⚠️ Products to Rethink")
                st.dataframe(
                    merged.tail(2)[['product_name', 'quantity', 'profit', 'sentiment', 'score']]
                    .style.format({'quantity': '{:.0f}', 'profit': '₹{:.2f}', 'sentiment': '{:.2f}', 'score': '{:.2f}'})
                )

                # --- Weekly Trends ---
                st.markdown("### 📅 Weekly Top Sellers & Most Profitable")
                sales['week'] = sales['date'].dt.to_period('W').astype(str)
                weekly_sales = sales.groupby(['week', 'product_name']).agg({
                    'quantity': 'sum',
                    'profit': 'sum'
                }).reset_index()

                top_weekly = weekly_sales.sort_values(['week', 'quantity'], ascending=[True, False]).groupby('week').head(1)
                top_profit_weekly = weekly_sales.sort_values(['week', 'profit'], ascending=[True, False]).groupby('week').head(1)

                st.write("**Top Seller Each Week**")
                st.dataframe(top_weekly)

                st.write("**Most Profitable Each Week**")
                st.dataframe(top_profit_weekly)

                # --- Monthly Trends ---
                st.markdown("### 📆 Monthly Top Sellers & Most Profitable")
                sales['month'] = sales['date'].dt.to_period('M').astype(str)
                monthly_sales = sales.groupby(['month', 'product_name']).agg({
                    'quantity': 'sum',
                    'profit': 'sum'
                }).reset_index()

                top_monthly = monthly_sales.sort_values(['month', 'quantity'], ascending=[True, False]).groupby('month').head(1)
                top_profit_monthly = monthly_sales.sort_values(['month', 'profit'], ascending=[True, False]).groupby('month').head(1)

                st.write("**Top Seller Each Month**")
                st.dataframe(top_monthly)

                st.write("**Most Profitable Each Month**")
                st.dataframe(top_profit_monthly)

                # --- Visualisation ---
                st.plotly_chart(
                    px.bar(merged, x='product_name', y='score', color='score',
                           title="📊 Product Performance Score", text='score'),
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"❌ Advisor failed: {e}")
    else:
        st.info("📎 Upload both sales and review CSVs for advisor suggestions.")

# ========== 7. Download Report ==========
with tabs[6]:
    st.subheader("📥 Download Final Sales & Reviews Report")
    
    if sales_df is not None and review_df is not None:
        try:
            temp_dir = tempfile.mkdtemp()
            
            # Prepare sales data
            sales_df['date'] = pd.to_datetime(sales_df['date'], errors='coerce')
            
            # Calculate revenue if not already present
            if 'revenue' not in sales_df.columns:
                if 'price' in sales_df.columns and 'quantity' in sales_df.columns:
                    sales_df['revenue'] = sales_df['price'] * sales_df['quantity']
                else:
                    st.error("❌ Missing required columns ('price' and 'quantity') to calculate revenue")
                    st.stop()  # Use st.stop() instead of return to halt execution
            
            # Monthly sales data
            monthly_sales = sales_df.groupby(sales_df['date'].dt.to_period('M'))['revenue'].sum()
            
            # Monthly Sales Chart
            plt.figure(figsize=(6, 4))
            monthly_sales.plot(kind='line', marker='o', title="Monthly Sales Trend")
            plt.ylabel("Revenue")
            monthly_sales_chart = os.path.join(temp_dir, "monthly_sales.png")
            plt.savefig(monthly_sales_chart, bbox_inches='tight')
            plt.close()

            # Category Sales Chart (if category exists)
            if 'category' in sales_df.columns:
                category_sales = sales_df.groupby('category')['revenue'].sum()
                plt.figure(figsize=(6, 4))
                category_sales.plot(kind='pie', autopct='%1.1f%%', title="Sales by Category")
                plt.ylabel("")
                category_sales_chart = os.path.join(temp_dir, "category_sales.png")
                plt.savefig(category_sales_chart, bbox_inches='tight')
                plt.close()
            else:
                category_sales_chart = None
                st.warning("⚠️ 'category' column not found - skipping category analysis")

            # Prepare review data
            if 'rating' in review_df.columns:
                avg_ratings = review_df.groupby('product_name')['rating'].mean().sort_values(ascending=False).head(10)
                plt.figure(figsize=(6, 4))
                avg_ratings.plot(kind='bar', title="Top 10 Products by Average Rating")
                plt.ylabel("Rating")
                avg_ratings_chart = os.path.join(temp_dir, "avg_ratings.png")
                plt.savefig(avg_ratings_chart, bbox_inches='tight')
                plt.close()
            else:
                avg_ratings_chart = None
                st.warning("⚠️ 'rating' column not found - skipping rating analysis")

            # Calculate metrics for report
            total_revenue = sales_df['revenue'].sum()
            
            # Count unique orders (fallback to row count if no order_id)
            if 'order_id' in sales_df.columns:
                total_orders = sales_df['order_id'].nunique()
            else:
                total_orders = len(sales_df)
                st.warning("⚠️ 'order_id' column not found - using row count as order count")

            # Average rating (if available)
            if 'rating' in review_df.columns:
                avg_rating = review_df['rating'].mean()
                avg_rating_text = f"{avg_rating:.2f}"
            else:
                avg_rating_text = "N/A (No rating column)"

            # Create PDF report
            buffer = BytesIO()
            pdf = SimpleDocTemplate(buffer, pagesize=A4)
            styles = getSampleStyleSheet()
            elements = []

            elements.append(Paragraph("📊 Sales & Reviews Report", styles['Title']))
            elements.append(Spacer(1, 12))

            # Add metrics to report
            elements.append(Paragraph(f"<b>Total Revenue:</b> ₹{total_revenue:,.2f}", styles['Normal']))
            elements.append(Paragraph(f"<b>Total Orders:</b> {total_orders}", styles['Normal']))
            elements.append(Paragraph(f"<b>Average Rating:</b> {avg_rating_text}", styles['Normal']))
            elements.append(Spacer(1, 12))

            # Add charts to report
            elements.append(Paragraph("📈 Monthly Sales Trend", styles['Heading2']))
            elements.append(Image(monthly_sales_chart, width=400, height=250))
            elements.append(Spacer(1, 12))

            if category_sales_chart:
                elements.append(Paragraph("📦 Sales by Category", styles['Heading2']))
                elements.append(Image(category_sales_chart, width=400, height=250))
                elements.append(Spacer(1, 12))

            if avg_ratings_chart:
                elements.append(Paragraph("⭐ Top Products by Rating", styles['Heading2']))
                elements.append(Image(avg_ratings_chart, width=400, height=250))
                elements.append(Spacer(1, 12))

            pdf.build(elements)

            st.download_button(
                label="📄 Download Final PDF Report",
                data=buffer.getvalue(),
                file_name="business_insights_report.pdf",
                mime="application/pdf"
            )

        except Exception as e:
            st.error(f"❌ Report generation failed: {str(e)}")
    else:
        st.info("📎 Please upload both Sales and Reviews CSV files to generate the report.")