import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================Load Dataset========================================#
@st.cache_data
def load_data():
    local_path = "dashboard/all_data.csv"
    url = "https://github.com/projekardana/e-commerce-analysysis-data-science/blob/cfddfd8c889bd6aecc8b5621908057e6f3252f30/dashboard/all_data.csv"


    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
        st.success("Success Data dimuat di local")
    else :
        df = pd.read_csv(url)
        st.success("Success Data dimuat dari server GitHub")
    return df

all_data = load_data()

# ======================Sidebar Menu====================== #
with st.sidebar:
    st.image("img/Logo.png")
    st.title("Dashboard E-Commerce")
page = st.sidebar.radio("Pilih Halaman", [
    "Beranda",
    "Delivery Analysis",
    "Payment Methods",
    "Order Trend",
    "Geolocation Analysis",
    "Analysis Lanjutan (RFM)"
])

# ======================Beranda====================== #
if page == "Beranda":
    st.title("E-Commerce Dashboard")
    st.markdown("Dashboard ini menampilkan hasil analisis dari dataset `all_data.csv`")

    st.subheader("Preview Dataset")
    st.dataframe(all_data.head())

    st.subheader("Info Kolom Dataset")
    st.write(list(all_data.columns))

# ======================Delivery Analysis====================== #
elif page == "Delivery Analysis":
    st.title("Rata-rata Waktu Pengiriman")

    df = all_data.copy()
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])

    df = df.dropna(subset=['order_purchase_timestamp','order_delivered_customer_date'])

    df['delivery_time_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
    avg_delivery = df['delivery_time_days'].mean()

    st.metric("Rata-rata Waktu Pengiriman (Hari)", round(avg_delivery, 2))

    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(df['delivery_time_days'], bins=30, kde=True, ax=ax)
    ax.set_title("Distribusi Waktu Pengiriman")
    ax.set_xlabel("Hari")
    st.pyplot(fig)

# ======================Payment Methods====================== #
elif page == "Payment Methods":
    st.title("Metode Pembayaran Populer")

    payment_counts = all_data['payment_type'].value_counts().reset_index()
    payment_counts.columns = ["Payment Type", "Count"]

    fig, ax = plt.subplots(figsize=(7,5))
    sns.barplot(data=payment_counts, x="Count", y="Payment Type", palette="viridis", ax=ax)
    ax.set_title("Distribusi Metode Pembayaran")
    st.pyplot(fig)

# ======================Order Trend====================== #
elif page == "Order Trend":
    st.title("Tren Jumlah Pesanan per Bulan")

    orders_per_month = all_data.groupby("order_month")['order_id'].nunique().reset_index()
    orders_per_month.columns = ["Month", "Total Orders"]

    fig, ax = plt.subplots(figsize=(10,5))
    sns.lineplot(data=orders_per_month, x="Month", y="Total Orders", marker="o", ax=ax)
    ax.set_title("Tren Jumlah Pesanan Bulanan")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# ==============================Geolocation Analysis============================== #
elif page == "Geolocation Analysis":
    st.header("Sebaran Pesanan Pelanggan Berdasarkan Lokasi")

    orders = pd.read_csv("order_df.csv")
    customers = pd.read_csv("customers.csv")
    geolocation = pd.read_csv("geolocation.csv")

    geo_agg = geolocation.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]].mean().reset_index()
    orders_customer = orders.merge(customers, on="customer_id", how="left")

    orders_geo = orders_customer.merge(
        geo_agg,
        left_on="customer_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left"
    )

    orders_geo = orders_geo.rename(columns={
        "geolocation_lat": "lat",
        "geolocation_lng": "lon"
    })

    orders_geo = orders_geo.dropna(subset=["lat", "lon"])

    if len(orders_geo) > 2000:
        orders_geo = orders_geo.sample(2000, random_state=42)

    #     Menampilkan Peta
    st.map(orders_geo[["lat", "lon"]])

# ====================== Analysis Lanjutan (RFM) ====================== #
elif page == "Analysis Lanjutan (RFM)":
    st.title("Analysis Lanjutan (RFM)")

    df = all_data.copy()
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])

    snapshot_date = df['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('customer_id').agg({
        'order_purchase_timestamp': lambda x: (snapshot_date - x.max()).days,
        'order_id': 'count',
        'payment_value': 'sum'
    }).reset_index()
    rfm.columns = ['customer_id', 'Recency', 'Frequency', 'Monetary']

    # Skor RFM
    rfm['R'] = pd.qcut(rfm['Recency'], 4, labels=[4,3,2,1])
    rfm['F'] = pd.qcut(rfm['Frequency'].rank(method="first"), 4, labels=[1,2,3,4])
    rfm['M'] = pd.qcut(rfm['Monetary'], 4, labels=[1,2,3,4])
    rfm['RFM_Segment'] = rfm['R'].astype(str) + rfm['F'].astype(str) + rfm['M'].astype(str)
    rfm['RFM_Score'] = rfm[['R','F','M']].astype(int).sum(axis=1)

    rfm['Segment'] = 'Regular'
    rfm.loc[rfm['RFM_Score'] >= 9, 'Segment'] = 'Best Customers'
    rfm.loc[rfm['RFM_Score'] <= 5, 'Segment'] = 'Lost Customers'

    st.subheader("Distribusi Segmen Pelanggan")
    seg_counts = rfm['Segment'].value_counts().reset_index()
    seg_counts.columns = ['Segment', 'Count']

    fig, ax = plt.subplots(figsize=(7,5))
    sns.barplot(data=seg_counts, x="Segment", y="Count", palette="Set2", ax=ax)
    ax.set_title("Distribusi Customer Segments")
    st.pyplot(fig)

    st.subheader("Data RFM")
    st.dataframe(rfm.head())