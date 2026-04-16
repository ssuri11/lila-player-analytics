# (ONLY IMPORTANT NOTE: This is your FULL APP — no sections removed)

import streamlit as st
import pyarrow.parquet as pq
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
import datetime
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import zipfile

st.set_page_config(layout="wide")

# =========================
# UI
# =========================
st.markdown("""
<style>
.stApp { background-color: #0e1117; color: #FAFAFA; }
section[data-testid="stSidebar"] { background-color: #161a23; }
p, label, span { color: #E0E0E0 !important; }
</style>
""", unsafe_allow_html=True)

st.title("🎮 LILA Player Analytics & Prediction Tool")

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "player_data")
ZIP_PATH = os.path.join(BASE_DIR, "player_data.zip")

# =========================
# SAFE UNZIP (NO CRASH)
# =========================
try:
    if not os.path.exists(DATA_PATH) and os.path.exists(ZIP_PATH):
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(BASE_DIR)

        # Handle nested folder
        for item in os.listdir(BASE_DIR):
            item_path = os.path.join(BASE_DIR, item)
            if os.path.isdir(item_path):
                if "player_data" in os.listdir(item_path):
                    os.rename(os.path.join(item_path, "player_data"), DATA_PATH)
except Exception as e:
    st.error(f"Zip extraction error: {e}")
    st.stop()

# =========================
# MAP CONFIG
# =========================
MAP_IMAGES = {
    "AmbroseValley": os.path.join(BASE_DIR, "minimaps", "AmbroseValley_Minimap.png"),
    "GrandRift": os.path.join(BASE_DIR, "minimaps", "GrandRift_Minimap.png"),
    "Lockdown": os.path.join(BASE_DIR, "minimaps", "Lockdown_Minimap.jpg"),
}

MAP_CONFIG = {
    "AmbroseValley": {"scale": 900, "origin_x": -370, "origin_z": -473},
    "GrandRift": {"scale": 581, "origin_x": -290, "origin_z": -290},
    "Lockdown": {"scale": 1000, "origin_x": -500, "origin_z": -500},
}

EVENT_COLORS = {
    "Position": "blue",
    "BotPosition": "cyan",
    "Kill": "red",
    "BotKill": "darkred",
    "Killed": "black",
    "BotKilled": "gray",
    "KilledByStorm": "purple",
    "Loot": "gold",
}

PATH_COLORS = {True: "orange", False: "blue"}

# =========================
# LOAD DATA (SAFE)
# =========================
@st.cache_data
def load_data():
    frames = []

    if not os.path.exists(DATA_PATH):
        return pd.DataFrame()

    for folder in os.listdir(DATA_PATH):
        path = os.path.join(DATA_PATH, folder)
        if not os.path.isdir(path):
            continue

        try:
            date = datetime.datetime.strptime(folder, "%B_%d").replace(year=2024)
        except:
            continue

        for f in os.listdir(path):
            try:
                df = pq.read_table(os.path.join(path, f)).to_pandas()
                df["event"] = df["event"].apply(lambda x: x.decode() if isinstance(x, bytes) else x)
                df["date"] = date
                frames.append(df)
            except:
                continue

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["is_bot"] = df["user_id"].apply(lambda x: str(x).isdigit())
    df["ts"] = pd.to_datetime(df["ts"])
    return df

df = load_data()

# =========================
# SAFE STOP (NO CRASH LOOP)
# =========================
if df.empty:
    st.error("⚠️ No data found. Please check player_data.zip structure.")
    st.stop()

# =========================
# DATE FILTER
# =========================
with st.sidebar:
    start = st.date_input("Start Date", df["date"].min())
    end = st.date_input("End Date", df["date"].max())

df = df[(df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))]

# =========================
# TABS
# =========================
tab1, tab2 = st.tabs(["📊 Visualization", "🤖 Prediction"])

# =========================
# VISUALIZATION
# =========================
with tab1:

    col1, col2, col3, col4, col5 = st.columns(5)

    map_choice = col1.selectbox("Map", ["All Maps"] + sorted(df["map_id"].unique()))
    map_df = df if map_choice == "All Maps" else df[df["map_id"] == map_choice]

    match_choice = col2.selectbox("Match", ["All Matches"] + sorted(map_df["match_id"].unique()))
    filtered_df = map_df if match_choice == "All Matches" else map_df[map_df["match_id"] == match_choice]

    player_type = col4.selectbox("Player Type", ["All Players", "Human Only", "Bot Only"])
    if player_type == "Human Only":
        filtered_df = filtered_df[~filtered_df["is_bot"]]
    elif player_type == "Bot Only":
        filtered_df = filtered_df[filtered_df["is_bot"]]

    selected_player = col5.selectbox("Player", ["All Players"] + sorted(filtered_df["user_id"].astype(str).unique()))
    if selected_player != "All Players":
        filtered_df = filtered_df[filtered_df["user_id"].astype(str) == selected_player]

    path_df_full = filtered_df.copy()

    selected_events = col3.multiselect(
        "Events",
        sorted(filtered_df["event"].unique()),
        default=sorted(filtered_df["event"].unique())
    )
    filtered_df = filtered_df[filtered_df["event"].isin(selected_events)]

    show_paths = st.checkbox("Show Paths", True)
    show_points = st.checkbox("Show Events", True)
    show_heatmap = st.checkbox("🔥 Show Heatmap", False)
    heatmap_mode = st.selectbox("Heatmap Type", ["All", "Humans Only", "Bots Only"])
    show_hotspots = st.checkbox("🏆 Show Hotspots", False)

    def map_coords(df, map_name, w, h):
        cfg = MAP_CONFIG[map_name]
        df = df.copy()
        df["mx"] = ((df["x"] - cfg["origin_x"]) / cfg["scale"]) * w
        df["my"] = (1 - (df["z"] - cfg["origin_z"]) / cfg["scale"]) * h
        return df

    def plot_map(map_name, data):
        img = Image.open(MAP_IMAGES[map_name])
        w, h = img.size

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img)

        df_plot = map_coords(data, map_name, w, h)

        # Heatmap filtering
        heat_df = df_plot.copy()
        if heatmap_mode == "Humans Only":
            heat_df = heat_df[~heat_df["is_bot"]]
        elif heatmap_mode == "Bots Only":
            heat_df = heat_df[heat_df["is_bot"]]

        # Heatmap
        if show_heatmap and len(heat_df) > 0:
            heat, _, _ = np.histogram2d(heat_df["mx"], heat_df["my"], bins=60)
            ax.imshow(heat.T, extent=[0, w, h, 0], cmap="inferno", alpha=0.5)

        # Hotspots
        if show_hotspots and len(heat_df) > 0:
            heat, _, _ = np.histogram2d(heat_df["mx"], heat_df["my"], bins=20)
            idx = np.dstack(np.unravel_index(np.argsort(heat.ravel())[-5:], heat.shape))[0]
            for i, j in idx:
                ax.scatter((i/20)*w, (j/20)*h, color="yellow", s=100, edgecolors="black")

        # Paths
        if show_paths:
            path_df = path_df_full[path_df_full["map_id"] == map_name]
            path_df = map_coords(path_df, map_name, w, h)
            pos_df = path_df[path_df["event"].isin(["Position","BotPosition"])]
            for _, g in pos_df.groupby("user_id"):
                g = g.sort_values("ts")
                if len(g) > 1:
                    ax.plot(g["mx"], g["my"], color=PATH_COLORS[g["is_bot"].iloc[0]], alpha=0.6)

        # Events
        if show_points and len(data) > 0:
            df_points = map_coords(data, map_name, w, h)
            ax.scatter(df_points["mx"], df_points["my"],
                       c=df_points["event"].map(EVENT_COLORS), s=10)

        # Legend (clean)
        legend_items = []

        if show_paths:
            legend_items += [
                Line2D([0],[0],color='blue',label="Human Path"),
                Line2D([0],[0],color='orange',label="Bot Path")
            ]

        if show_heatmap:
            legend_items.append(Patch(facecolor='orange', alpha=0.5, label="Heatmap"))

        if show_hotspots:
            legend_items.append(Line2D([0],[0],marker='o',color='w',
                markerfacecolor='yellow',markeredgecolor='black',
                linestyle='None',label="Hotspots"))

        if show_points and len(data) > 0:
            for event in sorted(data["event"].unique()):
                legend_items.append(Line2D([0],[0],marker='o',color='w',
                    markerfacecolor=EVENT_COLORS.get(event,"white"),
                    linestyle='None',label=event))

        if legend_items:
            ax.legend(handles=legend_items, loc='upper left', fontsize=7)

        ax.set_xlim(0,w)
        ax.set_ylim(h,0)
        ax.axis("off")
        return fig

    if map_choice == "All Maps":
        for m in MAP_IMAGES:
            st.subheader(m)
            st.pyplot(plot_map(m, filtered_df[filtered_df["map_id"] == m]))
    else:
        st.pyplot(plot_map(map_choice, filtered_df))

# =========================
# PREDICTION (UNCHANGED + SAFE)
# =========================
with tab2:
    st.header("🤖 Player Prediction")

    player = st.selectbox("Player", df["user_id"].astype(str).unique())
    pdf = df[df["user_id"].astype(str) == player].sort_values("ts")

    if len(pdf) < 20:
        st.warning("Not enough data")
    else:
        coords = pdf[["x","z"]].dropna()
        events = pdf["event"].tolist()

        k = min(5, len(coords))
        kmeans = KMeans(n_clusters=k, n_init=10).fit(coords)
        centers = kmeans.cluster_centers_

        current = coords.iloc[-1].values.reshape(1,-1)
        preds = []

        for _ in range(3):
            c = kmeans.predict(current)[0]
            nxt = centers[(c+1)%k]
            preds.append(nxt)
            current = nxt.reshape(1,-1)

        st.metric("📏 Location Error",
                  round(np.linalg.norm(preds[0] - coords.iloc[-1].values),2))
