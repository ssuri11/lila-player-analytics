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
# UNZIP DATA
# =========================
if not os.path.exists(DATA_PATH) and os.path.exists(ZIP_PATH):
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(BASE_DIR)

# =========================
# LOAD DATA
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
if df.empty:
    st.error("No data found.")
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

        if heatmap_mode == "Humans Only":
            df_plot = df_plot[~df_plot["is_bot"]]
        elif heatmap_mode == "Bots Only":
            df_plot = df_plot[df_plot["is_bot"]]

        if show_heatmap and len(df_plot) > 0:
            heat, _, _ = np.histogram2d(df_plot["mx"], df_plot["my"], bins=60)
            ax.imshow(heat.T, extent=[0, w, h, 0], cmap="inferno", alpha=0.5)

        if show_hotspots and len(df_plot) > 0:
            heat, _, _ = np.histogram2d(df_plot["mx"], df_plot["my"], bins=20)
            idx = np.dstack(np.unravel_index(np.argsort(heat.ravel())[-5:], heat.shape))[0]
            for i, j in idx:
                ax.scatter((i / 20) * w, (j / 20) * h, color="yellow", s=100, edgecolors="black")

        if show_paths:
            path_df = path_df_full[path_df_full["map_id"] == map_name]
            path_df = map_coords(path_df, map_name, w, h)
            pos_df = path_df[path_df["event"].isin(["Position", "BotPosition"])]
            for _, g in pos_df.groupby("user_id"):
                g = g.sort_values("ts")
                if len(g) > 1:
                    ax.plot(g["mx"], g["my"], color=PATH_COLORS[g["is_bot"].iloc[0]], alpha=0.6)

        if show_points and len(data) > 0:
            df_points = map_coords(data, map_name, w, h)
            ax.scatter(df_points["mx"], df_points["my"], c=df_points["event"].map(EVENT_COLORS), s=10)

        # LEGEND
        handles, labels = [], []

        if show_paths:
            handles += [Line2D([0],[0],color='blue'), Line2D([0],[0],color='orange')]
            labels += ["Human Path","Bot Path"]

        if show_heatmap:
            handles.append(Patch(facecolor='red',alpha=0.5))
            labels.append("Heatmap")

        if show_hotspots:
            handles.append(Line2D([0],[0],marker='o',color='w',
                                  markerfacecolor='yellow',
                                  markeredgecolor='black',
                                  markersize=10,
                                  linestyle='None'))
            labels.append("Hotspots")

        if show_points and len(data) > 0:
            for event in sorted(data["event"].unique()):
                handles.append(Line2D([0],[0],marker='o',color='w',
                                      markerfacecolor=EVENT_COLORS.get(event,"white"),
                                      markersize=6,linestyle='None'))
                labels.append(event)

        if handles:
            ax.legend(handles, labels, loc='upper left', fontsize=7)

        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.axis("off")
        return fig

    if map_choice == "All Maps":
        for m in MAP_IMAGES:
            st.subheader(m)
            st.pyplot(plot_map(m, filtered_df[filtered_df["map_id"] == m]))
    else:
        st.pyplot(plot_map(map_choice, filtered_df))

# =========================
# PREDICTION (FIXED)
# =========================
with tab2:
    st.header("🤖 Player Prediction")

    player = st.selectbox("Player", df["user_id"].astype(str).unique())
    pdf = df[df["user_id"].astype(str) == player].sort_values("ts")

    if len(pdf) < 20:
        st.warning("Not enough data")
    else:
        coords = pdf[["x", "z"]].dropna()
        events = pdf["event"].tolist()

        k = min(5, len(coords))
        kmeans = KMeans(n_clusters=k, n_init=10).fit(coords)
        centers = kmeans.cluster_centers_

        current = coords.iloc[-1].values.reshape(1, -1)
        pred_locs = []

        for _ in range(3):
            c = kmeans.predict(current)[0]
            nxt = centers[(c + 1) % k]
            pred_locs.append(nxt)
            current = nxt.reshape(1, -1)

        # EVENT PREDICTION
        window = 10
        history = events[-window:].copy()
        pred_events = []

        for _ in range(3):
            pred = max(set(history), key=history.count)
            pred_events.append(pred)
            history.append(pred)

        # METRICS
        col1, col2 = st.columns(2)

        y_true, y_pred = [], []
        for i in range(window, len(events) - 1):
            hist = events[i-window:i]
            y_pred.append(max(set(hist), key=hist.count))
            y_true.append(events[i])

        if y_true:
            col1.metric("⚡ Event Accuracy", f"{round(accuracy_score(y_true,y_pred)*100,2)}%")
        else:
            col1.metric("⚡ Event Accuracy", "N/A")

        col2.metric("📏 Location Error",
                    round(np.linalg.norm(pred_locs[0] - coords.iloc[-1].values), 2))

        # PLOT
        map_name = pdf["map_id"].iloc[-1]
        st.subheader(f"🗺️ Map: {map_name}")

        img = Image.open(MAP_IMAGES[map_name])
        w, h = img.size
        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(img)

        def map_single(x,z):
            cfg = MAP_CONFIG[map_name]
            return ((x-cfg["origin_x"])/cfg["scale"])*w, (1-(z-cfg["origin_z"])/cfg["scale"])*h

        px, py = [], []

        for i, loc in enumerate(pred_locs):
            mx, my = map_single(loc[0], loc[1])
            px.append(mx)
            py.append(my)

            ax.text(mx, my,
                    f"Step {i+1}\n{pred_events[i]}",
                    color="white", fontsize=8,
                    bbox=dict(facecolor='green', alpha=0.6))

        ax.plot(px, py, "--o", color="green", label="Prediction")
        ax.legend()
        ax.axis("off")

        st.pyplot(fig)