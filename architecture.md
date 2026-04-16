**LILA Player Analytics \& Prediction Tool**



Overview



This project is an interactive analytics and prediction tool designed to analyze player behavior in the LILA game environment. It enables exploration of player movement, event patterns, and includes a predictive module to estimate future player actions and trajectories.



The goal was to go beyond static visualization and build a system that combines data processing, spatial visualization, and lightweight predictive modeling in a clean, user-friendly interface.



\---



**Features**



**I. Visualization Module**



* Map-based visualization of player activity across multiple game maps
* Overlay of:

  * Player movement paths
  * Event markers (kills, loot, etc.)
* Supports multiple maps with different coordinate systems



\---



**II. Interactive Filters**



* Date range filter (supports single-day selection)
* Map selection (including "All Maps" view)
* Match ID filter
* Player type filter (Bot / Human / All)
* Player ID filter
* Event type filter



These filters allow users to drill down into specific gameplay scenarios and analyze behavior at different levels.



\---



**III. Player Behavior Tracking**



* Player paths are constructed using sequential position data
* Paths are color-coded:

  * Humans → Blue
  * Bots → Orange
* Designed to clearly differentiate behavioral patterns



\---



**IV. Prediction Module**



A separate tab introduces predictive capabilities:



a) Location Prediction



* Uses clustering (KMeans) on historical player coordinates
* Identifies spatial patterns in player movement
* Predicts future positions based on cluster transitions



b) Event Prediction



* Uses a rolling window of past events
* Predicts future events based on frequency patterns



c) Multi-step Forecasting



* Predicts the next 3 steps of:

  * Player movement
  * Player events



d) Visual Prediction Output



* Predictions are displayed directly on the map
* Includes:

  * Green dashed trajectory for predicted movement
  * Step-wise annotations (Step 1, Step 2, Step 3 with events)
  * Map name for context



e) Model Evaluation



To assess prediction quality:



* Event Prediction Accuracy (%)
* Location Prediction Error (Euclidean distance)



This ensures predictions are both interpretable and measurable.



\---



**V. Design Decisions**



a) Separation of Concerns



* Data Layer → Data loading and preprocessing
* Visualization Layer → Interactive filtering and rendering
* Prediction Layer → Independent modeling and evaluation



\---



b) Path vs Event Decoupling



* Player paths are independent of event filters
* Event filters only affect visible markers, not trajectory data



This ensures consistent movement visualization.



\---



**VI. UX Considerations**



* Sidebar used for global filters (persistent and intuitive)
* Tabs used to separate:

  * Exploration (Visualization)
  * Insights (Prediction)
* Clean layout for readability and usability



\---



**VII. Tech Stack**



* Python
* Streamlit
* Pandas
* NumPy
* Matplotlib
* Scikit-learn



\---



**VIII. Data Handling**



* Data loaded from partitioned parquet files
* Folder names parsed into datetime format
* Byte-encoded fields decoded
* Additional features derived:

  * Bot vs Human classification
  * Timestamp normalization



\---



**IX. How to Run**



```bash

pip install -r requirements.txt

streamlit run app.py

```



\---

---

## X. System Architecture

The system follows a modular architecture with clearly defined data flow across layers:

### 1. Data Ingestion Layer

* Reads partitioned parquet files from local storage
* Handles:

  * Multiple date folders
  * Byte decoding for event fields
  * Schema normalization

---

### 2. Data Processing Layer

* Transforms raw data into analysis-ready format
* Key transformations:

  * Timestamp parsing
  * Bot vs Human classification
  * Coordinate normalization (map-specific transformations)

---

### 3. Data Access Layer

* Provides filtered datasets based on user input:

  * Date
  * Map
  * Match
  * Player
  * Event type

This layer ensures efficient querying without recomputing the entire dataset.

---

### 4. Visualization Layer

* Renders spatial insights using map overlays
* Components:

  * Player paths
  * Event markers
  * Heatmaps
  * Hotspots

This layer consumes filtered data and transforms coordinates into pixel space.

---

### 5. Prediction Layer

* Independent module operating on player-specific data
* Sub-components:

  * Location prediction (KMeans clustering)
  * Event prediction (rolling window model)
  * Multi-step forecasting

---

### 6. Evaluation Layer

* Computes performance metrics:

  * Event prediction accuracy
  * Location error (Euclidean distance)

---

### 7. Presentation Layer (UI)

* Built using Streamlit
* Includes:

  * Sidebar filters
  * Tabs for separation of concerns
  * Interactive controls for toggling visualization layers

---

### Data Flow

Raw Data → Processing → Filtering → Visualization / Prediction → User Interface

This ensures a clean separation between computation and presentation.

---

## XI. Component Interaction

* User inputs from the UI dynamically update the Data Access Layer
* Filtered data is passed to:

  * Visualization Layer (for map rendering)
  * Prediction Layer (for model computation)
* Visualization and Prediction modules operate independently but share the same filtered dataset
* Outputs are rendered back to the UI in real time

This design ensures modularity and avoids tight coupling between components.

---

## XII. Scalability & Extensibility

While the current implementation uses local parquet files, the architecture can be extended as follows:

### Storage

* Replace local storage with:

  * Cloud storage (S3 / GCS)
  * Data warehouse (BigQuery / Snowflake)

### Backend APIs

* Introduce APIs to:

  * Serve filtered data
  * Run prediction models asynchronously

### Real-time Processing

* Integrate streaming systems (Kafka / PubSub) to process live game events

### Model Improvements

* Replace rule-based prediction with:

  * Sequence models (LSTM / Transformers)
  * Reinforcement learning for trajectory prediction

This ensures the system can scale from a prototype to a production-grade analytics platform.

---

## XIII. Insights Integration

The insights generated from player behavior analysis feed back into the system as:

* Game design recommendations:

  * Map restructuring
  * Incentive placement (loot, missions)

* Feature enhancements:

  * Heatmaps and hotspot visualization added to identify high-activity zones
  * Bot vs Human analysis for behavioral comparison

* Future modeling improvements:

  * Insights guide feature engineering for predictive models

This creates a feedback loop between analytics and game design, enabling data-driven decision making.

---



\---



**XIV. Summary**



This project demonstrates an end-to-end pipeline combining:



* Data processing
* Interactive visualization
* Behavioral analysis
* Predictive modeling
* Model evaluation



All within a single, user-friendly application.



\---



Sriya Suri

16-04-2026



