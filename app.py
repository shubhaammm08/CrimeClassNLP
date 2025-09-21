import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Crime Classification Predictor",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-result {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
    st.session_state.models = {}
    st.session_state.tfidf = None
    st.session_state.df = None
    st.session_state.training_complete = False

# Helper Functions
@st.cache_data
def load_and_process_data(folder_path):
    """Load and process CSV files"""
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not csv_files:
        st.error(f"No CSV files found in {folder_path}")
        return None
    
    dataframes = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, file in enumerate(csv_files):
        try:
            status_text.text(f"Loading file {i+1}/{len(csv_files)}: {os.path.basename(file)}")
            df = pd.read_csv(file, low_memory=False)
            if len(df) > 0:
                dataframes.append(df)
            progress_bar.progress((i + 1) / len(csv_files))
        except Exception as e:
            st.warning(f"Skipped {os.path.basename(file)}: {str(e)}")
    
    progress_bar.empty()
    status_text.empty()
    
    if not dataframes:
        st.error("No valid dataframes loaded")
        return None
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

def clean_and_standardize_data(df):
    """Clean and standardize the dataset"""
    # Standardize state names
    if 'STATE/UT' in df.columns:
        df['STATE/UT'] = df['STATE/UT'].astype(str).str.upper().str.strip()
        df = df[~df['STATE/UT'].str.contains('TOTAL|Total|ALL-INDIA|All-India', na=False)]
    
    # Standardize district names
    if 'DISTRICT' in df.columns:
        df['DISTRICT'] = df['DISTRICT'].astype(str).str.upper().str.strip()
        df = df[~df['DISTRICT'].str.contains('TOTAL|Total', na=False)]
    
    # Remove rows with missing key columns
    df = df.dropna(subset=['STATE/UT', 'DISTRICT']).reset_index(drop=True)
    
    return df

def select_major_crime_features(df):
    """Select major crime categories"""
    major_crime_patterns = [
        'MURDER', 'RAPE', 'KIDNAPPING', 'DACOITY', 'ROBBERY', 
        'BURGLARY', 'THEFT', 'RIOTS', 'CHEATING', 'ARSON',
        'HURT', 'DOWRY', 'ASSAULT', 'FRAUD', 'EXTORTION'
    ]
    
    selected_columns = []
    for col in df.columns:
        col_upper = col.upper()
        if any(crime in col_upper for crime in major_crime_patterns):
            if not any(skip in col_upper for skip in ['TOTAL', 'GRAND', 'SECTION']):
                selected_columns.append(col)
    
    return selected_columns

def create_crime_description(row, crime_columns):
    """Create text description from crime data"""
    descriptions = []
    
    # Add district identifier
    district = str(row['DISTRICT']).lower().replace(' ', '_').replace('.', '').replace('(', '').replace(')', '')
    descriptions.append(f"district_{district}")
    
    # Add year if available
    if 'YEAR' in row and not pd.isna(row['YEAR']):
        year = int(row['YEAR'])
        if year < 2005:
            descriptions.append("period_early_2000s")
        elif year < 2010:
            descriptions.append("period_mid_2000s")
        else:
            descriptions.append("period_recent")
    
    # Process crime counts
    crime_totals = []
    for col in crime_columns:
        if col in row and not pd.isna(row[col]):
            count = max(0, int(row[col]))
            if count > 0:
                crime_totals.append((col, count))
    
    crime_totals.sort(key=lambda x: x[1], reverse=True)
    total_crimes = sum(count for _, count in crime_totals) or 1
    
    for crime_type, count in crime_totals:
        if count > 0:
            crime_name = crime_type.lower().replace(' ', '_').replace('/', '_').replace('&', 'and')
            frequency_ratio = count / total_crimes
            
            if frequency_ratio > 0.3:
                repetitions = min(20, max(5, int(count / 50)))
            elif frequency_ratio > 0.1:
                repetitions = min(10, max(2, int(count / 100)))
            elif count > 10:
                repetitions = min(5, max(1, int(count / 200)))
            else:
                repetitions = 1
            
            descriptions.extend([f"crime_{crime_name}"] * repetitions)
    
    return ' '.join(descriptions)

def train_models(X_train, y_train, X_test, y_test):
    """Train multiple models and return results"""
    models = {
        'Logistic Regression': LogisticRegression(
            random_state=42, max_iter=1000, class_weight='balanced', C=1.0
        ),
        'Random Forest': RandomForestClassifier(
            random_state=42, n_estimators=100, class_weight='balanced', max_depth=20
        ),
        'SVM (Linear)': SVC(
            random_state=42, kernel='linear', class_weight='balanced', probability=True
        )
    }
    
    model_results = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (model_name, model) in enumerate(models.items()):
        status_text.text(f"Training {model_name}...")
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        model_results[model_name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred
        }
        
        progress_bar.progress((i + 1) / len(models))
    
    progress_bar.empty()
    status_text.empty()
    
    return model_results

# Main App
def main():
    st.markdown('<h1 class="main-header">🚔 Crime Classification Predictor</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Data Loading & Training", "Model Prediction", "Data Analysis"])
    
    if page == "Data Loading & Training":
        data_loading_page()
    elif page == "Model Prediction":
        prediction_page()
    else:
        analysis_page()

def data_loading_page():
    st.header("📊 Data Loading & Model Training")
    
    # File path input
    folder_path = st.text_input(
        "Enter the path to your crime data folder:",
        value=r"C:\Users\ssk08\OneDrive\Desktop\NLP Project\Model 2\crime",
        help="Path to the folder containing CSV files"
    )
    
    if st.button("Load Data and Train Models", type="primary"):
        if not os.path.exists(folder_path):
            st.error("Path does not exist. Please check the folder path.")
            return
        
        with st.spinner("Loading and processing data..."):
            # Load data
            df = load_and_process_data(folder_path)
            if df is None:
                return
            
            # Clean data
            df = clean_and_standardize_data(df)
            
            # Filter states with sufficient data
            min_samples = st.sidebar.slider("Minimum samples per state", 20, 100, 50)
            state_counts = df['STATE/UT'].value_counts()
            valid_states = state_counts[state_counts >= min_samples].index
            df = df[df['STATE/UT'].isin(valid_states)].reset_index(drop=True)
            
            st.success(f"Loaded {len(df)} samples across {df['STATE/UT'].nunique()} states")
            
            # Display data overview
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Unique States", df['STATE/UT'].nunique())
            with col3:
                st.metric("Unique Districts", df['DISTRICT'].nunique())
            
            # Feature engineering
            crime_columns = select_major_crime_features(df)
            for col in crime_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            df['crime_text'] = df.apply(lambda row: create_crime_description(row, crime_columns), axis=1)
            df = df[df['crime_text'].str.len() > 10].reset_index(drop=True)
            
            # Split data
            X = df['crime_text']
            y = df['STATE/UT']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # TF-IDF Vectorization
            st.info("Creating TF-IDF features...")
            tfidf = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 3),
                min_df=3,
                max_df=0.8,
                stop_words='english',
                sublinear_tf=True,
                norm='l2'
            )
            
            X_train_tfidf = tfidf.fit_transform(X_train)
            X_test_tfidf = tfidf.transform(X_test)
            
            # Train models
            st.info("Training models...")
            model_results = train_models(X_train_tfidf, y_train, X_test_tfidf, y_test)
            
            # Store in session state
            st.session_state.models = model_results
            st.session_state.tfidf = tfidf
            st.session_state.df = df
            st.session_state.y_test = y_test
            st.session_state.model_trained = True
            st.session_state.training_complete = True
            
            # Display results
            st.subheader("Model Performance")
            results_df = pd.DataFrame([
                {"Model": name, "Accuracy": f"{results['accuracy']:.4f}", "Accuracy %": f"{results['accuracy']*100:.2f}%"}
                for name, results in model_results.items()
            ])
            st.dataframe(results_df, use_container_width=True)
            
            # Find best model
            best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['accuracy'])
            best_accuracy = model_results[best_model_name]['accuracy']
            
            st.success(f"🏆 Best Model: {best_model_name} with {best_accuracy:.4f} accuracy ({best_accuracy*100:.2f}%)")
            
            # Visualization
            fig = px.bar(
                results_df, 
                x="Model", 
                y="Accuracy %", 
                title="Model Performance Comparison",
                color="Model"
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

def prediction_page():
    st.header("🔮 Crime Pattern Prediction")
    
    if not st.session_state.model_trained:
        st.warning("Please train the models first in the 'Data Loading & Training' page.")
        return
    
    # Model selection
    model_names = list(st.session_state.models.keys())
    selected_model = st.selectbox("Choose a model:", model_names)
    
    model = st.session_state.models[selected_model]['model']
    tfidf = st.session_state.tfidf
    
    # Input methods
    input_method = st.radio("Choose input method:", ["Manual Input", "Example Patterns"])
    
    if input_method == "Manual Input":
        st.subheader("Enter Crime Pattern")
        st.info("""
        **Tips for better predictions:**
        - Include district: `district_mumbai`
        - Add crimes: `crime_murder crime_theft crime_robbery`
        - Add time period: `period_recent` or `period_early_2000s`
        
        **Example:** `district_bangalore crime_theft crime_cheating crime_fraud period_recent`
        """)
        
        user_input = st.text_area(
            "Crime Pattern:",
            placeholder="district_mumbai crime_theft crime_burglary crime_cheating period_recent",
            height=100
        )
        
        if st.button("Predict State", type="primary"):
            if user_input.strip():
                prediction, confidence, probabilities = predict_state(user_input, model, tfidf)
                display_prediction_results(prediction, confidence, probabilities, selected_model)
            else:
                st.error("Please enter a crime pattern.")
    
    else:  # Example patterns
        st.subheader("Select Example Pattern")
        examples = {
            "Mumbai - High theft and fraud": "district_mumbai crime_theft crime_theft crime_burglary crime_cheating crime_fraud period_recent",
            "Delhi - Violent crimes": "district_delhi crime_murder crime_rape crime_kidnapping crime_theft crime_riots period_recent",
            "Chennai - Domestic violence": "district_chennai crime_dowry_deaths crime_assault crime_cruelty_by_husband period_recent",
            "Kolkata - Property crimes": "district_kolkata crime_theft crime_burglary crime_cheating period_mid_2000s",
            "Rural area - Traditional crimes": "district_rural crime_murder crime_kidnapping crime_dacoity period_early_2000s"
        }
        
        selected_example = st.selectbox("Choose an example:", list(examples.keys()))
        example_pattern = examples[selected_example]
        
        st.text_area("Selected Pattern:", value=example_pattern, height=100, disabled=True)
        
        if st.button("Predict State for Example", type="primary"):
            prediction, confidence, probabilities = predict_state(example_pattern, model, tfidf)
            display_prediction_results(prediction, confidence, probabilities, selected_model)

def predict_state(crime_text, model, tfidf):
    """Make prediction for given crime pattern"""
    text_tfidf = tfidf.transform([crime_text])
    prediction = model.predict(text_tfidf)[0]
    
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(text_tfidf)[0]
        confidence = np.max(probabilities)
        classes = model.classes_
        prob_dict = dict(zip(classes, probabilities))
    else:
        confidence = 1.0
        prob_dict = {prediction: 1.0}
    
    return prediction, confidence, prob_dict

def display_prediction_results(prediction, confidence, probabilities, model_name):
    """Display prediction results with visualizations"""
    # Main prediction result
    st.markdown(f"""
    <div class="prediction-result">
        <h2 style="color: #1f77b4; margin-bottom: 1rem;">Prediction Results</h2>
        <p><strong>Model Used:</strong> {model_name}</p>
        <p><strong>Predicted State:</strong> <span style="font-size: 1.5em; color: #d62728;">{prediction}</span></p>
        <p><strong>Confidence Score:</strong> {confidence:.3f} ({confidence*100:.1f}%)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Top predictions table
    st.subheader("Top 5 State Predictions")
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:5]
    
    results_df = pd.DataFrame([
        {"Rank": i+1, "State": state, "Probability": f"{prob:.3f}", "Percentage": f"{prob*100:.1f}%"}
        for i, (state, prob) in enumerate(sorted_probs)
    ])
    
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    
    # Visualization
    if len(sorted_probs) > 1:
        fig = px.bar(
            x=[prob for _, prob in sorted_probs],
            y=[state for state, _ in sorted_probs],
            orientation='h',
            title="State Prediction Probabilities",
            labels={'x': 'Probability', 'y': 'State'},
            color=[prob for _, prob in sorted_probs],
            color_continuous_scale='viridis'
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

def analysis_page():
    st.header("📈 Data Analysis & Insights")
    
    if not st.session_state.model_trained:
        st.warning("Please train the models first in the 'Data Loading & Training' page.")
        return
    
    df = st.session_state.df
    models = st.session_state.models
    
    # Data overview
    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("States/UTs", df['STATE/UT'].nunique())
    with col3:
        st.metric("Districts", df['DISTRICT'].nunique())
    with col4:
        if 'YEAR' in df.columns:
            year_range = f"{df['YEAR'].min():.0f}-{df['YEAR'].max():.0f}"
            st.metric("Year Range", year_range)
    
    # State distribution
    st.subheader("State Distribution")
    state_counts = df['STATE/UT'].value_counts().head(15)
    
    fig = px.bar(
        x=state_counts.values,
        y=state_counts.index,
        orientation='h',
        title="Top 15 States by Number of Records",
        labels={'x': 'Number of Records', 'y': 'State/UT'}
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Model comparison
    st.subheader("Model Performance Analysis")
    performance_data = [
        {"Model": name, "Accuracy": results['accuracy']}
        for name, results in models.items()
    ]
    performance_df = pd.DataFrame(performance_data)
    
    fig = px.bar(
        performance_df,
        x="Model",
        y="Accuracy",
        title="Model Accuracy Comparison",
        color="Accuracy",
        color_continuous_scale='viridis'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Best model detailed analysis
    best_model_name = max(models.keys(), key=lambda k: models[k]['accuracy'])
    best_model_results = models[best_model_name]
    
    st.subheader(f"Detailed Analysis - {best_model_name}")
    
    # Classification report
    if 'y_test' in st.session_state:
        y_test = st.session_state.y_test
        y_pred = best_model_results['predictions']
        
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        
        # Extract per-class metrics
        class_metrics = []
        for state, metrics in report_dict.items():
            if isinstance(metrics, dict) and 'f1-score' in metrics:
                class_metrics.append({
                    'State': state,
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1-score'],
                    'Support': metrics['support']
                })
        
        if class_metrics:
            metrics_df = pd.DataFrame(class_metrics)
            metrics_df = metrics_df.sort_values('F1-Score', ascending=False)
            
            # Top and bottom performing states
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Top 10 Best Performing States:**")
                top_states = metrics_df.head(10)[['State', 'F1-Score']]
                st.dataframe(top_states, hide_index=True)
            
            with col2:
                st.write("**Bottom 10 States (Need Improvement):**")
                bottom_states = metrics_df.tail(10)[['State', 'F1-Score']]
                st.dataframe(bottom_states, hide_index=True)
            
            # F1-Score distribution
            fig = px.histogram(
                metrics_df,
                x='F1-Score',
                nbins=20,
                title='Distribution of F1-Scores Across States'
            )
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()