import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import io
import json
import urllib.parse
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Define the list of allowed societies
ALLOWED_SOCIETIES = [
    'DHA Phase 6', 'Park View City', 'DHA Phase 7', 'Valencia', 'DHA Phase 2', 'Eden Boulevard', 
    'Eastern Housing', 'Tariq Gardens', 'Ferozepur Road', 'Allama Iqbal Town', 'DHA Phase 5', 
    'DHA 9 Town', 'Al-Ahmad Garden', 'Dream Gardens Phase 1', 'Dream Gardens Phase 2', 
    'New Lahore City', 'DHA Phase 1', 'Lake City', 'Formanites Housing', 'Central Park', 
    'Pak Arab Phase 1', 'Bahria Town', 'Palm City', 'DHA Phase 3', 'DHA Phase 8', 
    'OPF Housing Scheme', 'Punjab Coop', 'Gulberg', 'Garden Town', 'Eden Palace Villas', 
    'DHA Phase 4', 'Military Accounts Housing Society', 'Fort Villas', 'Bahria Orchard Phase 1', 
    'Mughalpura', 'Johar Town Phase 2', 'Divine Gardens', 'Elite Town', 'DHA 11 Rahbar', 
    'PAF Falcon Complex', 'PGECHS Phase 2', 'Wapda Town Phase 1', 'UET Housing', 'Sabzazar', 
    'Nasheman-e-Iqbal Phase 2', 'Faisal Town', 'Khuda Buksh Colony', 'Shadab Garden', 'Askari 5', 
    'Al Rehman Garden Phase 2', 'Township', 'Cantt', 'LDA Avenue', 'MM Alam Road', 
    'Khayaban-e-Amin', 'Airport Road', 'Lalazaar Garden Phase 1', 'IEP Engineers Town', 
    'Canal Bank', 'Iqbal Avenue', 'Lahore Press Club Housing Scheme', 'Johar Town Phase 1', 
    'Union Livings', 'New Canal Park', 'Harbanspura', 'Aamir Town', 'Canal Point', 
    'Mehar Fayaz Colony', 'Lahore Meadows City', 'NFC Phase 1', 'Model Town', 
    'Audit & Accounts Phase 1', 'Eden Residencia', 'Paragon City', 'Al Hafeez Garden Phase 1', 
    'Al Rehman Garden Phase 4', 'Jubilee Town', 'Defence Raya', 'Gulberg 2', 'Gulberg 3', 
    'Bismillah Housing', 'Palm Villas', 'Etihad Town Phase 1', 'Lalazaar Garden Phase 2', 
    'Hamza Town Phase 2', 'College Road', 'Sajid Garden', 'Nishtar Colony', 
    'Bahria Orchard Low Cost', 'Muslim Town', 'Askari 11', 'Nawab Town', 'Askari 10', 
    'Canal Garden', 'Al-Ghani Garden Phase 2', 'Green Town', 'Architects Engineers Housing Society', 
    'Aftab Garden', 'Al-Kabir Town Phase 2', 'Fazaia Phase 1', 'Airline Housing', 
    'Gulshan-e-Lahore', 'Cavalry Ground', 'Audit & Accounts Phase 1, Audit & Accounts Housing Society', 
    'Al-Kabir Town Phase 1', 'Bankers Avenue', 'Bedian Road', 'PCSIR Phase 2', 
    'Lahore Medical Phase 1', 'Izmir Town', 'Beacon House Society', 'NESPAK Phase 3', 
    'Bahria Nasheman', 'Tajpura', 'Lahore Motorway City', 'Nasheman-e-Iqbal Phase 1', 
    'PIA Housing Scheme', 'Al Haram Garden', 'Mian Amiruddin Park', 'OLC', 'Zaman Park', 
    'Halloki Gardens', 'T & T Aabpara Housing Society', 'Pine Avenue', 'Sanda', 
    'Penta Square', 'Chinar Bagh', 'Punjab University Employees Society', 'Bankers Housing', 
    'Mustafa Town', 'Thokar Niaz Baig', 'Pico Road', 'Sultan Town', 'Bahria Nasheman, Lahore', 
    'EME Society', 'Al-Hafiz Town', 'High Court Society', 'AWT Phase 2', 'Air Avenue', 
    'Garrison Homes', 'Royal Residencia', 'Agrics Town', 'Al Jalil Garden', 'Raiwind Road', 
    'IBL Housing', 'Green Cap', 'Rehan Garden Phase 1', 'Government Employees Cooperative Housing Society', 
    'Kahna', 'Al Rehman Garden Phase 1', 'Shadman', 'Abid Majeed Road', 'PCSIR Phase 1', 
    'Aitchison Society', 'Walton Road', 'Green City', 'DHA EME', 'Al Falah Town', 
    'Punjab Small Industries', 'Al Raheem Garden', 'Manawan', 'Jallo', 'Al-Hadi Garden', 
    'Canal View', 'Vital Homes EE, Vital Homes Housing Scheme', 'Azam Gardens', 'Ali Park', 
    'Vital Homes DD, Vital Homes Housing Scheme', 'Eden Value Homes', 'Marghzar Officers Colony', 
    'Ghous Garden', 'Ashraf Garden', 'Bahria Orchard Phase 4', 
    'Eastern Housing - East Bay Commercial Hub, Eastern Housing Lahore, Wagha Town', 
    'Al Hamra Town', 'SJ Garden', 'Grand Avenues Housing Scheme', 'Samanabad', 'Shah Farid', 
    'Aabpara Coop Housing Society', 'Eden Canal Villas', 'SA Gardens Phase 2', 'Fazaia Phase 2', 
    'Kamahan Road', 'Fahad Town', 'Lahore Medical Housing Scheme Phase 3, Lahore Medical Housing Society', 
    'Vital Homes AA, Vital Homes Housing Scheme', 'Fateh Garh', 'Eden Avenue', 'Garhi Shahu', 
    'Tricon Village', 'Eden Lane Villas', 'Sami Town', 'Pak Arab Phase 2', 
    'Lahore Medical Housing Scheme Phase 2, Lahore Medical Housing Society', 'Rail Town', 
    'Gulgasht Colony Rustam', 'Sadaat Town', 'Shershah Colony', 'Abdalians', 
    'Sui Gas Society Phase 1', 'Sozo Town', 'Askari 3', 'DHA Phase 9', 'Ravi Gardens', 
    'TIP Housing', 'Sui Gas Society Phase 1, Sui Gas Housing Society', 'Dream Avenue', 
    'Maryam Town', 'Gulshan-e-Ravi', 'Eden City', 'Shah Jamal', 'Venus Housing', 
    'Al-Hamd Gardens', 'Ahbab Colony', 'Shaheen Colony', 'Gosha-e-Ahbab', 'Canal Valley', 
    'Askari 9', 'Rehan Garden Phase 2', 'Moeez Town', 'Shoukat Town', 'Fazlia Colony', 
    'Farooq Colony', 'Al-Hamd Park', 'Public Health', 'PGECHS Phase 1', 'Gajju Matah', 
    'BOR', 'Lahore Garden', 'P&D Housing', 'Wapda Town Phase 2', 'Eden Gardens', 
    'Al Noor Park', 'Lahore Smart City', 'Chughtai Garden', 'Chungi Amar Sadhu', 
    'Aashiana Road', 'Golf View Residencia - Phase 1, Golf View Residencia, Bahria Town', 
    'Manhala Road', 'Zaamin City Phase 1', 'Park Avenue', 'Sukh Chayn Gardens', 'Chauburji', 
    'Revenue Society', 'Altaf Colony', 'Islamapura', 'NFC Phase 2', 'GT Road', 
    'Zaheer Villas', 'HBFC Housing', 'Larechs Avenue', 'OLC 2', 'Mozang', 'Karbath', 
    'Abid Garden', 'Vital Homes', 'Bahria Orchard Phase 3', 'Sofia Farm Houses', 
    'Shahid Town', 'Al Hafeez Garden Phase 2', 'Edenabad, Eden, Lahore', 'Amina Park', 
    'Sheraz Town', 'Tech Society', 'Habib Homes', 'Ichhra', 'Mohlanwal', 'Awan Town', 
    'Taj Bagh', 'Etihad Town Phase 3', 'Sue-e-Asal', 'Al Kareem Garden', 'Gul-e-Damin', 
    'Multan Road', 'Sunny Park', 'Al Rehman Garden Phase 3', 'Imam Town', 
    'Al-Kareem Premier', 'Al Faisal Town', 'Rehman Gardens', 'Ulfat Green City', 
    'Judicial Colony', 'Madar-e-Millat Road', 'Medina Town', 'Shalimar Town', 
    'Al Kareem City', 'UMT Road', 'Rustam Park', 'Safari Garden', 'Rehmanpura', 
    'Yateem Khana Chowk', 'Canal Fort II', 'Rizwan Garden', 'Lawrence Road', 
    'New Iqbal Park', 'Ferozewala', 'New Chauburji Park', 
    'Valencia - Commercial Zone A1, Valencia Housing Society', 'Mohafiz Town Phase 2', 
    'Islam Nagar, Lahore', 'Saroba Gardens', 'LDA City Phase 1', 'Mia Fazal Deen', 
    'Clifton Colony', 'Begum Kot', 'Barkat Colony', 'New Super Town', 'Metro City', 
    'Ali View Garden', 'Lahore Villas', 'Atomic Energy Society', 'Kotli Abdur Rahman', 
    'Peco Road', 'Peer Colony', 'Hassan Town', 'Shalimar Link Road', 'Naz Town', 
    'Green Avenue', 'Khayaban-e-Quaid', 'Ali Housing', 'Subhan Allah Garden', 
    'Gulshan-e-Habib', 'Shah Khawar Town', 'Green Park', 'Passco Housing', 
    'Old Officers Colony', 'Sadat Coop', 'GCP Housing', 'New Samanabad', 
    'NESPAK Phase 2', 'Supreme Villas', 'Gulshan-e-Mustafa Housing Society', 
    'Shalimar Housing', 'Audit & Accounts Phase 2, Audit & Accounts Housing Society', 
    'Jora Pull', 'Lahore Medical Housing Scheme Phase 1, Lahore Medical Housing Society', 
    'Nazir Garden', 'Combo Colony', 'Upper Mall', 'PAF Colony', 
    'Millat Tractors Employees', 'Al-Ghani Garden Phase 3', 'Al-Ghani Garden Phase 1', 
    'New Garden Town', 'Al-Hamad Colony', 'Heir', 'Al Rehman Garden Phase 7', 
    'Salli Town', 'Royal Garden', 'Garrison Unique Villas', 'Gold Land Garden', 
    'Hajvery Housing', 'Shahzada New Abadi', 'Iqbal Park', 'Sheraz Villas', 
    'Super Town', 'Rewaz Garden', 'Defence Fort', 'Royal Enclave, Al Raheem Gardens Phase 5', 
    'Raj Garh', 'National Town', 'New Kashmir Park', 'Greenland Housing Scheme', 
    'Sahafi Colony', 'Dubai Town', 'Kahna Kacha', 'Afzal Park', 'Bedian Greens Farm', 
    'NESPAK Scheme Phase 2, NESPAK Housing Scheme', 'SA Gardens Phase 1', 
    'Samanzar Colony', 'Mehmood Booti', 'Kot Khawaja Saeed', 'Gulshan Ali Colony', 
    'Siddiqia Society', 'Nishat Colony', 'Madni Park', 'Madina Garden', 'Dubban Pura', 
    'Mateen Avenue', 'Spanish Villas', 'Temple Road', 'Indigo Canal Homes', 
    'Dawood Residency', 'Daroghewala', 'Shahdara', 'Sham Nagar', 'Ilyas Colony', 
    'Skyland Waterpark', 'Sufiabad', 'Mehboob Garden', 'Rehmat Colony', 
    'Al-Qayyum Garden', 'Sodiwal', 'Shama Road', 'Chung', 'Al Kabir Orchard', 
    'Ittifaq Town', 'Muslim Nagar', 'Jail Road', 'Edenabad', 'Bagh-e-Iram', 
    'Wahdat Road', 'Jazac City', 'Race Course Road', 'Islamabad Colony', 
    'Icon Valley', 'Scheme Mor', 'Omega Residencia', 'Firdous Market, Gulberg', 
    'Qadri Colony', 'Baghbanpura', 'Ismail Town', 'Bagh Gul Begum', 
    'Nadeem Shaheed Road', 'Islamia Park', 'Abbas', 'Zaman Colony', 
    'Sui Gas Society Phase 2, Sui Gas Housing Society', 'Emerald City', 
    'Mohafiz Town Phase 1', 'Salamatpura', 'Overseas Low Cost, Bahria Orchard Phase 2', 
    'Cantt View', 'Ashiana-e-Quaid', 'Makkah Colony', 'Canal Burg'
]

# Define TargetMeanEncoder (from training code)
class TargetMeanEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, target_col='Price', smoothing=20):
        self.cols = cols if cols else []
        self.target_col = target_col
        self.smoothing = smoothing
        self.mapping_ = {}
        self.global_mean_ = None

    def fit(self, X, y):
        self.global_mean_ = y.mean()
        for col in self.cols:
            dfm = pd.DataFrame({col: X[col].astype(str), 'y': y.values})
            stats = dfm.groupby(col)['y'].agg(['count', 'mean']).reset_index()
            counts = stats['count'].values
            means = stats['mean'].values
            smooth = (means * counts + self.smoothing * self.global_mean_) / (counts + self.smoothing)
            self.mapping_[col] = dict(zip(stats[col].astype(str), smooth))
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            X[col + "_te"] = X[col].astype(str).map(self.mapping_.get(col, {})).fillna(self.global_mean_)
        return X

# Load the model and encoder
model_data = joblib.load('best_house_price_pipeline_v3.pkl')
pipeline = model_data['pipeline']
te = model_data['te']
numeric_features = model_data['numeric_features']
cat_features_onehot = model_data['cat_features_onehot']
binary_features = model_data['binary_features']
interaction_features = model_data['interaction_features']

# Load and cache society statistics
@st.cache_data
def load_stats():
    # Define default values for 'Others' and 'DHA 11 Rahbar'
    fallback_defaults = {
        'Others': {
            'bedrooms': 3,
            'bathrooms': 3,
            'area_marla': 5,
            'Area_mean': 5 * 225,  # In square feet
            'kitchens': 1,
            'store_rooms': 0,
            'servant_quarters': 0,
            'avg_price_crore': 0.9,
            'area_min_marla': 1,  # Allow small plots
            'area_max_marla': 50,
            'bedrooms_min': 1,
            'bedrooms_max': 10,
            'bathrooms_min': 1,
            'bathrooms_max': 10
        },
        'DHA 11 Rahbar': {
            'bedrooms': 4.64,
            'bathrooms': 4.69,
            'area_marla': 7.38,
            'Area_mean': 7.38 * 225,  # In square feet
            'kitchens': 1.0,
            'store_rooms': 0.0,
            'servant_quarters': 0.0,
            'avg_price_crore': 2.75,
            'area_min_marla': 3,  # Adjusted to allow 3 Marla
            'area_max_marla': 28.8,
            'bedrooms_min': 1,
            'bedrooms_max': 10,
            'bathrooms_min': 1,
            'bathrooms_max': 6
        }
    }

    try:
        df = pd.read_csv('stats.csv')
        # Verify required columns
        required_cols = ['Society', 'Price_mean', 'Area_mean', 'Bedrooms_mean', 'Bathrooms_mean', 
                        'Kitchens_mean', 'Store Rooms_mean', 'Servant Quarters_mean',
                        'Price_min', 'Price_max', 'Area_min', 'Area_max', 
                        'Bedrooms_min', 'Bedrooms_max', 'Bathrooms_min', 'Bathrooms_max']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing columns in stats.csv: {missing_cols}. Using fallback defaults.")
            return pd.DataFrame(), fallback_defaults, list(fallback_defaults.keys())

        # Clean society names: remove suffixes, strip spaces, preserve case
        df['Society'] = df['Society'].str.replace(r',\s*LAHORE.*$|,\s*HOUSING.*$', '', regex=True, case=False).str.strip()
        # Normalize society names to match ALLOWED_SOCIETIES
        society_mapping = {
            'Dha 11 Rahbar': 'DHA 11 Rahbar',
            'Dha Phase 11 - Rahbar': 'DHA 11 Rahbar',
            'Dha Phase 11 Rahbar': 'DHA 11 Rahbar',
            'Tech Society': 'Tech Society',
            'TECH SOCIETY': 'Tech Society',
            'Audit & Accounts Phase 1, Audit & Accounts Housing Society': 'Audit & Accounts Phase 1',
            'Sui Gas Society Phase 1, Sui Gas Housing Society': 'Sui Gas Society Phase 1',
            'Sui Gas Society Phase 2, Sui Gas Housing Society': 'Sui Gas Society Phase 2',
            'Vital Homes Aa, Vital Homes Housing Scheme': 'Vital Homes AA',
            'Vital Homes Dd, Vital Homes Housing Scheme': 'Vital Homes DD',
            'Vital Homes Ee, Vital Homes Housing Scheme': 'Vital Homes EE',
            'Valencia - Commercial Zone A1, Valencia Housing Society': 'Valencia Commercial Zone A1',
            'Royal Enclave, Al Raheem Gardens Phase 5': 'Royal Enclave',
            'Eastern Housing - East Bay Commercial Hub, Eastern Housing Lahore, Wagha Town': 'Eastern Housing',
            'Lahore Medical Housing Scheme Phase 1, Lahore Medical Housing Society': 'Lahore Medical Phase 1',
            'Lahore Medical Housing Scheme Phase 2, Lahore Medical Housing Society': 'Lahore Medical Phase 2',
            'Lahore Medical Housing Scheme Phase 3, Lahore Medical Housing Society': 'Lahore Medical Phase 3',
            'Audit & Accounts Phase 2, Audit & Accounts Housing Society': 'Audit & Accounts Phase 2',
            'Bahria Nasheman, Lahore': 'Bahria Nasheman',
            'Edenabad, Eden, Lahore': 'Edenabad',
            'Golf View Residencia - Phase 1, Golf View Residencia, Bahria Town': 'Golf View Residencia Phase 1',
            'Nespak Scheme Phase 2, Nespak Housing Scheme': 'NESPAK Phase 2',
            'Overseas Low Cost, Bahria Orchard Phase 2': 'Bahria Orchard Low Cost'
        }
        df['Society'] = df['Society'].replace(society_mapping)
        # Filter to allowed societies
        df = df[df['Society'].isin(ALLOWED_SOCIETIES)]
        if df.empty:
            st.error("No societies from the allowed list found in stats.csv. Using fallback defaults.")
            return pd.DataFrame(), fallback_defaults, list(fallback_defaults.keys())

        # Deduplicate by taking mean of numeric columns
        numeric_cols = ['Price_mean', 'Area_mean', 'Bedrooms_mean', 'Bathrooms_mean', 
                        'Kitchens_mean', 'Store Rooms_mean', 'Servant Quarters_mean',
                        'Price_min', 'Price_max', 'Area_min', 'Area_max', 
                        'Bedrooms_min', 'Bedrooms_max', 'Bathrooms_min', 'Bathrooms_max']
        df = df.groupby('Society')[numeric_cols].mean().reset_index()

        # Create defaults dictionary
        defaults = {
            row['Society']: {
                'bedrooms': row['Bedrooms_mean'],
                'bathrooms': row['Bathrooms_mean'],
                'area_marla': row['Area_mean'] / 225,
                'Area_mean': row['Area_mean'],  # In square feet
                'kitchens': row['Kitchens_mean'],
                'store_rooms': row['Store Rooms_mean'],
                'servant_quarters': row['Servant Quarters_mean'],
                'avg_price_crore': row['Price_mean'] / 1e7,
                'area_min_marla': row['Area_min'] / 225,
                'area_max_marla': row['Area_max'] / 225,
                'bedrooms_min': row['Bedrooms_min'],
                'bedrooms_max': row['Bedrooms_max'],
                'bathrooms_min': row['Bathrooms_min'],
                'bathrooms_max': row['Bathrooms_max']
            } for _, row in df.iterrows()
        }

        # Ensure 'Others' and 'DHA 11 Rahbar' are included
        defaults.update(fallback_defaults)
        societies = sorted(list(set(df['Society'].tolist()) | {'Others', 'DHA 11 Rahbar'}))

        # Debug: Check for high area_min_marla
        high_min_societies = [s for s, d in defaults.items() if d['area_min_marla'] > 3]
        if high_min_societies:
            st.warning(f"Societies with area_min_marla > 3 Marla: {', '.join(high_min_societies)}. Check stats.csv for accuracy.")
        
        # Debug: Verify defaults and societies
        if 'Others' not in defaults:
            st.error("Failed to add 'Others' to defaults.")
        if 'DHA 11 Rahbar' not in defaults:
            st.warning("DHA 11 Rahbar not found in stats.csv. Using fallback.")
        if not all('Area_mean' in defaults[s] for s in defaults):
            st.error("Some societies in defaults are missing 'Area_mean'.")
        st.write("Available societies:", societies)
        return df, defaults, societies

    except Exception as e:
        st.error(f"Error loading stats.csv: {e}. Using fallback defaults.")
        return pd.DataFrame(), fallback_defaults, list(fallback_defaults.keys())

stats_df, defaults, societies = load_stats()

# Load precomputed coordinates (mocked)
@st.cache_data
def load_coordinates():
    try:
        with open('coords.json', 'r') as f:
            return json.load(f)
    except:
        return {
            'DHA 11 Rahbar': [31.3925, 74.1353],
            'Audit & Accounts Phase 1': [31.4900, 74.4100],
            'Tech Society': [31.4700, 74.4500],
            'Others': [31.5204, 74.3587]
        }

coords = load_coordinates()

# CSS for styling
st.markdown("""
    <style>
    .stApp {
        background-image: url('house_bg.jpg');
        background-size: cover;
        background-attachment: fixed;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
    }
    .stSelectbox, .stNumberInput, .stSlider, .stExpander {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 10px;
        border-radius: 5px;
    }
    .stProgress > div > div {
        background-color: #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

# Theme toggle
theme = st.selectbox("Theme", ["Light", "Dark"], index=0)
if theme == "Dark":
    st.markdown("""
        <style>
        .stApp {
            background-color: #1E1E1E;
            color: white;
        }
        .stSelectbox, .stNumberInput, .stSlider, .stExpander {
            background-color: rgba(50, 50, 50, 0.9);
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

# Header
st.title("🏡 Estimate Your Dream Home Value in Lahore")
st.write("Predict house prices for properties in Lahore, Pakistan.")

# Progress meter
progress = 0
input_count = 0
total_inputs = 10

# Sidebar inputs
st.sidebar.header("Property Details")
with st.sidebar.expander("Basic Details"):
    default_society = societies[0] if societies else 'Others'
    help_text = f"Choose a society. E.g., {default_society}: {defaults[default_society]['area_min_marla']:.1f}–{defaults[default_society]['area_max_marla']:.1f} Marla."
    society = st.selectbox(
        "Select Society/Area Name in Lahore",
        options=societies,
        help=help_text
    )
    input_count += 1

    property_type = st.selectbox(
        "Property Type",
        options=['house', 'bungalow', 'villa', 'apartment', 'flat'],
        help="Select the type of property."
    )
    input_count += 1

    furnished = st.selectbox(
        "Furnished",
        options=["True", "False"],
        help="Is the property furnished?"
    )
    input_count += 1

    luxury = st.selectbox(
        "Luxury Property",
        options=["Yes", "No"],
        help="Does the property have luxury features? (Affects price)"
    )
    input_count += 1

    if luxury == "Yes":
        luxury_features_ux = st.multiselect(
            "Luxury Features (Optional, No Price Impact)",
            options=["Pool", "Gym", "Smart Home", "Garden", "Home Theater", "Others"],
            help="For personalization only."
        )
        input_count += 1

with st.sidebar.expander("Size and Layout"):
    area_unit = st.selectbox(
        "Area Unit",
        options=["Marla", "Kanal"],
        help="1 Marla ≈ 225 sq. ft, 1 Kanal ≈ 4500 sq. ft"
    )
    input_count += 1
    default_values = defaults.get(society, defaults['Others'])
    default_area = default_values['area_marla']
    area_size = st.number_input(
        f"Area in {area_unit}",
        min_value=0.1,
        max_value=50.0 if area_unit == "Marla" else 2.5,
        value=default_area if area_unit == "Marla" else default_area / 20,
        step=0.1,
        help=f"Typical in {society}: {default_area:.1f} Marla ({default_values['Area_mean']:.0f} sq.ft)."
    )
    input_count += 1
    area_sqft = area_size * 225 if area_unit == "Marla" else area_size * 4500

    # Validation for area (soft warnings, allow small sizes)
    area_min_marla = default_values['area_min_marla']
    area_max_marla = default_values['area_max_marla']
    if area_unit == "Marla":
        if area_size < area_min_marla:
            st.warning(f"{area_size:.1f} Marla is smaller than typical for {society} ({area_min_marla:.1f}–{area_max_marla:.1f} Marla), but you can proceed.")
        elif area_size > area_max_marla:
            st.warning(f"{area_size:.1f} Marla is larger than typical for {society} ({area_min_marla:.1f}–{area_max_marla:.1f} Marla).")
    else:  # Kanal
        if area_size < area_min_marla / 20:
            st.warning(f"{area_size:.1f} Kanal is smaller than typical for {society} ({area_min_marla / 20:.1f}–{area_max_marla / 20:.1f} Kanal), but you can proceed.")
        elif area_size > area_max_marla / 20:
            st.warning(f"{area_size:.1f} Kanal is larger than typical for {society} ({area_min_marla / 20:.1f}–{area_max_marla / 20:.1f} Kanal).")

    bedrooms = st.slider(
        "Bedrooms",
        min_value=1,  # Allow any number of bedrooms
        max_value=10,
        value=int(default_values['bedrooms']),
        help=f"Typical in {society}: {default_values['bedrooms']:.1f} ({default_values['bedrooms_min']}–{default_values['bedrooms_max']})."
    )
    input_count += 1

    bathrooms = st.slider(
        "Bathrooms",
        min_value=1,  # Allow any number of bathrooms
        max_value=10,
        value=int(min(default_values['bathrooms'], bedrooms + 1)),
        help=f"Typical in {society}: {default_values['bathrooms']:.1f} ({default_values['bathrooms_min']}–{default_values['bathrooms_max']})."
    )
    input_count += 1

    kitchens = st.slider(
        "Kitchens",
        min_value=1,
        max_value=5,
        value=int(default_values['kitchens']),
        help=f"Typical in {society}: {default_values['kitchens']:.1f}."
    )
    input_count += 1

    store_rooms = st.slider(
        "Store Rooms",
        min_value=0,
        max_value=5,
        value=int(default_values['store_rooms']),
        help=f"Typical in {society}: {default_values['store_rooms']:.1f}."
    )

    servant_quarters = 0
    if property_type in ['villa', 'bungalow']:
        servant_quarters = st.slider(
            "Servant Quarters",
            min_value=0,
            max_value=5,
            value=int(default_values['servant_quarters']),
            help=f"Typical in {society}: {default_values['servant_quarters']:.1f}."
        )

    prime_location = st.selectbox(
        "Prime Location",
        options=["Prime", "Not Prime"],
        help="Is the property in a prime location within the society?"
    )

# Validation
if bedrooms < bathrooms - 2:
    st.warning("Unusual: More bathrooms than bedrooms. Please check.")
if kitchens > bedrooms:
    st.warning("Unusual: More kitchens than bedrooms. Please check.")
if area_size > default_values['area_marla'] * 2:
    st.warning(f"Unusual: {area_size:.1f} {area_unit} is significantly larger than typical for {society} ({default_values['area_marla']:.1f} Marla).")

# Progress meter
progress = input_count / total_inputs
st.sidebar.progress(progress)
st.sidebar.write(f"Progress: {int(progress * 100)}% Complete")

# Prepare input for model
input_data = pd.DataFrame({
    'Society': [society],
    'Type': [property_type],
    'Area': [area_sqft],
    'Bedrooms': [bedrooms],
    'Bathrooms': [bathrooms],
    'Kitchens': [kitchens],
    'Store Rooms': [store_rooms],
    'Servant Quarters': [servant_quarters],
    'Furnished': [1 if furnished == "True" else 0],
    'Prime_Location': [1 if prime_location == "Prime" else 0],
    'Luxury_Features': [1 if luxury == "Yes" else 0]
})

# Add interaction features
input_data['Area_per_Bedroom'] = input_data['Area'] / input_data['Bedrooms'].replace(0, np.nan)
input_data['Area_per_Bedroom'] = input_data['Area_per_Bedroom'].fillna(input_data['Area_per_Bedroom'].median())
input_data['Area_x_Luxury'] = input_data['Area'] * input_data['Luxury_Features']
input_data['Bed_x_Bath'] = input_data['Bedrooms'] * input_data['Bathrooms']

# Apply target mean encoding
input_data = te.transform(input_data)

# Ensure correct feature order
input_features = numeric_features + cat_features_onehot + binary_features
input_data = input_data[input_features]

# Predict price
@st.cache_data
def predict_price(input_data):
    return pipeline.predict(input_data)[0]

predicted_price = predict_price(input_data)
predicted_price_crore = predicted_price / 1e7
predicted_price_lacs = predicted_price / 1e5
predicted_price_million = predicted_price / 1e6
confidence_interval = (predicted_price_crore * 0.8, predicted_price_crore * 1.2)
avg_price_crore = default_values['avg_price_crore']

# Price display
price_display = f"{predicted_price_crore:.2f} Crore ({predicted_price_million:.2f} million)"
if predicted_price_crore < 1:
    price_display = f"{predicted_price_lacs:.2f} Lacs ({predicted_price_million:.2f} million)"
price_color = "green" if predicted_price_crore < avg_price_crore * 0.5 else "orange" if predicted_price_crore < avg_price_crore * 1.5 else "red"

# Layout with columns
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown(f"<h3 style='color: {price_color};'>Estimated Price: {price_display}</h3>", unsafe_allow_html=True)
    st.write(f"Confidence Range: {confidence_interval[0]:.2f}–{confidence_interval[1]:.2f} Crore")
    st.info(f"💡 Adding a bedroom typically increases price by ~10% in {society}.")

# Feature Importance
@st.cache_data
def get_feature_importance():
    try:
        feature_names = pipeline.named_steps['preprocess'].get_feature_names_out()
        importances = pipeline.named_steps['model'].feature_importances_
        importance_dict = {name: imp for name, imp in zip(feature_names, importances)}
        agg_importance = {
            'Area': importance_dict.get('num__Area', 0),
            'Bedrooms': importance_dict.get('num__Bedrooms', 0),
            'Bathrooms': importance_dict.get('num__Bathrooms', 0),
            'Kitchens': importance_dict.get('num__Kitchens', 0),
            'Store Rooms': importance_dict.get('num__Store Rooms', 0),
            'Servant Quarters': importance_dict.get('num__Servant Quarters', 0),
            'Area_per_Bedroom': importance_dict.get('num__Area_per_Bedroom', 0),
            'Area_x_Luxury': importance_dict.get('num__Area_x_Luxury', 0),
            'Bed_x_Bath': importance_dict.get('num__Bed_x_Bath', 0),
            'Society': importance_dict.get('num__Society_te', 0),
            'Type': sum(imp for name, imp in importance_dict.items() if name.startswith('cat__Type')),
            'Furnished': importance_dict.get('remainder__Furnished', 0),
            'Prime_Location': importance_dict.get('remainder__Prime_Location', 0),
            'Luxury_Features': importance_dict.get('remainder__Luxury_Features', 0)
        }
        return agg_importance
    except:
        return {
            'Area': 0.4,
            'Bedrooms': 0.2,
            'Bathrooms': 0.15,
            'Luxury_Features': 0.15,
            'Prime_Location': 0.1,
            'Society': 0.05,
            'Area_per_Bedroom': 0.05,
            'Area_x_Luxury': 0.05,
            'Bed_x_Bath': 0.05
        }

feature_importance = get_feature_importance()
fig, ax = plt.subplots()
bars = sns.barplot(x=list(feature_importance.values()), y=list(feature_importance.keys()), ax=ax)
ax.set_title("Feature Impact on Price")
for i, bar in enumerate(bars.patches):
    bar.set_label(f"+1 unit → +{int(list(feature_importance.values())[i] * 100)}% price")
st.pyplot(fig)

# Comparison Chart
status = "Above Average" if predicted_price_crore > avg_price_crore else "Below Average" if predicted_price_crore < avg_price_crore else "Average"
scale = st.selectbox("Price Chart Scale", ["Linear", "Log"], index=0)
fig, ax = plt.subplots(figsize=(6, 2))
bars = sns.barplot(x=[predicted_price_crore, avg_price_crore], y=['Your Property', f'{society} Average'], ax=ax)
ax.set_xlabel("Price (Crore)")
ax.set_xscale('log' if scale == "Log" else 'linear')
for bar in bars.patches:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.2f} Cr", ha='center', va='center', color='white', fontsize=10)
st.pyplot(fig)
st.write(f"Your property is {status} for {society} (Avg: {avg_price_crore:.2f} Crore).")

# Simplified Map
with col2:
    st.subheader(f"Location of {society}")
    lat, lon = coords.get(society, [31.5204, 74.3587])
    if society not in coords:
        st.warning(f"Coordinates for {society} not found. Using default Lahore location.")
    st.write(f"Latitude: {lat:.4f}, Longitude: {lon:.4f}")
    st.map(pd.DataFrame([[lat, lon]], columns=["lat", "lon"]), zoom=12)
    st.write("© OpenStreetMap contributors")

# Social Sharing
with st.expander("Share Your Dream Home"):
    def create_share_image():
        img = Image.open('house_bg.jpg').resize((400, 300))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 18)
            logo_font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()
            logo_font = ImageFont.load_default()

        draw.rectangle((0, 0, 400, 300), fill=(0, 0, 0, 100))
        draw.text((10, 10), "DreamHomeEstimator", fill="white", font=logo_font)
        draw.text((10, 30), f"{society}, Lahore", fill="white", font=font)
        draw.text((10, 50), f"Price: {price_display}", fill="white", font=font)
        draw.text((10, 80), f"🛏️ {bedrooms} Beds, 🛁 {bathrooms} Baths", fill="white", font=font)
        draw.text((10, 100), f"📏 {area_size:.1f} {area_unit}", fill="white", font=font)
        draw.text((10, 120), f"🏠 {property_type.capitalize()}, Furnished: {furnished}", fill="white", font=font)
        draw.text((10, 140), f"Luxury: {luxury}", fill="white", font=font)
        draw.text((10, 160), f"via DreamHomeEstimator App", fill="white", font=logo_font)

        max_bar_width = 90
        max_price = max(predicted_price_crore, avg_price_crore) * 2
        user_width = min(max_bar_width, (predicted_price_crore / max_price) * max_bar_width if scale == "Linear" else (np.log1p(predicted_price_crore) / np.log1p(max_price)) * max_bar_width)
        avg_width = min(max_bar_width, (avg_price_crore / max_price) * max_bar_width if scale == "Linear" else (np.log1p(avg_price_crore) / np.log1p(max_price)) * max_bar_width)
        draw.rectangle((10, 180, 190, 200), fill="gray")
        draw.rectangle((10, 180, 10 + user_width, 190), fill="blue")
        draw.rectangle((100, 190, 100 + avg_width, 200), fill="green")
        draw.text((10, 205), f"Your Property ({predicted_price_crore:.2f} Cr) vs Avg ({avg_price_crore:.2f} Cr)", fill="white", font=ImageFont.load_default())

        qr = qrcode.QRCode(box_size=2)
        qr.add_data("https://dreamhomeestimator.com")
        qr.make()
        qr_img = qr.make_image(fill_color="black", back_color="white").resize((50, 50))
        img.paste(qr_img, (340, 240))
        return img

    if st.button("Generate Shareable Image"):
        img = create_share_image()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        st.image(img, caption="Your Dream Home Estimate")
        st.download_button("Download Image", data=buf.getvalue(), file_name="dream_home.png", mime="image/png")

    caption = f"Check out my dream home in {society}, Lahore: {price_display}! 🛏️ {bedrooms} Beds, 🛁 {bathrooms} Baths, 📏 {area_size:.1f} {area_unit}. What's yours? via DreamHomeEstimator"
    encoded_caption = urllib.parse.quote(caption)
    st.markdown(f"""
        Share on:
        - [Twitter/X](https://twitter.com/intent/tweet?text={encoded_caption})
        - [WhatsApp](https://wa.me/?text={encoded_caption})
        - [Facebook](https://www.facebook.com/sharer/sharer.php?u=https://dreamhomeestimator.com&quote={encoded_caption})
        - [LinkedIn](https://www.linkedin.com/sharing/share-offsite/?url=https://dreamhomeestimator.com&title={encoded_caption})
    """)

# Alternatives
st.subheader("Explore Alternatives")
st.write(f"Cheaper: {max(1, bedrooms-1)}-bed {property_type} in {society}: ~{(predicted_price_crore * 0.8):.2f} Crore")
st.write(f"Premium: {min(10, bedrooms+1)}-bed {property_type} in {society}: ~{(predicted_price_crore * 1.2):.2f} Crore")

# Log luxury features
if luxury == "Yes" and luxury_features_ux:
    st.write(f"Selected Luxury Features (for analytics, no price impact): {', '.join(luxury_features_ux)}")