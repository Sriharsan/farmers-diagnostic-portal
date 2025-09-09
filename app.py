import streamlit as st
import pandas as pd
import numpy as np
import json
import yaml
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import base64
from io import BytesIO
import requests
import os
from dotenv import load_dotenv

load_dotenv()


# Page config
st.set_page_config(
    page_title="🌾 Farmers Disease Diagnostic Portal",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #2E8B57 0%, #32CD32 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background: #f0f9f0;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #2E8B57;
}
.diagnosis-result {
    background: #e8f5e8;
    padding: 1.5rem;
    border-radius: 10px;
    border: 1px solid #2E8B57;
    margin: 1rem 0;
}
.warning-box {
    background: #fff3cd;
    padding: 1rem;
    border-radius: 5px;
    border: 1px solid #ffeaa7;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'submissions' not in st.session_state:
    st.session_state.submissions = []
if 'user_remedies' not in st.session_state:
    st.session_state.user_remedies = []

# Load sample data
@st.cache_data
def load_sample_data():
    """Load sample disease data and remedies"""
    sample_diseases = {
        "plant": {
            "Tomato Late Blight": {
                "symptoms": ["Dark brown spots on leaves", "White fungal growth", "Fruit rotting"],
                "causes": "Phytophthora infestans fungus",
                "treatment": "Apply copper-based fungicide, remove affected parts",
                "prevention": "Ensure good air circulation, avoid overhead watering"
            },
            "Wheat Rust": {
                "symptoms": ["Orange-red pustules on leaves", "Yellowing of leaves", "Stunted growth"],
                "causes": "Puccinia fungal infection",
                "treatment": "Apply fungicide, use resistant varieties",
                "prevention": "Crop rotation, proper field sanitation"
            }
        },
        "livestock": {
            "Lumpy Skin Disease": {
                "symptoms": ["Skin nodules", "Fever", "Reduced milk production"],
                "causes": "Lumpy skin disease virus",
                "treatment": "Supportive care, antibiotics for secondary infections",
                "prevention": "Vaccination, vector control"
            },
            "Foot and Mouth Disease": {
                "symptoms": ["Blisters on mouth and feet", "Drooling", "Lameness"],
                "causes": "FMD virus",
                "treatment": "Isolation, supportive care",
                "prevention": "Vaccination, quarantine measures"
            }
        }
    }
    return sample_diseases


def get_weather_data(location):
    """Get current weather data for location"""
    api_key = os.getenv('OPENWEATHER_API_KEY')
    if not api_key:
        return None
    
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather"
        params = {
            'q': location,
            'appid': api_key,
            'units': 'metric'
        }
        response = requests.get(url, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            return {
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'description': data['weather'][0]['description'],
                'location': data['name']
            }
    except Exception as e:
        print(f"Weather API error: {e}")
    
    return None

def enhance_diagnosis_with_weather(disease, confidence, location):
    """Enhance diagnosis confidence based on weather conditions"""
    weather = get_weather_data(location)
    if not weather:
        return confidence, "No weather data available"
    
    # Disease-weather correlation rules
    weather_factors = {
        'Late_blight': {
            'high_humidity': (weather['humidity'] > 80, 1.2, "High humidity favors blight development"),
            'cool_temp': (10 <= weather['temperature'] <= 25, 1.15, "Cool temperatures increase blight risk")
        },
        'rust': {
            'moderate_humidity': (50 <= weather['humidity'] <= 70, 1.1, "Moderate humidity supports rust spores"),
            'moderate_temp': (15 <= weather['temperature'] <= 25, 1.1, "Ideal temperature for rust development")
        },
        'scab': {
            'wet_conditions': ('rain' in weather['description'].lower(), 1.25, "Wet conditions promote scab infection"),
            'spring_temp': (10 <= weather['temperature'] <= 20, 1.1, "Cool spring weather favors scab")
        }
    }
    
    adjusted_confidence = confidence
    weather_insight = f"Current weather: {weather['temperature']}°C, {weather['humidity']}% humidity"
    
    # Check if disease matches weather patterns
    for pattern_name, conditions in weather_factors.items():
        if pattern_name.lower() in disease.lower():
            for condition_name, (condition_met, multiplier, explanation) in conditions.items():
                if condition_met:
                    adjusted_confidence = min(adjusted_confidence * multiplier, 0.98)
                    weather_insight += f". {explanation}"
                    break
    
    return adjusted_confidence, weather_insight

# Replace the diagnosis section in "Disease Diagnosis" tab (around line 120)
# Find this part in your app.py and replace:

if st.button("🚀 Diagnose Disease", type="primary"):
    with st.spinner("🔄 AI analyzing image and symptoms..."):
        # AI Prediction
        ai_disease, confidence = predict_disease(image, category)
        
        # Get location for weather context
        location_input = st.session_state.get('user_location', 'Coimbatore, India')
        
        # Enhance with weather data
        adjusted_confidence, weather_insight = enhance_diagnosis_with_weather(
            ai_disease, confidence, location_input
        )
        
        # Knowledge-based reasoning
        symptom_list = [s.strip() for s in symptoms.split('\n') if s.strip()] if symptoms else []
        kb_matches = apply_knowledge_rules(symptom_list, category)
        
        st.markdown("""
        <div class="diagnosis-result">
            <h3>🎯 Diagnosis Results</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Display results
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            confidence_emoji = "🔥" if adjusted_confidence > 0.85 else "✅" if adjusted_confidence > 0.7 else "⚠️"
            st.metric(f"{confidence_emoji} AI Prediction", ai_disease, f"{adjusted_confidence:.1%} confidence")
            
        with result_col2:
            if kb_matches:
                st.metric("🧠 Knowledge Base Match", kb_matches[0][0], f"{kb_matches[0][1]:.1%} match")
        
        # Weather context
        if weather_insight:
            st.info(f"🌤️ **Weather Context**: {weather_insight}")
        
        # Show detailed info
        diseases = load_sample_data()
        disease_info = diseases[category].get(ai_disease, {})
        
        if disease_info:
            st.subheader("📋 Disease Information")
            st.write(f"**Cause:** {disease_info.get('causes', 'Unknown')}")
            st.write(f"**Treatment:** {disease_info.get('treatment', 'Consult expert')}")
            st.write(f"**Prevention:** {disease_info.get('prevention', 'Follow good practices')}")
            
            # Confidence indicator
            if adjusted_confidence > confidence:
                st.success("✅ Weather conditions support this diagnosis")
            elif adjusted_confidence < confidence:
                st.warning("⚠️ Weather conditions don't typically favor this disease")


# Mock AI model prediction
def predict_disease(image, category):
    """Mock AI prediction - replace with actual model"""
    diseases = load_sample_data()
    disease_names = list(diseases[category].keys())
    
    # Simulate prediction
    predicted_disease = np.random.choice(disease_names)
    confidence = np.random.uniform(0.75, 0.95)
    
    return predicted_disease, confidence

# Knowledge base reasoning
def apply_knowledge_rules(symptoms, category):
    """Apply rule-based reasoning for diagnosis"""
    diseases = load_sample_data()
    matches = []
    
    for disease, info in diseases[category].items():
        symptom_match = sum(1 for s in symptoms if any(keyword in s.lower() for keyword in info['symptoms'][0].lower().split()))
        if symptom_match > 0:
            matches.append((disease, symptom_match / len(info['symptoms'])))
    
    return sorted(matches, key=lambda x: x[1], reverse=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🌾 Farmers Disease Diagnostic & Reporting Portal</h1>
    <p>AI-Powered Diagnosis | Knowledge-Based Insights | Community Driven</p>
</div>
""", unsafe_allow_html=True)

# Replace the sidebar section (around line 87) with this enhanced version:

# Sidebar navigation
st.sidebar.title("🚀 Navigation")

# Location input
st.sidebar.subheader("📍 Your Location")
user_location = st.sidebar.text_input(
    "Enter your location:", 
    value=st.session_state.get('user_location', 'Coimbatore, Tamil Nadu'),
    help="For weather-based diagnosis enhancement"
)
if user_location:
    st.session_state['user_location'] = user_location

# Weather display
if user_location:
    weather_data = get_weather_data(user_location)
    if weather_data:
        st.sidebar.success(f"🌤️ {weather_data['temperature']}°C, {weather_data['humidity']}% humidity")
    else:
        st.sidebar.info("🌐 Weather data unavailable")

st.sidebar.markdown("---")

tab = st.sidebar.selectbox(
    "Choose Section:",
    ["🔬 Disease Diagnosis", "💊 Treatment Remedies", "📋 Report & Submit", "📊 Analytics Dashboard", "ℹ️ About"]
)

# Quick stats in sidebar
st.sidebar.subheader("📊 Quick Stats")
if st.session_state.submissions:
    st.sidebar.metric("Your Reports", len(st.session_state.submissions))
    st.sidebar.metric("Community Remedies", len(st.session_state.user_remedies))
else:
    st.sidebar.info("Start by reporting a disease case!")

# QR Code placeholder
st.sidebar.subheader("📱 Mobile Access")
st.sidebar.info("🔗 Share this URL for mobile access")
st.sidebar.code("your-app-url.streamlit.app", language=None)

# Tab 1: Disease Diagnosis
if tab == "🔬 Disease Diagnosis":
    st.header("🔬 AI-Powered Disease Diagnosis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image of affected crop/livestock:",
            type=['jpg', 'jpeg', 'png'],
            help="Upload clear images for better diagnosis"
        )
        
        category = st.selectbox(
            "Select Category:",
            ["plant", "livestock"],
            format_func=lambda x: "🌱 Plant/Crop" if x == "plant" else "🐄 Livestock"
        )
        
        st.subheader("📝 Describe Symptoms")
        symptoms = st.text_area(
            "List observed symptoms (one per line):",
            placeholder="e.g.,\nYellowing leaves\nBrown spots\nWilting"
        )
    
    with col2:
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("🚀 Diagnose Disease", type="primary"):
                with st.spinner("🔄 AI analyzing image and symptoms..."):
                    # AI Prediction
                    ai_disease, confidence = predict_disease(image, category)
                    
                    # Knowledge-based reasoning
                    symptom_list = [s.strip() for s in symptoms.split('\n') if s.strip()] if symptoms else []
                    kb_matches = apply_knowledge_rules(symptom_list, category)
                    
                    st.markdown("""
                    <div class="diagnosis-result">
                        <h3>🎯 Diagnosis Results</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display results
                    result_col1, result_col2 = st.columns(2)
                    
                    with result_col1:
                        st.metric("🤖 AI Prediction", ai_disease, f"{confidence:.1%} confidence")
                        
                    with result_col2:
                        if kb_matches:
                            st.metric("🧠 Knowledge Base Match", kb_matches[0][0], f"{kb_matches[0][1]:.1%} match")
                    
                    # Show detailed info
                    diseases = load_sample_data()
                    disease_info = diseases[category].get(ai_disease, {})
                    
                    if disease_info:
                        st.subheader("📋 Disease Information")
                        st.write(f"**Cause:** {disease_info.get('causes', 'Unknown')}")
                        st.write(f"**Treatment:** {disease_info.get('treatment', 'Consult expert')}")
                        st.write(f"**Prevention:** {disease_info.get('prevention', 'Follow good practices')}")

# Tab 2: Treatment Remedies
elif tab == "💊 Treatment Remedies":
    st.header("💊 Treatment Remedies & Solutions")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔍 Search Remedies")
        search_disease = st.text_input("Enter disease name or symptoms:")
        remedy_category = st.selectbox("Category:", ["plant", "livestock"])
        
        diseases = load_sample_data()
        
        if search_disease:
            st.subheader(f"📖 Remedies for: {search_disease}")
            # Simple search logic
            found_remedies = []
            for disease, info in diseases[remedy_category].items():
                if search_disease.lower() in disease.lower():
                    found_remedies.append((disease, info))
            
            if found_remedies:
                for disease, info in found_remedies:
                    with st.expander(f"🌿 {disease}"):
                        st.write(f"**Treatment:** {info['treatment']}")
                        st.write(f"**Prevention:** {info['prevention']}")
            else:
                st.info("No exact matches found. Showing all available remedies:")
                for disease, info in diseases[remedy_category].items():
                    with st.expander(f"🌿 {disease}"):
                        st.write(f"**Treatment:** {info['treatment']}")
                        st.write(f"**Prevention:** {info['prevention']}")
    
    with col2:
        st.subheader("👥 Community Remedies")
        if st.session_state.user_remedies:
            for i, remedy in enumerate(st.session_state.user_remedies):
                with st.expander(f"Remedy #{i+1}"):
                    st.write(f"**Disease:** {remedy['disease']}")
                    st.write(f"**Remedy:** {remedy['remedy']}")
                    st.write(f"**Submitted by:** {remedy['farmer']}")
        else:
            st.info("No community remedies yet. Be the first to share!")

# Tab 3: Report & Submit
elif tab == "📋 Report & Submit":
    st.header("📋 Report Disease & Share Remedies")
    
    tab3_option = st.selectbox("What would you like to do?", 
                              ["🚨 Report New Disease Case", "💡 Share Home Remedy"])
    
    if tab3_option == "🚨 Report New Disease Case":
        st.subheader("🚨 Report New Disease Case")
        
        with st.form("disease_report"):
            farmer_name = st.text_input("Your Name:")
            location = st.text_input("Location (Village, State):")
            report_category = st.selectbox("Category:", ["plant", "livestock"])
            disease_name = st.text_input("Disease/Problem Name:")
            description = st.text_area("Detailed Description:")
            severity = st.selectbox("Severity Level:", ["Low", "Medium", "High", "Critical"])
            
            submitted = st.form_submit_button("📤 Submit Report")
            
            if submitted and farmer_name and location and disease_name:
                new_submission = {
                    "id": len(st.session_state.submissions) + 1,
                    "farmer": farmer_name,
                    "location": location,
                    "category": report_category,
                    "disease": disease_name,
                    "description": description,
                    "severity": severity,
                    "timestamp": datetime.now().isoformat(),
                    "status": "Open"
                }
                st.session_state.submissions.append(new_submission)
                st.success("✅ Report submitted successfully!")
    
    else:  # Share remedy
        st.subheader("💡 Share Your Home Remedy")
        
        with st.form("remedy_share"):
            farmer_name = st.text_input("Your Name:")
            remedy_disease = st.text_input("Disease it treats:")
            remedy_description = st.text_area("Remedy Description:")
            effectiveness = st.slider("How effective was it?", 1, 5, 3)
            
            submitted = st.form_submit_button("🤝 Share Remedy")
            
            if submitted and farmer_name and remedy_disease:
                new_remedy = {
                    "farmer": farmer_name,
                    "disease": remedy_disease,
                    "remedy": remedy_description,
                    "effectiveness": effectiveness,
                    "timestamp": datetime.now().isoformat()
                }
                st.session_state.user_remedies.append(new_remedy)
                st.success("✅ Thank you for sharing your knowledge!")

# Replace the Analytics Dashboard section (Tab 4) with this enhanced version:

elif tab == "📊 Analytics Dashboard":
    st.header("📊 Analytics & Disease Monitoring Dashboard")
    
    # Real-time timestamp
    st.caption(f"🕒 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Sample analytics data
    if st.session_state.submissions:
        df = pd.DataFrame(st.session_state.submissions)
        
        # Enhanced metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Reports", len(df))
        with col2:
            active_cases = len(df[df['status'] == 'Open'])
            st.metric("Active Cases", active_cases)
        with col3:
            critical_cases = len(df[df['severity'] == 'Critical'])
            st.metric("Critical Cases", critical_cases, 
                     delta=f"+{critical_cases}" if critical_cases > 0 else None,
                     delta_color="inverse")
        with col4:
            st.metric("Locations Affected", df['location'].nunique())
        with col5:
            avg_severity_score = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
            avg_score = df['severity'].map(avg_severity_score).mean()
            st.metric("Avg Severity", f"{avg_score:.1f}/4")
        
        # Alert section
        if critical_cases > 0:
            st.error(f"🚨 **ALERT**: {critical_cases} critical cases require immediate attention!")
        
        high_cases = len(df[df['severity'] == 'High'])
        if high_cases > 3:
            st.warning(f"⚠️ **Warning**: {high_cases} high-severity cases detected")
        
        # Charts section
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.subheader("📈 Reports by Category")
            category_counts = df['category'].value_counts()
            
            # Enhanced pie chart
            fig = px.pie(
                values=category_counts.values, 
                names=category_counts.index,
                title="Disease Reports by Category",
                color_discrete_map={'plant': '#228B22', 'livestock': '#FF6B35'}
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with chart_col2:
            st.subheader("🚨 Severity Distribution")
            severity_counts = df['severity'].value_counts()
            
            # Enhanced bar chart with custom colors
            colors = {'Low': '#228B22', 'Medium': '#FFD700', 'High': '#FF8C00', 'Critical': '#DC143C'}
            fig = px.bar(
                x=severity_counts.index, 
                y=severity_counts.values,
                title="Cases by Severity Level",
                color=severity_counts.index,
                color_discrete_map=colors
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Time series if enough data
        if len(df) > 1:
            st.subheader("📅 Submission Timeline")
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            daily_counts = df.groupby('date').size().reset_index(name='count')
            
            fig = px.line(
                daily_counts, 
                x='date', 
                y='count',
                title='Daily Disease Reports',
                markers=True
            )
            fig.update_layout(xaxis_title="Date", yaxis_title="Number of Reports")
            st.plotly_chart(fig, use_container_width=True)
        
        # Location heatmap
        location_counts = df['location'].value_counts().head(10)
        if len(location_counts) > 1:
            st.subheader("🗺️ Geographic Distribution")
            fig = px.bar(
                x=location_counts.values,
                y=location_counts.index,
                orientation='h',
                title='Top Affected Locations',
                color=location_counts.values,
                color_continuous_scale='Reds'
            )
            fig.update_layout(yaxis_title="Location", xaxis_title="Number of Cases")
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent submissions table
        st.subheader("📋 Recent Submissions")
        recent_df = df[['farmer', 'location', 'disease', 'severity', 'status', 'timestamp']].head(10)
        recent_df['timestamp'] = pd.to_datetime(recent_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        st.dataframe(recent_df, use_container_width=True)
        
        # Export option
        if st.button("📥 Export Data as CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"disease_reports_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    else:
        st.info("📊 No data available yet. Reports will appear here once submitted.")
        
        # Enhanced sample visualization
        sample_data = pd.DataFrame({
            'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            'Plant Diseases': [15, 22, 28, 35, 42, 38],
            'Livestock Diseases': [8, 12, 18, 25, 22, 20]
        })
        
        fig = px.line(
            sample_data, 
            x='Month', 
            y=['Plant Diseases', 'Livestock Diseases'],
            title='Sample Disease Trend (Demo Data)',
            markers=True
        )
        fig.update_layout(yaxis_title="Number of Cases")
        st.plotly_chart(fig, use_container_width=True)
        
        # Sample metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sample Total", "127", "↗️ +15%")
        with col2:
            st.metric("Sample Critical", "8", "↗️ +2")
        with col3:
            st.metric("Sample Locations", "25", "↗️ +3")

# Tab 5: About
else:  # About tab
    st.header("ℹ️ About This Portal")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Mission
        To empower farmers with AI-driven disease diagnosis and community knowledge sharing for better agricultural outcomes.
        
        ### ⭐ Key Features
        - **AI-Powered Diagnosis**: Upload images for instant disease identification
        - **Knowledge-Based Insights**: Rule-based reasoning for reliable advice
        - **Community Remedies**: Share and discover traditional treatments
        - **Real-time Reporting**: Track disease outbreaks in your region
        - **Mobile Accessible**: Works on any smartphone via web browser
        
        ### 🔬 Technology Stack
        - **Frontend**: Streamlit (Mobile-responsive)
        - **AI/ML**: CNN models for image classification
        - **Knowledge Base**: Rule-based reasoning engine
        - **Analytics**: Plotly for interactive visualizations
        - **Deployment**: Streamlit Cloud
        
        ### 🌱 Impact
        - Faster disease identification
        - Reduced crop/livestock losses  
        - Empowered farming communities
        - Data-driven agricultural decisions
        """)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>🎯 Quick Demo</h4>
            <p>1. Go to Disease Diagnosis</p>
            <p>2. Upload a plant/livestock image</p>
            <p>3. Describe symptoms</p>
            <p>4. Get instant AI diagnosis</p>
            <p>5. View treatment recommendations</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ Disclaimer:</strong><br>
            This is a demo system. For actual farming decisions, 
            please consult qualified agricultural experts and veterinarians.
        </div>
        """, unsafe_allow_html=True)
        
        # QR Code placeholder
        st.subheader("📱 Mobile Access")
        st.info("QR Code for mobile access would be generated here pointing to the deployed Streamlit app URL")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🌾 Farmers Disease Diagnostic Portal | Built with ❤️ for Smart India Hackathon 2024</p>
    <p>Empowering Agriculture through AI & Community Knowledge</p>
</div>
""", unsafe_allow_html=True)