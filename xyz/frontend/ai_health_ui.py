import streamlit as st
import requests
import json

# Page Config must be the first Streamlit command
st.set_page_config(
    page_title="Women's Health AI Assistant",
    page_icon="👩‍⚕️",
    layout="wide"
)

def set_custom_style():
    st.markdown("""
        <style>
        /* Global Styles */
        .main {
            background: linear-gradient(135deg, #FFF0F5 0%, #FFFAFA 100%);
            font-family: 'Helvetica Neue', sans-serif;
        }
        
        /* Welcome Section */
        .welcome-header {
            text-align: center;
            padding: 40px 20px;
            background: linear-gradient(160deg, rgba(255,192,203,0.3) 0%, rgba(255,240,245,0.3) 100%);
            border-radius: 20px;
            margin-bottom: 40px;
            box-shadow: 0 4px 15px rgba(255,105,180,0.1);
        }
        
        .welcome-title {
            color: #FF1493;
            font-size: 3.5em;
            font-weight: 800;
            text-shadow: 2px 2px 4px rgba(255,105,180,0.2);
            margin-bottom: 0.5em;
        }
        
        .welcome-subtitle {
            color: #FF69B4;
            font-size: 1.4em;
            font-weight: 500;
            font-style: italic;
            line-height: 1.6;
        }
        
        /* Cards */
        .metric-card {
            background: white;
            padding: 30px;
            border-radius: 20px;
            box-shadow: 0 6px 15px rgba(255,105,180,0.1);
            margin: 15px 0;
            border-left: 6px solid #FF69B4;
            transition: transform 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 20px rgba(255,105,180,0.15);
        }
        
        /* Buttons */
        .stButton>button {
            background: linear-gradient(45deg, #FF69B4 30%, #FF1493 90%);
            color: white;
            border-radius: 25px;
            padding: 15px 35px;
            font-weight: bold;
            font-size: 1.1em;
            border: none;
            box-shadow: 0 4px 15px rgba(255,105,180,0.3);
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(255,105,180,0.4);
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #FF1493;
            font-weight: 700;
            margin: 1em 0;
        }
        
        /* Sliders and Inputs */
        .stSlider>div>div {
            background-color: #FF69B4;
        }
        
        .stProgress>div>div {
            background: linear-gradient(90deg, #FF69B4 0%, #FF1493 100%);
        }
        
        /* Metrics */
        div[data-testid="stMetricValue"] {
            font-size: 2.2rem;
            color: #FF1493;
            font-weight: 700;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: rgba(255,240,245,0.6);
            padding: 10px;
            border-radius: 15px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: white;
            border-radius: 10px;
            padding: 10px 20px;
            color: #FF69B4;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(45deg, #FF69B4 30%, #FF1493 90%);
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    set_custom_style()

    # Welcome Section
    st.markdown("""
        <div class='welcome-header'>
            <div class='welcome-title'>✨ Bloom & Glow ✨</div>
            <div class='welcome-subtitle'>
                Your Personal AI Health & Wellness Guide<br>
                🌸 Empowering Women's Health Journey 🌸
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Rest of your existing main() function code...

    # Title with decorative elements
    st.markdown("<h1>✨ Women's Wellness AI Assistant ✨</h1>", unsafe_allow_html=True)
    
    # Inspirational Quote
    st.markdown("""
        <div style='text-align: center; padding: 20px; background-color: rgba(255, 105, 180, 0.1); 
        border-radius: 10px; margin-bottom: 30px;'>
        <span style='font-size: 1.2em; color: #FF69B4; font-style: italic;'>
        💝 "Empowering women to take control of their health journey through AI-driven insights"
        </span>
        </div>
    """, unsafe_allow_html=True)

    # Main Content in Tabs
    tab1, tab2 = st.tabs(["🔍 Health Assessment", "📚 Wellness Resources"])

    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🌸 Your Health Profile")
            with st.container():
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                age = st.slider("Age", 18, 100, 30)
                bmi = st.number_input("BMI", 15.0, 50.0, 22.0)
                physical_activity = st.select_slider(
                    "Physical Activity Level",
                    options=["Sedentary", "Light", "Moderate", "Active", "Very Active"],
                    value="Moderate"
                )
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("### 🫀 Health Metrics")
            with st.container():
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                mental_health = st.slider("Mental Wellbeing Score", 0, 100, 70)
                reproductive_health = st.slider("Reproductive Health Score", 0, 100, 80)
                chronic_conditions = st.multiselect(
                    "Select Any Chronic Conditions",
                    ["None", "Diabetes", "Hypertension", "PCOS", "Thyroid", "Other"]
                )
                menopause_status = st.radio("Menopause Status", ["No", "Yes"])
                st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("""
                <div class='metric-card' style='text-align: center;'>
                    <h3>💖 Daily Health Tips</h3>
                    <p>• Stay active daily<br>
                    • Practice mindfulness<br>
                    • Maintain balanced nutrition<br>
                    • Prioritize mental health<br>
                    • Get regular check-ups</p>
                </div>
            """, unsafe_allow_html=True)

        # Analysis Button
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("✨ Get Personalized Health Insights ✨", type="primary"):
            with st.spinner("🌸 Analyzing your health profile..."):
                activity_map = {
                    "Sedentary": 2, "Light": 4, "Moderate": 6, 
                    "Active": 8, "Very Active": 10
                }
                
                data = {
                    "Age": age,
                    "BMI": bmi,
                    "Chronic_Conditions": len(chronic_conditions) if "None" not in chronic_conditions else 0,
                    "Physical_Activity": activity_map[physical_activity],
                    "Mental_Health_Score": mental_health,
                    "Reproductive_Health_Score": reproductive_health,
                    "Menopause_Status": 1 if menopause_status == "Yes" else 0
                }

                try:
                    response = requests.post(
                        "http://127.0.0.1:5000/predict",
                        json=data,
                        headers={"Content-Type": "application/json"},
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success("✨ Analysis Complete!")
                        
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Health Profile", f"Group {result['cluster'] + 1}")
                        with col2:
                            st.metric("Wellness Score", f"{100 - (result['risk_score'] * 20):.0f}/100")
                        with col3:
                            st.metric("Priority Level", "High" if result['risk_score'] > 0.5 else "Normal")

                        st.markdown("### 🌟 Personalized Recommendations")
                        st.info(result['base_recommendation'])
                        
                        st.markdown("### 💫 AI Health Insights")
                        st.write(result['ai_insights'])
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.error(f"Error: Server returned status code {response.status_code}")

                except requests.exceptions.ConnectionError:
                    st.error("❌ Connection Error: Unable to connect to the AI Health API. Please make sure the backend server is running.")
                except requests.exceptions.Timeout:
                    st.error("⏰ Timeout Error: The server took too long to respond. Please try again.")
                except json.JSONDecodeError:
                    st.error("📄 Error: Received invalid response from server")
                except Exception as e:
                    st.error(f"💥 An unexpected error occurred: {str(e)}")

    with tab2:
        st.markdown("### 📚 Wellness Resources")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div class='metric-card'>
                    <h3>🧘‍♀️ Daily Wellness Practices</h3>
                    <ul>
                        <li>Morning meditation</li>
                        <li>Gentle yoga</li>
                        <li>Mindful breathing</li>
                        <li>Evening relaxation</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class='metric-card'>
                    <h3>💝 Health Tips</h3>
                    <ul>
                        <li>🌿 Practice mindful eating</li>
                        <li>💪 Regular exercise (30 mins/day)</li>
                        <li>😴 Get 7-8 hours of sleep</li>
                        <li>🚰 Stay hydrated</li>
                        <li>🧘‍♀️ Stress management</li>
                        <li>🩺 Regular check-ups</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()