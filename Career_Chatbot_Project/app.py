"""
Enhanced Career Guidance Chatbot - Streamlit Web Application with Comprehensive Analytics
This application provides career guidance with visualizations and performance metrics
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Career Guidance Analytics Hub",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for responsive design
st.markdown("""
<style>
    /* Main styling */
    .main {
        padding: 0rem 1rem;
    }
    
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        color: white;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .chat-container {
        max-height: 400px;
        overflow-y: auto;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        background: #f8f9fa;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        animation: fadeIn 0.5s ease-in;
    }
    
    .user-message {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        border-left: 4px solid #2196f3;
        margin-left: 2rem;
    }
    
    .bot-message {
        background: linear-gradient(135deg, #f1f8e9, #c8e6c9);
        border-left: 4px solid #4caf50;
        margin-right: 2rem;
    }
    
    .career-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #ff9800;
        transition: transform 0.3s ease;
    }
    
    .career-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .confidence-bar {
        background: #e0e0e0;
        border-radius: 20px;
        overflow: hidden;
        height: 25px;
        margin: 1rem 0;
        position: relative;
    }
    
    .confidence-fill {
        background: linear-gradient(90deg, #4caf50, #8bc34a, #cddc39);
        height: 100%;
        border-radius: 20px;
        transition: width 0.8s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 10px;
        color: #262730;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            padding: 1rem;
            margin: -0.5rem -0.5rem 1rem -0.5rem;
        }
        
        .stats-grid {
            grid-template-columns: 1fr;
        }
        
        .chat-message {
            padding: 0.75rem;
        }
        
        .user-message {
            margin-left: 0.5rem;
        }
        
        .bot-message {
            margin-right: 0.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

def clean_text(text):
    """Clean text for model prediction"""
    if pd.isna(text) or text == "":
        return ""
    
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

@st.cache_resource
def load_model_and_data():
    """Load the trained model, vectorizer and results data"""
    try:
        model = joblib.load('intent_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        
        # Load results if available
        results_df = None
        if os.path.exists('model_comparison_results.csv'):
            results_df = pd.read_csv('model_comparison_results.csv')
        
        # Load original dataset if available
        dataset_df = None
        if os.path.exists('career_guidance_dataset.csv'):
            dataset_df = pd.read_csv('career_guidance_dataset.csv')
        
        return model, vectorizer, results_df, dataset_df, True
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.error("Please run 'python train_model.py' first to train the model.")
        return None, None, None, None, False

def get_comprehensive_career_info():
    """Return comprehensive career information"""
    return {
        "Data Scientist": {
            "description": "Analyzes complex data to extract insights and build predictive models for business decisions",
            "skills": ["Python/R", "Statistics", "Machine Learning", "Data Visualization", "SQL", "Deep Learning"],
            "education": "Bachelor's/Master's in Computer Science, Statistics, Mathematics, or related field",
            "salary_range": "$70,000 - $150,000+",
            "growth_outlook": "Very High (22% growth expected)",
            "industries": ["Technology", "Healthcare", "Finance", "E-commerce", "Consulting"],
            "work_environment": "Office/Remote, collaborative team settings",
            "daily_tasks": ["Data analysis", "Model building", "Report generation", "Stakeholder meetings"]
        },
        "Software Engineer": {
            "description": "Designs, develops, and maintains software applications and systems",
            "skills": ["Programming Languages", "Problem Solving", "System Design", "Version Control", "Testing"],
            "education": "Bachelor's in Computer Science, Software Engineering, or related field",
            "salary_range": "$60,000 - $140,000+",
            "growth_outlook": "High (13% growth expected)",
            "industries": ["Technology", "Banking", "Healthcare", "Gaming", "Startups"],
            "work_environment": "Office/Remote, agile development teams",
            "daily_tasks": ["Code development", "Testing", "Code reviews", "Documentation"]
        },
        "Digital Marketing Analyst": {
            "description": "Analyzes digital marketing campaigns and consumer behavior to optimize marketing strategies",
            "skills": ["Google Analytics", "SEO/SEM", "Data Analysis", "Social Media Marketing", "A/B Testing"],
            "education": "Bachelor's in Marketing, Business, Communications, or related field",
            "salary_range": "$45,000 - $85,000+",
            "growth_outlook": "High (19% growth expected)",
            "industries": ["Advertising", "E-commerce", "Media", "Retail", "Consulting"],
            "work_environment": "Office/Remote, cross-functional teams",
            "daily_tasks": ["Campaign analysis", "Report creation", "Strategy planning", "Performance optimization"]
        },
        "UX/UI Designer": {
            "description": "Creates user-friendly interfaces and improves overall user experience for digital products",
            "skills": ["Design Tools (Figma, Sketch)", "User Research", "Prototyping", "Visual Design", "Wireframing"],
            "education": "Bachelor's in Design, HCI, Psychology, or related field",
            "salary_range": "$55,000 - $110,000+",
            "growth_outlook": "High (13% growth expected)",
            "industries": ["Technology", "E-commerce", "Gaming", "Healthcare", "Finance"],
            "work_environment": "Office/Remote, design teams",
            "daily_tasks": ["User research", "Design creation", "Prototyping", "User testing"]
        },
        "Business Analyst": {
            "description": "Analyzes business processes and recommends solutions for organizational improvement",
            "skills": ["Data Analysis", "Business Process Modeling", "Requirements Gathering", "Communication", "SQL"],
            "education": "Bachelor's in Business, Economics, Engineering, or related field",
            "salary_range": "$55,000 - $95,000+",
            "growth_outlook": "High (11% growth expected)",
            "industries": ["Consulting", "Finance", "Healthcare", "Technology", "Government"],
            "work_environment": "Office/Remote, cross-departmental collaboration",
            "daily_tasks": ["Process analysis", "Requirements documentation", "Stakeholder meetings", "Solution design"]
        },
        "Marketing Manager": {
            "description": "Plans, develops, and implements marketing strategies to promote products or services",
            "skills": ["Strategic Planning", "Digital Marketing", "Brand Management", "Market Research", "Analytics"],
            "education": "Bachelor's in Marketing, Business, or related field",
            "salary_range": "$50,000 - $100,000+",
            "growth_outlook": "High (10% growth expected)",
            "industries": ["Retail", "Technology", "Healthcare", "Finance", "Consumer Goods"],
            "work_environment": "Office/Remote, cross-functional collaboration",
            "daily_tasks": ["Campaign planning", "Market research", "Budget management", "Team coordination"]
        },
        "Web Developer": {
            "description": "Creates and maintains websites and web applications using various programming languages",
            "skills": ["HTML/CSS", "JavaScript", "React/Vue.js", "Backend Development", "Database Management"],
            "education": "Bachelor's in Computer Science, Web Development bootcamp, or self-taught",
            "salary_range": "$45,000 - $90,000+",
            "growth_outlook": "High (13% growth expected)",
            "industries": ["Technology", "E-commerce", "Media", "Startups", "Agencies"],
            "work_environment": "Office/Remote, development teams",
            "daily_tasks": ["Code development", "Website maintenance", "Testing", "Client communication"]
        },
        "Default": {
            "description": "Explore various career opportunities based on your interests and skills",
            "skills": ["Research your field of interest", "Develop relevant skills", "Network with professionals"],
            "education": "Varies by field",
            "salary_range": "Varies by role and location",
            "growth_outlook": "Research specific field trends",
            "industries": ["Various"],
            "work_environment": "Varies",
            "daily_tasks": ["Field-specific tasks"]
        }
    }

def calculate_model_metrics(model, vectorizer, dataset_df):
    """Calculate comprehensive model metrics"""
    if dataset_df is None:
        return None
    
    try:
        # Prepare data
        X = dataset_df['question'].fillna('').apply(clean_text)
        y = dataset_df['role']
        
        # Remove empty questions
        mask = X != ''
        X = X[mask]
        y = y[mask]
        
        # Vectorize
        X_vec = vectorizer.transform(X)
        
        # Predictions
        y_pred = model.predict(X_vec)
        y_pred_proba = model.predict_proba(X_vec)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='weighted')
        precision = precision_score(y, y_pred, average='weighted')
        recall = recall_score(y, y_pred, average='weighted')
        
        # Confidence distribution
        confidence_scores = np.max(y_pred_proba, axis=1)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'confidence_scores': confidence_scores,
            'predictions': y_pred,
            'actual': y,
            'classes': model.classes_
        }
    except Exception as e:
        st.error(f"Error calculating metrics: {e}")
        return None

def create_metrics_visualization(metrics):
    """Create comprehensive metrics visualizations"""
    if metrics is None:
        return None, None, None, None
    
    # 1. Performance Metrics Bar Chart
    metrics_fig = go.Figure()
    metric_names = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
    metric_values = [metrics['accuracy'], metrics['f1_score'], metrics['precision'], metrics['recall']]
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
    
    metrics_fig.add_trace(go.Bar(
        x=metric_names,
        y=metric_values,
        marker_color=colors,
        text=[f'{val:.3f}' for val in metric_values],
        textposition='auto',
        textfont_size=14
    ))
    
    metrics_fig.update_layout(
        title={'text': 'Model Performance Metrics', 'x': 0.5, 'font': {'size': 20}},
        yaxis_title='Score',
        yaxis=dict(range=[0, 1]),
        template='plotly_white',
        height=400
    )
    
    # 2. Confusion Matrix Heatmap
    cm = confusion_matrix(metrics['actual'], metrics['predictions'])
    classes = metrics['classes']
    
    confusion_fig = px.imshow(
        cm,
        x=classes,
        y=classes,
        color_continuous_scale='Blues',
        aspect='auto',
        title='Confusion Matrix'
    )
    confusion_fig.update_layout(
        title={'x': 0.5, 'font': {'size': 20}},
        xaxis_title='Predicted',
        yaxis_title='Actual',
        height=500
    )
    
    # 3. Confidence Distribution
    confidence_fig = px.histogram(
        x=metrics['confidence_scores'],
        nbins=30,
        title='Prediction Confidence Distribution',
        color_discrete_sequence=['#667eea']
    )
    confidence_fig.update_layout(
        title={'x': 0.5, 'font': {'size': 20}},
        xaxis_title='Confidence Score',
        yaxis_title='Frequency',
        template='plotly_white',
        height=400
    )
    
    # 4. Class Distribution
    class_counts = pd.Series(metrics['actual']).value_counts()
    class_fig = px.pie(
        values=class_counts.values,
        names=class_counts.index,
        title='Career Classes Distribution'
    )
    class_fig.update_layout(
        title={'x': 0.5, 'font': {'size': 20}},
        height=400
    )
    
    return metrics_fig, confusion_fig, confidence_fig, class_fig

def create_model_comparison_viz(results_df):
    """Create model comparison visualizations"""
    if results_df is None:
        return None, None
    
    # Clean results
    results_clean = results_df.dropna(subset=['Test Accuracy'])
    
    if results_clean.empty:
        return None, None
    
    # 1. Model Comparison Bar Chart
    comparison_fig = px.bar(
        results_clean,
        x='Model',
        y='Test Accuracy',
        color='Vectorizer',
        title='Model Performance Comparison',
        barmode='group'
    )
    comparison_fig.update_layout(
        title={'x': 0.5, 'font': {'size': 20}},
        height=500,
        template='plotly_white'
    )
    
    # 2. Training Time vs Accuracy Scatter
    scatter_fig = px.scatter(
        results_clean,
        x='Training Time (s)',
        y='Test Accuracy',
        color='Vectorizer',
        size='CV Mean',
        hover_data=['Model'],
        title='Training Time vs Accuracy'
    )
    scatter_fig.update_layout(
        title={'x': 0.5, 'font': {'size': 20}},
        height=400,
        template='plotly_white'
    )
    
    return comparison_fig, scatter_fig

def predict_career_with_details(question, model, vectorizer):
    """Enhanced prediction with detailed analysis"""
    if not question.strip():
        return None, 0, [], []
    
    clean_question = clean_text(question)
    if not clean_question:
        return None, 0, [], []
    
    question_vec = vectorizer.transform([clean_question])
    
    # Get prediction and probabilities
    prediction = model.predict(question_vec)[0]
    probabilities = model.predict_proba(question_vec)[0]
    confidence = probabilities.max()
    
    # Get top 5 predictions
    top_indices = probabilities.argsort()[-5:][::-1]
    top_careers = [model.classes_[i] for i in top_indices]
    top_probabilities = [probabilities[i] for i in top_indices]
    
    return prediction, confidence, top_careers, top_probabilities

def display_enhanced_career_info(career, confidence, top_careers, top_probs):
    """Display enhanced career information with top predictions"""
    career_info = get_comprehensive_career_info()
    info = career_info.get(career, career_info["Default"])
    
    # Use Streamlit native components instead of complex HTML
    st.markdown(f"## 🎯 {career}")
    st.markdown(f"**Description:** {info['description']}")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💼 Key Skills")
        for skill in info['skills']:
            st.markdown(f"• {skill}")
        
        st.markdown("### 🏢 Industries")
        st.markdown(", ".join(info['industries']))
        
        st.markdown("### 📋 Daily Tasks")
        for task in info['daily_tasks']:
            st.markdown(f"• {task}")
    
    with col2:
        st.markdown("### 🎓 Education")
        st.markdown(info['education'])
        
        st.markdown("### 💰 Salary Range")
        st.markdown(info['salary_range'])
        
        st.markdown("### 📈 Growth Outlook")
        st.markdown(info['growth_outlook'])
        
        st.markdown("### 🌍 Work Environment")
        st.markdown(info['work_environment'])
    
    # Confidence visualization using Streamlit progress bar
    st.markdown("### 🎯 Prediction Confidence")
    confidence_percentage = confidence * 100
    st.progress(confidence, text=f"{confidence_percentage:.1f}% Confident")
    
    # Top 5 predictions chart
    if len(top_careers) > 1:
        st.markdown("### 📊 Alternative Career Suggestions")
        
        fig = go.Figure()
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57']
        
        fig.add_trace(go.Bar(
            y=top_careers,
            x=[prob * 100 for prob in top_probs],
            orientation='h',
            marker_color=colors[:len(top_careers)],
            text=[f'{prob:.1%}' for prob in top_probs],
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Top Career Matches',
            xaxis_title='Confidence (%)',
            height=300,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Rest of the function remains the same...
def main():
    """Enhanced main application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌟 AI Career Guidance Analytics Hub</h1>
        <p>Discover your perfect career path with AI-powered insights and comprehensive analytics!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load resources
    model, vectorizer, results_df, dataset_df, model_loaded = load_model_and_data()
    
    if not model_loaded:
        st.error("🚨 Model not found! Please run the training script first.")
        st.stop()
    
    # Calculate metrics
    with st.spinner("🔄 Loading analytics..."):
        metrics = calculate_model_metrics(model, vectorizer, dataset_df)
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📊 Analytics", "🔬 Model Insights", "📈 Performance"])
    
    with tab1:
        # Main chat interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 💬 Career Guidance Chat")
            
            # User input
            user_input = st.text_area(
                "Ask me anything about careers:",
                placeholder="e.g., What career is best for someone interested in technology and data?",
                height=100
            )
            
            col_a, col_b, col_c = st.columns([1, 1, 1])
            with col_a:
                ask_button = st.button("🔍 Analyze Career", type="primary", use_container_width=True)
            with col_b:
                clear_button = st.button("🗑️ Clear Chat", use_container_width=True)
            with col_c:
                export_button = st.button("📥 Export Chat", use_container_width=True)
            
            if clear_button:
                st.session_state.chat_history = []
                st.rerun()
            
            # Process user input
            if ask_button and user_input.strip():
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                # Add user message
                st.session_state.chat_history.append({
                    'type': 'user',
                    'message': user_input,
                    'timestamp': timestamp
                })
                
                # Get enhanced prediction
                with st.spinner("🤖 Analyzing your career interests..."):
                    career, confidence, top_careers, top_probs = predict_career_with_details(
                        user_input, model, vectorizer
                    )
                
                if career and confidence > 0.1:
                    bot_response = f"Based on your interests and skills, I'd recommend exploring: **{career}**"
                else:
                    bot_response = "I need more specific information about your interests, skills, or career goals to provide better recommendations."
                    career = None
                
                # Add bot response
                st.session_state.chat_history.append({
                    'type': 'bot',
                    'message': bot_response,
                    'career': career,
                    'confidence': confidence,
                    'top_careers': top_careers,
                    'top_probs': top_probs,
                    'timestamp': timestamp
                })
                
                st.rerun()
            
            # Export chat history
            if export_button and st.session_state.chat_history:
                chat_data = []
                for chat in st.session_state.chat_history:
                    chat_data.append({
                        'Type': chat['type'],
                        'Message': chat['message'],
                        'Career': chat.get('career', ''),
                        'Confidence': chat.get('confidence', ''),
                        'Timestamp': chat['timestamp']
                    })
                
                df_export = pd.DataFrame(chat_data)
                csv = df_export.to_csv(index=False)
                st.download_button(
                    label="📄 Download Chat History",
                    data=csv,
                    file_name=f"career_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # Display chat history
            if st.session_state.chat_history:
                st.markdown("### 💭 Conversation History")
                
                with st.container():
                    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
                    
                    for chat in reversed(st.session_state.chat_history[-6:]):  # Show last 6 messages
                        if chat['type'] == 'user':
                            st.markdown(f"""
                            <div class="chat-message user-message">
                                <strong>You ({chat['timestamp']}):</strong><br>
                                {chat['message']}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="chat-message bot-message">
                                <strong>Career AI ({chat['timestamp']}):</strong><br>
                                {chat['message']}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Show detailed career info for latest prediction
                latest_bot = None
                for chat in reversed(st.session_state.chat_history):
                    if chat['type'] == 'bot' and chat.get('career'):
                        latest_bot = chat
                        break
                
                if latest_bot:
                    st.markdown("---")
                    display_enhanced_career_info(
                        latest_bot['career'], 
                        latest_bot['confidence'],
                        latest_bot.get('top_careers', []),
                        latest_bot.get('top_probs', [])
                    )
        
        with col2:
            st.markdown("### 🎯 Quick Stats")
            
            if metrics:
                st.markdown(f"""
                <div class="metric-container">
                    <h3>Model Accuracy</h3>
                    <h2>{metrics['accuracy']:.1%}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-container">
                    <h3>F1 Score</h3>
                    <h2>{metrics['f1_score']:.3f}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-container">
                    <h3>Careers Covered</h3>
                    <h2>{len(metrics['classes'])}</h2>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-container">
                    <h3>Model Status</h3>
                    <h2>✅ Ready</h2>
                </div>
                """, unsafe_allow_html=True)
            
            total_chats = len([c for c in st.session_state.chat_history if c['type'] == 'user'])
            st.markdown(f"""
            <div class="metric-container">
                <h3>Questions Asked</h3>
                <h2>{total_chats}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 💡 Sample Questions")
            sample_questions = [
                "What does a data scientist do daily?",
                "Skills needed for software engineering?",
                "Career path in digital marketing",
                "UX designer requirements",
                "Business analyst job description",
                "AI engineer career prospects"
            ]
            
            for i, question in enumerate(sample_questions):
                if st.button(f"💭 {question[:30]}...", key=f"sample_{i}", use_container_width=True):
                    # Add sample question to chat
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    st.session_state.chat_history.append({
                        'type': 'user',
                        'message': question,
                        'timestamp': timestamp
                    })
                    
                    # Get prediction for sample question
                    career, confidence, top_careers, top_probs = predict_career_with_details(
                        question, model, vectorizer
                    )
                    
                    if career and confidence > 0.1:
                        bot_response = f"Based on this question, I'd recommend exploring: **{career}**"
                    else:
                        bot_response = "Let me provide some general career guidance for this topic."
                    
                    st.session_state.chat_history.append({
                        'type': 'bot',
                        'message': bot_response,
                        'career': career,
                        'confidence': confidence,
                        'top_careers': top_careers,
                        'top_probs': top_probs,
                        'timestamp': timestamp
                    })
                    st.rerun()
    
    with tab2:
        st.markdown("### 📊 Career Analytics Dashboard")
        
        if dataset_df is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📈 Dataset Statistics")
                total_questions = len(dataset_df)
                unique_careers = dataset_df['role'].nunique()
                
                st.metric("Total Questions", total_questions)
                st.metric("Career Categories", unique_careers)
                st.metric("Average Question Length", f"{dataset_df['question'].str.len().mean():.0f} chars")
                
                # Career distribution pie chart
                career_counts = dataset_df['role'].value_counts()
                fig_pie = px.pie(
                    values=career_counts.values,
                    names=career_counts.index,
                    title="Career Categories Distribution"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.markdown("#### 🔍 Question Length Analysis")
                
                # Question length distribution
                dataset_df['question_length'] = dataset_df['question'].str.len()
                fig_hist = px.histogram(
                    dataset_df,
                    x='question_length',
                    title="Question Length Distribution",
                    color_discrete_sequence=['#667eea']
                )
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # Top 10 most common words
                st.markdown("#### 🔤 Most Common Words")
                all_text = ' '.join(dataset_df['question'].fillna('').astype(str))
                words = re.findall(r'\b\w+\b', all_text.lower())
                word_freq = pd.Series(words).value_counts().head(10)
                
                fig_words = px.bar(
                    x=word_freq.values,
                    y=word_freq.index,
                    orientation='h',
                    title="Top 10 Words",
                    color_discrete_sequence=['#4ecdc4']
                )
                st.plotly_chart(fig_words, use_container_width=True)
        else:
            st.warning("Dataset not available for analysis.")
    
    with tab3:
        st.markdown("### 🔬 Model Performance Insights")
        
        if metrics:
            col1, col2 = st.columns(2)
            
            with col1:
                # Create and display metrics visualizations
                metrics_fig, confusion_fig, confidence_fig, class_fig = create_metrics_visualization(metrics)
                
                if metrics_fig:
                    st.plotly_chart(metrics_fig, use_container_width=True)
                
                if confidence_fig:
                    st.plotly_chart(confidence_fig, use_container_width=True)
            
            with col2:
                if confusion_fig:
                    st.plotly_chart(confusion_fig, use_container_width=True)
                
                if class_fig:
                    st.plotly_chart(class_fig, use_container_width=True)
            
            # Detailed classification report
            st.markdown("#### 📋 Detailed Classification Report")
            
            try:
                class_report = classification_report(
                    metrics['actual'], 
                    metrics['predictions'], 
                    output_dict=True
                )
                
                # Convert to DataFrame for better display
                report_df = pd.DataFrame(class_report).transpose()
                report_df = report_df.round(3)
                
                st.dataframe(
                    report_df,
                    use_container_width=True,
                    column_config={
                        "precision": st.column_config.NumberColumn(
                            "Precision",
                            format="%.3f"
                        ),
                        "recall": st.column_config.NumberColumn(
                            "Recall", 
                            format="%.3f"
                        ),
                        "f1-score": st.column_config.NumberColumn(
                            "F1-Score",
                            format="%.3f"
                        ),
                        "support": st.column_config.NumberColumn(
                            "Support",
                            format="%d"
                        )
                    }
                )
            except Exception as e:
                st.error(f"Error generating classification report: {e}")
        else:
            st.warning("Model metrics not available.")
    
    with tab4:
        st.markdown("### 📈 Model Comparison & Performance")
        
        if results_df is not None:
            st.markdown("#### 🏆 Model Performance Comparison")
            
            # Display results table
            st.dataframe(
                results_df,
                use_container_width=True,
                column_config={
                    "Test Accuracy": st.column_config.NumberColumn(
                        "Test Accuracy",
                        format="%.4f"
                    ),
                    "CV Mean": st.column_config.NumberColumn(
                        "CV Mean",
                        format="%.4f"
                    ),
                    "CV Std": st.column_config.NumberColumn(
                        "CV Std", 
                        format="%.4f"
                    ),
                    "Training Time (s)": st.column_config.NumberColumn(
                        "Training Time (s)",
                        format="%.2f"
                    )
                }
            )
            
            # Create comparison visualizations
            comparison_fig, scatter_fig = create_model_comparison_viz(results_df)
            
            col1, col2 = st.columns(2)
            with col1:
                if comparison_fig:
                    st.plotly_chart(comparison_fig, use_container_width=True)
            
            with col2:
                if scatter_fig:
                    st.plotly_chart(scatter_fig, use_container_width=True)
            
            # Best model summary
            st.markdown("#### 🥇 Best Performing Models")
            
            if not results_df.empty:
                best_model = results_df.loc[results_df['Test Accuracy'].idxmax()]
                
                st.markdown(f"""
                <div class="career-card">
                    <h3>🏆 Best Overall Model</h3>
                    <p><strong>Model:</strong> {best_model['Model']}</p>
                    <p><strong>Vectorizer:</strong> {best_model['Vectorizer']}</p>
                    <p><strong>Test Accuracy:</strong> {best_model['Test Accuracy']:.4f}</p>
                    <p><strong>CV Score:</strong> {best_model['CV Mean']:.4f} ± {best_model['CV Std']:.4f}</p>
                    <p><strong>Training Time:</strong> {best_model['Training Time (s)']:.2f} seconds</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Fastest model
                fastest_model = results_df.loc[results_df['Training Time (s)'].idxmin()]
                st.markdown(f"""
                <div class="career-card">
                    <h3>⚡ Fastest Training Model</h3>
                    <p><strong>Model:</strong> {fastest_model['Model']}</p>
                    <p><strong>Vectorizer:</strong> {fastest_model['Vectorizer']}</p>
                    <p><strong>Training Time:</strong> {fastest_model['Training Time (s)']:.2f} seconds</p>
                    <p><strong>Test Accuracy:</strong> {fastest_model['Test Accuracy']:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Model comparison results not available. Run the training script to generate comparison data.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px; margin: 2rem 0;">
        <h4>🚀 Career Guidance AI</h4>
        <p>Powered by Machine Learning • Built with Streamlit</p>
        <p><em>Helping you discover your perfect career path through AI-driven insights!</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()