"""
Enhanced Career Guidance Chatbot - Streamlit Web Application with Visualizations
This application provides career guidance with comprehensive analytics and visualizations
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
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import learning_curve
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
    
    .viz-button {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .viz-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.25);
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
    info = career_info.get(career, {
        "description": "Career information not available in current database",
        "skills": ["Research required"],
        "education": "Varies by field",
        "salary_range": "Research required",
        "growth_outlook": "Research required",
        "industries": ["Various"],
        "work_environment": "Varies",
        "daily_tasks": ["Field-specific tasks"]
    })
    
    # Main career card
    st.markdown(f"""
    <div class="career-card">
        <h2>🎯 {career}</h2>
        <p style="font-size: 1.1em; color: #555;"><strong>Description:</strong> {info['description']}</p>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; margin: 1rem 0;">
            <div>
                <h4>💼 Key Skills:</h4>
                <ul>
                    {''.join([f"<li>{skill}</li>" for skill in info['skills']])}
                </ul>
            </div>
            
            <div>
                <h4>🎓 Education:</h4>
                <p>{info['education']}</p>
                
                <h4>💰 Salary Range:</h4>
                <p>{info['salary_range']}</p>
                
                <h4>📈 Growth Outlook:</h4>
                <p>{info['growth_outlook']}</p>
            </div>
        </div>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;">
            <div>
                <h4>🏢 Industries:</h4>
                <p>{', '.join(info['industries'])}</p>
            </div>
            
            <div>
                <h4>🌍 Work Environment:</h4>
                <p>{info['work_environment']}</p>
            </div>
        </div>
        
        <div>
            <h4>📋 Daily Tasks:</h4>
            <ul>
                {''.join([f"<li>{task}</li>" for task in info['daily_tasks']])}
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Confidence visualization
    st.markdown("### 🎯 Prediction Confidence")
    confidence_percentage = int(confidence * 100)
    st.markdown(f"""
    <div class="confidence-bar">
        <div class="confidence-fill" style="width: {confidence_percentage}%">
            {confidence_percentage}% Confident
        </div>
    </div>
    """, unsafe_allow_html=True)
    
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
    if 'show_analytics' not in st.session_state:
        st.session_state.show_analytics = False
    
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
                analytics_button = st.button("📊 View Analytics", use_container_width=True)
            
            if analytics_button:
                st.session_state.show_analytics = not st.session_state.show_analytics
            
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
                if st.button(f"💭 {question}", key=f"sample_{i}", use_container_width=True):
                    st.session_state.sample_input = question
                    st.rerun()
    
    with tab2:
        st.markdown("### 📊 Model Performance Analytics")
        
        if metrics:
            # Create visualizations
            metrics_fig, confusion_fig, confidence_fig, class_fig = create_metrics_visualization(metrics)
            
            # Display metrics in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(metrics_fig, use_container_width=True)
                st.plotly_chart(confidence_fig, use_container_width=True)
            
            with col2:
                st.plotly_chart(class_fig, use_container_width=True)
                
                # Detailed metrics table
                st.markdown("#### 📋 Detailed Metrics")
                metrics_data = {
                    'Metric': ['Accuracy', 'F1-Score', 'Precision', 'Recall'],
                    'Score': [
                        f"{metrics['accuracy']:.4f}",
                        f"{metrics['f1_score']:.4f}",
                        f"{metrics['precision']:.4f}",
                        f"{metrics['recall']:.4f}"
                    ],
                    'Percentage': [
                        f"{metrics['accuracy']:.1%}",
                        f"{metrics['f1_score']:.1%}",
                        f"{metrics['precision']:.1%}",
                        f"{metrics['recall']:.1%}"
                    ]
                }
                st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
            
            # Confusion matrix (full width)
            st.plotly_chart(confusion_fig, use_container_width=True)
        
        else:
            st.warning("📝 Analytics not available. Dataset required for detailed metrics.")
    
    with tab3:
        st.markdown("### 🔬 Model Comparison Insights")
        
        if results_df is not None:
            comparison_fig, scatter_fig = create_model_comparison_viz(results_df)
            
            if comparison_fig:
                st.plotly_chart(comparison_fig, use_container_width=True)
                st.plotly_chart(scatter_fig, use_container_width=True)
            
            # Results table
            st.markdown("#### 📋 Model Comparison Results")
            if not results_df.empty:
                # Display top performing models
                top_models = results_df.nlargest(5, 'Test Accuracy')
                st.dataframe(
                    top_models[['Model', 'Vectorizer', 'Test Accuracy', 'CV Mean', 'Training Time (s)']].round(4),
                    use_container_width=True
                )
        else:
            st.info("📝 Model comparison results not available. Run model training with comparison to see insights.")
    
    with tab4:
        st.markdown("### 📈 Performance Monitoring")
        
        # Career distribution from chat history
        if st.session_state.chat_history:
            career_predictions = []
            for chat in st.session_state.chat_history:
                if chat['type'] == 'bot' and chat.get('career'):
                    career_predictions.append(chat['career'])
            
            if career_predictions:
                career_counts = pd.Series(career_predictions).value_counts()
                
                # Career prediction trends
                fig_trends = px.bar(
                    x=career_counts.index,
                    y=career_counts.values,
                    title="Most Recommended Careers in This Session",
                    color=career_counts.values,
                    color_continuous_scale="viridis"
                )
                fig_trends.update_layout(
                    xaxis_title="Career",
                    yaxis_title="Frequency",
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig_trends, use_container_width=True)
            
            # Confidence trends
            confidence_scores = []
            timestamps = []
            for chat in st.session_state.chat_history:
                if chat['type'] == 'bot' and chat.get('confidence'):
                    confidence_scores.append(chat['confidence'])
                    timestamps.append(chat['timestamp'])
            
            if confidence_scores:
                fig_confidence = px.line(
                    x=range(1, len(confidence_scores) + 1),
                    y=confidence_scores,
                    title="Prediction Confidence Over Time",
                    markers=True
                )
                fig_confidence.update_layout(
                    xaxis_title="Question Number",
                    yaxis_title="Confidence Score",
                    yaxis=dict(range=[0, 1]),
                    height=400
                )
                st.plotly_chart(fig_confidence, use_container_width=True)
        
        # System performance metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <h4>Session Questions</h4>
                <h3>{len([c for c in st.session_state.chat_history if c['type'] == 'user'])}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if metrics:
                avg_confidence = np.mean([c.get('confidence', 0) for c in st.session_state.chat_history if c['type'] == 'bot'])
                st.markdown(f"""
                <div class="metric-container">
                    <h4>Avg Confidence</h4>
                    <h3>{avg_confidence:.1%}</h3>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-container">
                    <h4>Model Status</h4>
                    <h3>✅ Ready</h3>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            if dataset_df is not None:
                st.markdown(f"""
                <div class="metric-container">
                    <h4>Training Data</h4>
                    <h3>{len(dataset_df)} samples</h3>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-container">
                    <h4>Model Type</h4>
                    <h3>ML Ready</h3>
                </div>
                """, unsafe_allow_html=True)

    # Sidebar with additional information
    with st.sidebar:
        st.markdown("### ℹ️ About This App")
        st.info(
            "This AI-powered career guidance tool uses machine learning to analyze your interests "
            "and recommend suitable career paths. Ask questions about careers, skills, or job roles!"
        )
        
        st.markdown("### 🚀 Features")
        st.markdown("""
        - **AI Career Matching**: Get personalized career recommendations
        - **Confidence Scoring**: See how confident the AI is in its suggestions
        - **Multiple Suggestions**: View alternative career options
        - **Detailed Insights**: Comprehensive career information
        - **Performance Analytics**: Model performance metrics and trends
        """)
        
        if st.session_state.chat_history:
            st.markdown("### 📊 Session Summary")
            total_questions = len([c for c in st.session_state.chat_history if c['type'] == 'user'])
            unique_careers = len(set([c.get('career') for c in st.session_state.chat_history if c.get('career')]))
            
            st.metric("Questions Asked", total_questions)
            st.metric("Unique Careers Suggested", unique_careers)
            
            if st.button("📥 Export Chat History"):
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
                    label="Download Chat History",
                    data=csv,
                    file_name=f"career_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        st.markdown("### 🔧 Model Information")
        if model_loaded:
            st.success("✅ Model loaded successfully")
            if metrics:
                st.info(f"🎯 Model Accuracy: {metrics['accuracy']:.1%}")
                st.info(f"📚 Careers Available: {len(metrics['classes'])}")
        else:
            st.error("❌ Model not found")
        
        st.markdown("### 💡 Tips for Better Results")
        st.markdown("""
        - Be specific about your interests
        - Mention your skills and background
        - Ask about specific job roles
        - Describe your career goals
        - Ask follow-up questions for clarity
        """)

    # Handle sample input from sidebar
    if hasattr(st.session_state, 'sample_input'):
        st.text_area(
            "Ask me anything about careers:",
            value=st.session_state.sample_input,
            height=100,
            key="main_input"
        )
        del st.session_state.sample_input

if __name__ == "__main__":
    main()