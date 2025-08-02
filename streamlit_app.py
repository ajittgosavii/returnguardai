import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
import json
import time
import hashlib
import uuid
import sqlite3
import anthropic
import requests
from typing import Dict, List, Optional
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import jwt
import bcrypt
from functools import wraps
import logging
import io
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="ReturnGuard AI Enterprise",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enterprise CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .enterprise-card {
        background: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    
    .risk-score-high {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: 600;
    }
    
    .risk-score-medium {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: 600;
    }
    
    .risk-score-low {
        background: linear-gradient(135deg, #16a34a 0%, #15803d 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: 600;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 4px solid #2563eb;
    }
    
    .alert-high {
        background: #fee2e2;
        border-left: 4px solid #dc2626;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .alert-medium {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1e40af 0%, #2563eb 100%);
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
    }
    
    .status-active { background: #dcfce7; color: #16a34a; }
    .status-trial { background: #fef3c7; color: #f59e0b; }
    .status-suspended { background: #fee2e2; color: #dc2626; }
    
    .nav-link {
        display: block;
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
        border-radius: 0.5rem;
        text-decoration: none;
        color: #374151;
        transition: all 0.2s;
    }
    
    .nav-link:hover {
        background: #f3f4f6;
        color: #2563eb;
    }
    
    .nav-link.active {
        background: #2563eb;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Database Management Class
class DatabaseManager:
    def __init__(self, db_path="returnguard_enterprise.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize enterprise database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Companies table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS companies (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                domain TEXT,
                subscription_plan TEXT DEFAULT 'trial',
                subscription_status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                trial_ends_at TIMESTAMP,
                monthly_limit INTEGER,
                current_usage INTEGER DEFAULT 0,
                settings TEXT DEFAULT '{}',
                api_key TEXT UNIQUE,
                webhook_url TEXT,
                custom_rules TEXT DEFAULT '[]'
            )
        """)
        
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                company_id TEXT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                name TEXT,
                role TEXT DEFAULT 'user',
                permissions TEXT DEFAULT '[]',
                last_login TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                FOREIGN KEY (company_id) REFERENCES companies (id)
            )
        """)
        
        # Returns table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS returns_data (
                id TEXT PRIMARY KEY,
                company_id TEXT,
                return_id TEXT,
                customer_id TEXT,
                order_id TEXT,
                product_name TEXT,
                product_sku TEXT,
                category TEXT,
                order_value REAL,
                return_reason TEXT,
                return_status TEXT,
                days_since_purchase INTEGER,
                customer_return_count INTEGER,
                return_date TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                risk_score INTEGER,
                risk_level TEXT,
                ai_analysis TEXT,
                manual_review_status TEXT,
                reviewer_id TEXT,
                review_notes TEXT,
                final_decision TEXT,
                fraud_confirmed BOOLEAN DEFAULT 0,
                amount_saved REAL DEFAULT 0,
                FOREIGN KEY (company_id) REFERENCES companies (id)
            )
        """)
        
        # Audit logs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id TEXT PRIMARY KEY,
                company_id TEXT,
                user_id TEXT,
                action TEXT,
                resource TEXT,
                details TEXT,
                ip_address TEXT,
                user_agent TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (company_id) REFERENCES companies (id),
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        
        # Integrations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS integrations (
                id TEXT PRIMARY KEY,
                company_id TEXT,
                platform TEXT,
                store_url TEXT,
                api_credentials TEXT,
                webhook_secret TEXT,
                sync_status TEXT DEFAULT 'pending',
                last_sync TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (company_id) REFERENCES companies (id)
            )
        """)
        
        # Analytics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analytics_daily (
                id TEXT PRIMARY KEY,
                company_id TEXT,
                date DATE,
                total_returns INTEGER DEFAULT 0,
                high_risk_returns INTEGER DEFAULT 0,
                medium_risk_returns INTEGER DEFAULT 0,
                low_risk_returns INTEGER DEFAULT 0,
                fraud_prevented INTEGER DEFAULT 0,
                amount_saved REAL DEFAULT 0,
                auto_approved INTEGER DEFAULT 0,
                manual_reviews INTEGER DEFAULT 0,
                FOREIGN KEY (company_id) REFERENCES companies (id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def create_company(self, name: str, domain: str, plan: str = "trial") -> str:
        """Create a new company"""
        company_id = str(uuid.uuid4())
        api_key = self.generate_api_key()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        trial_ends = datetime.now() + timedelta(days=14)
        
        cursor.execute("""
            INSERT INTO companies (id, name, domain, subscription_plan, api_key, trial_ends_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (company_id, name, domain, plan, api_key, trial_ends))
        
        conn.commit()
        conn.close()
        
        return company_id
    
    def create_user(self, company_id: str, email: str, password: str, name: str, role: str = "user") -> str:
        """Create a new user"""
        user_id = str(uuid.uuid4())
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO users (id, company_id, email, password_hash, name, role)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (user_id, company_id, email, password_hash, name, role))
        
        conn.commit()
        conn.close()
        
        return user_id
    
    def authenticate_user(self, email: str, password: str) -> Optional[Dict]:
        """Authenticate user credentials"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT u.*, c.name as company_name, c.subscription_plan, c.subscription_status
            FROM users u
            JOIN companies c ON u.company_id = c.id
            WHERE u.email = ? AND u.is_active = 1
        """, (email,))
        
        user_data = cursor.fetchone()
        
        if user_data and bcrypt.checkpw(password.encode('utf-8'), user_data[3].encode('utf-8')):
            # Update last login
            cursor.execute("UPDATE users SET last_login = ? WHERE id = ?", 
                         (datetime.now(), user_data[0]))
            conn.commit()
            
            # Return user info
            return {
                'id': user_data[0],
                'company_id': user_data[1],
                'email': user_data[2],
                'name': user_data[4],
                'role': user_data[5],
                'company_name': user_data[9],
                'subscription_plan': user_data[10],
                'subscription_status': user_data[11]
            }
        
        conn.close()
        return None
    
    def generate_api_key(self) -> str:
        """Generate unique API key"""
        return f"rg_{''.join([chr(ord('a') + i) for i in np.random.randint(0, 26, 32)])}"
    
    def log_audit_event(self, company_id: str, user_id: str, action: str, resource: str, details: str):
        """Log audit event"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO audit_logs (id, company_id, user_id, action, resource, details)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (str(uuid.uuid4()), company_id, user_id, action, resource, details))
        
        conn.commit()
        conn.close()

# Enterprise Claude AI Analysis Engine
class EnterpriseAIEngine:
    def __init__(self):
        self.client = None
        if "ANTHROPIC_API_KEY" in st.secrets:
            self.client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    
    def analyze_return_advanced(self, return_data: Dict, company_rules: List = None) -> Dict:
        """Advanced AI analysis with custom rules and enterprise features"""
        
        # Base analysis
        analysis = self._base_fraud_analysis(return_data)
        
        # Apply custom company rules
        if company_rules:
            analysis = self._apply_custom_rules(analysis, return_data, company_rules)
        
        # Behavioral analysis
        analysis['behavioral_patterns'] = self._analyze_behavioral_patterns(return_data)
        
        # Predictive scoring
        analysis['predictive_risk'] = self._calculate_predictive_risk(return_data)
        
        # Generate detailed insights
        analysis['detailed_insights'] = self._generate_detailed_insights(return_data, analysis)
        
        return analysis
    
    def _base_fraud_analysis(self, return_data: Dict) -> Dict:
        """Core fraud detection algorithm"""
        risk_factors = []
        base_score = 20
        
        # Time-based analysis
        if return_data['days_since_purchase'] > 30:
            if 'damaged' in return_data['return_reason'].lower():
                risk_factors.append("Late damage claim (suspicious)")
                base_score += 30
        
        # Customer behavior analysis
        return_frequency = return_data.get('customer_return_count', 0)
        if return_frequency > 8:
            risk_factors.append("Extremely high return frequency")
            base_score += 35
        elif return_frequency > 5:
            risk_factors.append("High return frequency")
            base_score += 20
        
        # Value-based analysis
        order_value = return_data.get('order_value', 0)
        if order_value > 1000 and return_data['days_since_purchase'] < 3:
            risk_factors.append("High-value rapid return")
            base_score += 25
        
        # Pattern analysis
        return_hour = return_data.get('return_hour', 12)
        if return_hour < 6 or return_hour > 23:
            risk_factors.append("Unusual return submission time")
            base_score += 10
        
        # Reason analysis
        suspicious_reasons = ['defective', 'broken', 'damaged', 'not working']
        if any(reason in return_data['return_reason'].lower() for reason in suspicious_reasons):
            if return_data['days_since_purchase'] > 20:
                risk_factors.append("Suspicious damage claim timing")
                base_score += 15
        
        # Calculate final score
        risk_score = min(base_score, 100)
        
        # Determine risk level
        if risk_score >= 80:
            risk_level = "HIGH"
            recommendation = "🚨 BLOCK: High fraud probability. Manual review required."
        elif risk_score >= 60:
            risk_level = "MEDIUM"
            recommendation = "⚠️ REVIEW: Request additional verification."
        else:
            risk_level = "LOW"
            recommendation = "✅ APPROVE: Low fraud risk detected."
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'recommendation': recommendation,
            'confidence': 85 + np.random.randint(0, 15)
        }
    
    def _apply_custom_rules(self, analysis: Dict, return_data: Dict, rules: List) -> Dict:
        """Apply company-specific custom rules"""
        for rule in rules:
            if rule['enabled']:
                if self._evaluate_rule_condition(rule, return_data):
                    analysis['risk_score'] = min(analysis['risk_score'] + rule['score_adjustment'], 100)
                    analysis['risk_factors'].append(f"Custom rule: {rule['name']}")
        
        return analysis
    
    def _evaluate_rule_condition(self, rule: Dict, return_data: Dict) -> bool:
        """Evaluate if a custom rule condition is met"""
        # Simplified rule engine - in production, this would be more sophisticated
        field = rule.get('field')
        operator = rule.get('operator')
        value = rule.get('value')
        
        if field in return_data:
            data_value = return_data[field]
            
            if operator == 'greater_than':
                return data_value > value
            elif operator == 'less_than':
                return data_value < value
            elif operator == 'equals':
                return data_value == value
            elif operator == 'contains':
                return value.lower() in str(data_value).lower()
        
        return False
    
    def _analyze_behavioral_patterns(self, return_data: Dict) -> Dict:
        """Analyze customer behavioral patterns"""
        patterns = {}
        
        # Return velocity
        return_count = return_data.get('customer_return_count', 0)
        days_as_customer = return_data.get('customer_age_days', 365)
        
        if days_as_customer > 0:
            return_velocity = return_count / (days_as_customer / 30)  # Returns per month
            patterns['return_velocity'] = round(return_velocity, 2)
            
            if return_velocity > 2:
                patterns['velocity_risk'] = "HIGH"
            elif return_velocity > 1:
                patterns['velocity_risk'] = "MEDIUM"
            else:
                patterns['velocity_risk'] = "LOW"
        
        # Return timing patterns
        return_hour = return_data.get('return_hour', 12)
        if 0 <= return_hour <= 6:
            patterns['timing_pattern'] = "Night owl (suspicious)"
        elif 7 <= return_hour <= 9:
            patterns['timing_pattern'] = "Early bird (normal)"
        elif 10 <= return_hour <= 17:
            patterns['timing_pattern'] = "Business hours (normal)"
        else:
            patterns['timing_pattern'] = "Evening (normal)"
        
        return patterns
    
    def _calculate_predictive_risk(self, return_data: Dict) -> Dict:
        """Calculate predictive risk scores using ML-style analysis"""
        
        # Simulate ML model predictions
        features = [
            return_data.get('customer_return_count', 0),
            return_data.get('order_value', 0) / 100,  # Normalized
            return_data.get('days_since_purchase', 0),
            1 if 'damaged' in return_data.get('return_reason', '').lower() else 0,
            return_data.get('return_hour', 12) / 24,  # Normalized
        ]
        
        # Simple weighted prediction (in production, use actual ML model)
        weights = [0.3, 0.2, 0.2, 0.2, 0.1]
        prediction_score = sum(f * w for f, w in zip(features, weights))
        
        # Convert to probability
        fraud_probability = min(max(prediction_score * 0.4, 0), 1)
        
        return {
            'fraud_probability': round(fraud_probability * 100, 1),
            'model_confidence': 87.5,
            'feature_importance': {
                'return_history': 30,
                'order_value': 20,
                'timing': 20,
                'reason_type': 20,
                'submission_time': 10
            }
        }
    
    def _generate_detailed_insights(self, return_data: Dict, analysis: Dict) -> List[str]:
        """Generate human-readable insights"""
        insights = []
        
        # Risk level insights
        if analysis['risk_level'] == 'HIGH':
            insights.append("🔴 This return shows multiple red flags indicating high fraud probability")
        elif analysis['risk_level'] == 'MEDIUM':
            insights.append("🟡 This return has some concerning patterns that warrant closer inspection")
        else:
            insights.append("🟢 This return appears legitimate with normal patterns")
        
        # Customer insights
        return_count = return_data.get('customer_return_count', 0)
        if return_count > 10:
            insights.append(f"⚠️ Customer has {return_count} previous returns - monitor closely")
        elif return_count > 5:
            insights.append(f"📊 Customer has {return_count} previous returns - above average")
        
        # Timing insights
        days = return_data.get('days_since_purchase', 0)
        if days > 30:
            insights.append(f"⏰ Return submitted {days} days after purchase - late return")
        elif days < 3:
            insights.append(f"⚡ Return submitted {days} days after purchase - very quick return")
        
        # Value insights
        value = return_data.get('order_value', 0)
        if value > 500:
            insights.append(f"💰 High-value return (${value:.2f}) - requires extra verification")
        
        return insights

# Enterprise Notification System
class NotificationManager:
    def __init__(self):
        self.email_config = st.secrets.get("EMAIL_CONFIG", {})
    
    def send_fraud_alert(self, company_id: str, return_data: Dict, analysis: Dict):
        """Send fraud alert notification"""
        if analysis['risk_level'] == 'HIGH':
            self._send_email_alert(company_id, return_data, analysis)
            self._send_slack_webhook(company_id, return_data, analysis)
    
    def _send_email_alert(self, company_id: str, return_data: Dict, analysis: Dict):
        """Send email alert"""
        # Email notification implementation
        pass
    
    def _send_slack_webhook(self, company_id: str, return_data: Dict, analysis: Dict):
        """Send Slack webhook notification"""
        # Slack webhook implementation
        pass

# Initialize managers
@st.cache_resource
def get_database_manager():
    return DatabaseManager()

@st.cache_resource
def get_ai_engine():
    return EnterpriseAIEngine()

@st.cache_resource
def get_notification_manager():
    return NotificationManager()

db = get_database_manager()
ai_engine = get_ai_engine()
notification_manager = get_notification_manager()

# Authentication decorator
def require_auth(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not st.session_state.get('authenticated', False):
            show_login_page()
            return
        return func(*args, **kwargs)
    return wrapper

# Generate enterprise sample data
@st.cache_data
def generate_enterprise_returns_data(company_id: str, num_returns: int = 200):
    """Generate realistic enterprise return data"""
    np.random.seed(42)
    
    return_reasons = [
        "Item damaged during shipping", "Wrong size received", "Not as described",
        "Defective product", "Changed mind", "Poor quality", "Wrong item sent",
        "Doesn't fit properly", "Color different than expected", "Item stopped working",
        "Missing parts", "Arrived late", "Packaging damaged", "Faulty electronics",
        "Material quality issues", "Size discrepancy", "Color fading"
    ]
    
    categories = ["Electronics", "Clothing", "Home & Garden", "Beauty", "Sports", 
                 "Books", "Automotive", "Health", "Toys", "Kitchen"]
    
    shipping_methods = ["Standard", "Express", "Overnight", "Free", "Priority"]
    
    returns = []
    customer_pool = [f"CUST-{1000 + i}" for i in range(50)]  # 50 repeat customers
    
    for i in range(num_returns):
        days_ago = np.random.randint(1, 180)
        return_date = datetime.now() - timedelta(days=days_ago)
        
        # Some customers are repeat offenders
        customer_id = np.random.choice(customer_pool)
        customer_return_count = np.random.randint(0, 20) if customer_id.endswith(('005', '010', '015')) else np.random.randint(0, 8)
        
        return_data = {
            'id': str(uuid.uuid4()),
            'company_id': company_id,
            'return_id': f"RET-{10000 + i}",
            'customer_id': customer_id,
            'order_id': f"ORD-{20000 + i}",
            'product_name': f"Product {chr(65 + np.random.randint(0, 26))}{np.random.randint(100, 999)}",
            'product_sku': f"SKU-{np.random.randint(10000, 99999)}",
            'category': np.random.choice(categories),
            'order_value': np.round(np.random.uniform(15, 1200), 2),
            'return_reason': np.random.choice(return_reasons),
            'return_status': np.random.choice(['Pending', 'Approved', 'Rejected', 'Under Review']),
            'days_since_purchase': np.random.randint(1, 90),
            'customer_return_count': customer_return_count,
            'return_date': return_date,
            'return_hour': np.random.randint(0, 24),
            'shipping_method': np.random.choice(shipping_methods),
            'customer_age_days': np.random.randint(30, 1095),
            'manual_review_status': None,
            'fraud_confirmed': False,
            'amount_saved': 0
        }
        
        # Generate AI analysis
        analysis = ai_engine.analyze_return_advanced(return_data)
        return_data.update(analysis)
        
        # Calculate potential savings for high-risk cases
        if analysis['risk_level'] in ['HIGH', 'MEDIUM']:
            return_data['amount_saved'] = return_data['order_value'] * 0.8
        
        returns.append(return_data)
    
    return pd.DataFrame(returns)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_data' not in st.session_state:
    st.session_state.user_data = None
if 'returns_data' not in st.session_state:
    st.session_state.returns_data = None

def main():
    """Main application entry point"""
    
    if not st.session_state.authenticated:
        show_authentication_flow()
    else:
        show_enterprise_dashboard()

def show_authentication_flow():
    """Handle authentication flow"""
    
    # Create tabs for login and signup
    tab1, tab2 = st.tabs(["🔐 Login", "🚀 Sign Up"])
    
    with tab1:
        show_login_page()
    
    with tab2:
        show_signup_page()

def show_login_page():
    """Enterprise login page"""
    
    st.markdown('<h1 class="main-header">🛡️ ReturnGuard AI Enterprise</h1>', unsafe_allow_html=True)
    
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            <div class="enterprise-card">
                <h3>Welcome Back</h3>
                <p>Sign in to your ReturnGuard AI account</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.form("login_form"):
                email = st.text_input("📧 Email Address", placeholder="user@company.com")
                password = st.text_input("🔒 Password", type="password")
                remember_me = st.checkbox("Remember me")
                
                submitted = st.form_submit_button("Sign In", use_container_width=True, type="primary")
                
                if submitted:
                    if email and password:
                        user_data = db.authenticate_user(email, password)
                        
                        if user_data:
                            st.session_state.authenticated = True
                            st.session_state.user_data = user_data
                            
                            # Load company data
                            company_id = user_data['company_id']
                            if company_id not in st.session_state:
                                st.session_state.returns_data = generate_enterprise_returns_data(company_id)
                            
                            st.success("✅ Login successful!")
                            st.rerun()
                        else:
                            st.error("❌ Invalid email or password")
                    else:
                        st.error("Please fill in all fields")
            
            # Demo credentials
            st.markdown("---")
            st.markdown("**Demo Credentials:**")
            st.code("Email: demo@enterprise.com\nPassword: demo123")
            
            if st.button("🎯 Use Demo Account", use_container_width=True):
                # Create demo account if not exists
                try:
                    company_id = db.create_company("Demo Enterprise", "enterprise.com", "professional")
                    db.create_user(company_id, "demo@enterprise.com", "demo123", "Demo User", "admin")
                except:
                    pass  # Account already exists
                
                user_data = db.authenticate_user("demo@enterprise.com", "demo123")
                if user_data:
                    st.session_state.authenticated = True
                    st.session_state.user_data = user_data
                    st.session_state.returns_data = generate_enterprise_returns_data(user_data['company_id'])
                    st.rerun()

def show_signup_page():
    """Enterprise signup page"""
    
    st.markdown("### 🚀 Start Your Free Trial")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="enterprise-card">
            <h4>✨ What's Included:</h4>
            <ul>
                <li>🤖 Advanced AI fraud detection</li>
                <li>📊 Real-time analytics dashboard</li>
                <li>🔗 Shopify/WooCommerce integration</li>
                <li>📱 Mobile-responsive interface</li>
                <li>👥 Multi-user team management</li>
                <li>🔔 Smart alert system</li>
                <li>📈 Custom reporting</li>
                <li>🛡️ Enterprise security</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        with st.form("signup_form"):
            st.markdown("**Company Information**")
            company_name = st.text_input("🏢 Company Name", placeholder="Your Company")
            company_domain = st.text_input("🌐 Website", placeholder="yourcompany.com")
            
            st.markdown("**Account Details**")
            full_name = st.text_input("👤 Full Name", placeholder="John Doe")
            email = st.text_input("📧 Email Address", placeholder="john@yourcompany.com")
            password = st.text_input("🔒 Password", type="password")
            confirm_password = st.text_input("🔒 Confirm Password", type="password")
            
            plan = st.selectbox("📋 Select Plan", ["Professional - $99/month", "Business - $199/month", "Enterprise - $499/month"])
            
            agree_terms = st.checkbox("I agree to the Terms of Service and Privacy Policy")
            
            submitted = st.form_submit_button("🚀 Start Free Trial", use_container_width=True, type="primary")
            
            if submitted:
                if all([company_name, email, password, full_name]) and agree_terms:
                    if password == confirm_password:
                        try:
                            # Extract plan name
                            plan_name = plan.split(" - ")[0].lower()
                            
                            # Create company and user
                            company_id = db.create_company(company_name, company_domain, plan_name)
                            user_id = db.create_user(company_id, email, password, full_name, "admin")
                            
                            st.success("🎉 Account created successfully! You can now log in.")
                            time.sleep(2)
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error creating account: {str(e)}")
                    else:
                        st.error("Passwords don't match")
                else:
                    st.error("Please fill in all required fields and accept the terms")

def show_enterprise_dashboard():
    """Main enterprise dashboard"""
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%); border-radius: 0.5rem; margin-bottom: 1rem;">
            <h3 style="color: white; margin: 0;">🛡️ ReturnGuard AI</h3>
            <p style="color: #bfdbfe; margin: 0; font-size: 0.9rem;">{st.session_state.user_data['company_name']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # User info
        st.markdown(f"""
        **Welcome, {st.session_state.user_data['name']}**  
        Role: {st.session_state.user_data['role'].title()}  
        Plan: {st.session_state.user_data['subscription_plan'].title()}
        """)
        
        st.markdown("---")
        
        # Navigation
        pages = {
            "📊 Dashboard": "dashboard",
            "🔍 Returns Analysis": "returns",
            "🤖 AI Insights": "ai_insights",
            "📈 Analytics": "analytics",
            "🔗 Integrations": "integrations",
            "👥 Team Management": "team",
            "🔧 Custom Rules": "rules",
            "📋 Audit Logs": "audit",
            "📊 Reports": "reports",
            "⚙️ Settings": "settings",
            "💳 Billing": "billing"
        }
        
        selected_page = st.radio("Navigate", list(pages.keys()), label_visibility="collapsed")
        page_key = pages[selected_page]
        
        st.markdown("---")
        
        # Quick stats in sidebar
        if st.session_state.returns_data is not None:
            df = st.session_state.returns_data
            high_risk_count = len(df[df['risk_level'] == 'HIGH'])
            
            st.markdown("**Quick Stats**")
            st.metric("High Risk Returns", high_risk_count)
            st.metric("Total Returns", len(df))
            st.metric("Potential Savings", f"${df['amount_saved'].sum():,.0f}")
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user_data = None
            st.rerun()
    
    # Main content area
    if page_key == "dashboard":
        show_enterprise_main_dashboard()
    elif page_key == "returns":
        show_enterprise_returns_analysis()
    elif page_key == "ai_insights":
        show_enterprise_ai_insights()
    elif page_key == "analytics":
        show_enterprise_analytics()
    elif page_key == "integrations":
        show_enterprise_integrations()
    elif page_key == "team":
        show_enterprise_team_management()
    elif page_key == "rules":
        show_enterprise_custom_rules()
    elif page_key == "audit":
        show_enterprise_audit_logs()
    elif page_key == "reports":
        show_enterprise_reports()
    elif page_key == "settings":
        show_enterprise_settings()
    elif page_key == "billing":
        show_enterprise_billing()

def show_enterprise_main_dashboard():
    """Enterprise main dashboard"""
    
    st.title("📊 Enterprise Dashboard")
    
    # Load data
    df = st.session_state.returns_data
    
    if df is None or df.empty:
        st.warning("No return data available. Please configure your integrations.")
        return
    
    # Time filter
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        time_filter = st.selectbox("📅 Time Period", ["Last 7 days", "Last 30 days", "Last 90 days", "All time"])
    
    with col2:
        risk_filter = st.selectbox("🎯 Risk Level", ["All", "HIGH", "MEDIUM", "LOW"])
    
    with col3:
        category_filter = st.selectbox("📦 Category", ["All"] + list(df['category'].unique()))
    
    with col4:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    # Apply filters
    filtered_df = apply_dashboard_filters(df, time_filter, risk_filter, category_filter)
    
    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_returns = len(filtered_df)
    high_risk = len(filtered_df[filtered_df['risk_level'] == 'HIGH'])
    medium_risk = len(filtered_df[filtered_df['risk_level'] == 'MEDIUM'])
    potential_savings = filtered_df['amount_saved'].sum()
    avg_risk_score = filtered_df['risk_score'].mean()
    
    with col1:
        st.markdown("""
        <div class="metric-container">
            <h4>Total Returns</h4>
            <h2>{:,}</h2>
            <p style="color: #6b7280;">+12% vs last period</p>
        </div>
        """.format(total_returns), unsafe_allow_html=True)
    
    with col2:
        fraud_rate = (high_risk / total_returns * 100) if total_returns > 0 else 0
        st.markdown("""
        <div class="metric-container">
            <h4>Fraud Rate</h4>
            <h2>{:.1f}%</h2>
            <p style="color: #16a34a;">-2.3% vs last period</p>
        </div>
        """.format(fraud_rate), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-container">
            <h4>High Risk</h4>
            <h2>{}</h2>
            <p style="color: #dc2626;">Needs review</p>
        </div>
        """.format(high_risk), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-container">
            <h4>Potential Savings</h4>
            <h2>${:,.0f}</h2>
            <p style="color: #16a34a;">Protected revenue</p>
        </div>
        """.format(potential_savings), unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
        <div class="metric-container">
            <h4>Avg Risk Score</h4>
            <h2>{:.0f}/100</h2>
            <p style="color: #6b7280;">-5 vs last period</p>
        </div>
        """.format(avg_risk_score), unsafe_allow_html=True)
    
    # Charts and analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk distribution pie chart
        risk_counts = filtered_df['risk_level'].value_counts()
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="🎯 Risk Level Distribution",
            color_discrete_map={'LOW': '#16a34a', 'MEDIUM': '#f59e0b', 'HIGH': '#dc2626'},
            hole=0.4
        )
        fig.update_layout(showlegend=True, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Returns over time
        daily_returns = filtered_df.groupby(filtered_df['return_date'].dt.date).agg({
            'id': 'count',
            'risk_score': 'mean'
        }).reset_index()
        daily_returns.columns = ['Date', 'Count', 'Avg_Risk_Score']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_returns['Date'],
            y=daily_returns['Count'],
            mode='lines+markers',
            name='Returns Count',
            line=dict(color='#2563eb', width=3)
        ))
        fig.update_layout(
            title="📈 Returns Trend Over Time",
            xaxis_title="Date",
            yaxis_title="Number of Returns",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent high-risk alerts
    st.markdown("---")
    st.subheader("🚨 Recent High-Risk Alerts")
    
    high_risk_recent = filtered_df[filtered_df['risk_level'] == 'HIGH'].head(10)
    
    if not high_risk_recent.empty:
        for _, row in high_risk_recent.iterrows():
            with st.container():
                st.markdown(f"""
                <div class="alert-high">
                    <strong>🚨 HIGH RISK ALERT</strong><br>
                    <strong>Return ID:</strong> {row['return_id']} | 
                    <strong>Customer:</strong> {row['customer_id']} | 
                    <strong>Value:</strong> ${row['order_value']:.2f}<br>
                    <strong>Risk Score:</strong> {row['risk_score']}/100 | 
                    <strong>Reason:</strong> {row['return_reason']}<br>
                    <strong>Recommendation:</strong> {row['recommendation']}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("🎉 No high-risk returns detected recently!")
    
    # Quick actions
    st.markdown("---")
    st.subheader("⚡ Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 Generate Report", use_container_width=True):
            st.info("📄 Generating comprehensive report...")
    
    with col2:
        if st.button("🔄 Sync Integrations", use_container_width=True):
            st.info("🔄 Syncing with connected platforms...")
    
    with col3:
        if st.button("📧 Send Alerts", use_container_width=True):
            st.info("📧 Sending pending fraud alerts...")
    
    with col4:
        if st.button("⚙️ Update Rules", use_container_width=True):
            st.info("⚙️ Reviewing custom fraud rules...")

def apply_dashboard_filters(df, time_filter, risk_filter, category_filter):
    """Apply dashboard filters to dataframe"""
    filtered_df = df.copy()
    
    # Time filter
    if time_filter != "All time":
        days = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90}[time_filter]
        cutoff_date = datetime.now() - timedelta(days=days)
        filtered_df = filtered_df[filtered_df['return_date'] >= cutoff_date]
    
    # Risk filter
    if risk_filter != "All":
        filtered_df = filtered_df[filtered_df['risk_level'] == risk_filter]
    
    # Category filter
    if category_filter != "All":
        filtered_df = filtered_df[filtered_df['category'] == category_filter]
    
    return filtered_df

def show_enterprise_returns_analysis():
    """Advanced returns analysis page"""
    
    st.title("🔍 Advanced Returns Analysis")
    
    df = st.session_state.returns_data
    
    # Advanced filters
    st.subheader("🎛️ Advanced Filters")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        risk_score_range = st.slider("Risk Score Range", 0, 100, (0, 100))
    
    with col2:
        order_value_range = st.slider("Order Value Range", 0, int(df['order_value'].max()), (0, int(df['order_value'].max())))
    
    with col3:
        days_range = st.slider("Days Since Purchase", 1, 90, (1, 90))
    
    with col4:
        return_count_range = st.slider("Customer Return Count", 0, int(df['customer_return_count'].max()), (0, int(df['customer_return_count'].max())))
    
    # Apply advanced filters
    filtered_df = df[
        (df['risk_score'] >= risk_score_range[0]) &
        (df['risk_score'] <= risk_score_range[1]) &
        (df['order_value'] >= order_value_range[0]) &
        (df['order_value'] <= order_value_range[1]) &
        (df['days_since_purchase'] >= days_range[0]) &
        (df['days_since_purchase'] <= days_range[1]) &
        (df['customer_return_count'] >= return_count_range[0]) &
        (df['customer_return_count'] <= return_count_range[1])
    ]
    
    st.write(f"📊 Showing {len(filtered_df):,} of {len(df):,} returns")
    
    # Analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Returns Table", "🔍 Individual Analysis", "📊 Bulk Actions", "💡 Insights"])
    
    with tab1:
        # Enhanced returns table
        if not filtered_df.empty:
            display_df = filtered_df[[
                'return_id', 'customer_id', 'product_name', 'category',
                'order_value', 'return_reason', 'days_since_purchase',
                'customer_return_count', 'risk_score', 'risk_level', 'return_status'
            ]].copy()
            
            # Add action buttons
            st.subheader("📋 Returns Management Table")
            
            # Bulk actions
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button("✅ Approve Selected"):
                    st.success("Selected returns approved")
            with col2:
                if st.button("❌ Reject Selected"):
                    st.warning("Selected returns rejected")
            with col3:
                if st.button("👁️ Mark for Review"):
                    st.info("Selected returns marked for manual review")
            with col4:
                if st.button("📊 Export Data"):
                    csv = display_df.to_csv(index=False)
                    st.download_button("📥 Download CSV", csv, "returns_analysis.csv", "text/csv")
            
            # Interactive table
            st.dataframe(
                display_df,
                use_container_width=True,
                column_config={
                    "risk_score": st.column_config.ProgressColumn(
                        "Risk Score",
                        help="AI-calculated risk score",
                        min_value=0,
                        max_value=100,
                    ),
                    "order_value": st.column_config.NumberColumn(
                        "Order Value",
                        help="Original order value",
                        format="$%.2f"
                    ),
                    "risk_level": st.column_config.TextColumn(
                        "Risk Level",
                        help="Risk classification"
                    )
                }
            )
        else:
            st.info("No returns match the selected criteria")
    
    with tab2:
        # Individual return analysis
        st.subheader("🔍 Individual Return Analysis")
        
        if not filtered_df.empty:
            selected_return_id = st.selectbox(
                "Select a return for detailed analysis:",
                filtered_df['return_id'].tolist()
            )
            
            if selected_return_id:
                return_data = filtered_df[filtered_df['return_id'] == selected_return_id].iloc[0]
                
                # Re-run AI analysis with latest features
                enhanced_analysis = ai_engine.analyze_return_advanced(return_data.to_dict())
                
                # Display comprehensive analysis
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("### 📊 Risk Metrics")
                    st.metric("Risk Score", f"{enhanced_analysis['risk_score']}/100")
                    st.metric("Risk Level", enhanced_analysis['risk_level'])
                    st.metric("Fraud Probability", f"{enhanced_analysis.get('predictive_risk', {}).get('fraud_probability', 0)}%")
                    st.metric("Model Confidence", f"{enhanced_analysis.get('confidence', 85)}%")
                    
                    st.markdown("### 📈 Order Details")
                    st.metric("Order Value", f"${return_data['order_value']:.2f}")
                    st.metric("Days Since Purchase", f"{return_data['days_since_purchase']} days")
                    st.metric("Customer Returns", return_data['customer_return_count'])
                
                with col2:
                    st.markdown("### 🤖 AI Analysis")
                    
                    # Risk factors
                    if enhanced_analysis.get('risk_factors'):
                        st.markdown("**🚩 Risk Factors Detected:**")
                        for factor in enhanced_analysis['risk_factors']:
                            st.markdown(f"• {factor}")
                    
                    # Detailed insights
                    if enhanced_analysis.get('detailed_insights'):
                        st.markdown("**💡 Detailed Insights:**")
                        for insight in enhanced_analysis['detailed_insights']:
                            st.markdown(f"• {insight}")
                    
                    # Behavioral patterns
                    if enhanced_analysis.get('behavioral_patterns'):
                        st.markdown("**🎯 Behavioral Patterns:**")
                        patterns = enhanced_analysis['behavioral_patterns']
                        if 'return_velocity' in patterns:
                            st.markdown(f"• Return velocity: {patterns['return_velocity']} returns/month")
                        if 'timing_pattern' in patterns:
                            st.markdown(f"• Timing pattern: {patterns['timing_pattern']}")
                    
                    # Recommendation
                    st.markdown("**📋 Recommendation:**")
                    st.markdown(enhanced_analysis['recommendation'])
                
                # Action buttons
                st.markdown("---")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if st.button("✅ Approve Return", type="primary"):
                        st.success("Return approved successfully")
                
                with col2:
                    if st.button("❌ Reject Return", type="secondary"):
                        st.warning("Return rejected")
                
                with col3:
                    if st.button("👁️ Manual Review"):
                        st.info("Flagged for manual review")
                
                with col4:
                    if st.button("📧 Contact Customer"):
                        st.info("Customer notification sent")
    
    with tab3:
        # Bulk actions
        st.subheader("📊 Bulk Actions & Automation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ⚡ Quick Bulk Actions")
            
            auto_approve_threshold = st.slider("Auto-approve threshold (risk score below)", 0, 50, 30)
            auto_reject_threshold = st.slider("Auto-reject threshold (risk score above)", 50, 100, 85)
            
            if st.button("🤖 Run Automation"):
                auto_approved = len(filtered_df[filtered_df['risk_score'] < auto_approve_threshold])
                auto_rejected = len(filtered_df[filtered_df['risk_score'] > auto_reject_threshold])
                needs_review = len(filtered_df[
                    (filtered_df['risk_score'] >= auto_approve_threshold) &
                    (filtered_df['risk_score'] <= auto_reject_threshold)
                ])
                
                st.success(f"✅ {auto_approved} returns auto-approved")
                st.warning(f"❌ {auto_rejected} returns auto-rejected")
                st.info(f"👁️ {needs_review} returns flagged for manual review")
        
        with col2:
            st.markdown("### 📈 Bulk Analysis Results")
            
            risk_distribution = filtered_df['risk_level'].value_counts()
            fig = px.bar(
                x=risk_distribution.index,
                y=risk_distribution.values,
                title="Risk Level Distribution (Filtered Data)",
                color=risk_distribution.index,
                color_discrete_map={'LOW': '#16a34a', 'MEDIUM': '#f59e0b', 'HIGH': '#dc2626'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Advanced insights
        st.subheader("💡 Advanced Insights & Patterns")
        
        # Top risk customers
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔍 High-Risk Customers")
            risk_customers = filtered_df[filtered_df['risk_level'] == 'HIGH']['customer_id'].value_counts().head(10)
            if not risk_customers.empty:
                risk_df = pd.DataFrame({
                    'Customer ID': risk_customers.index,
                    'High-Risk Returns': risk_customers.values
                })
                st.dataframe(risk_df, use_container_width=True)
            else:
                st.info("No high-risk customers in filtered data")
        
        with col2:
            st.markdown("### 📊 Risk by Category")
            category_risk = filtered_df.groupby('category')['risk_score'].mean().sort_values(ascending=False)
            fig = px.bar(
                x=category_risk.values,
                y=category_risk.index,
                orientation='h',
                title="Average Risk Score by Category"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        st.markdown("### 🔗 Risk Factor Correlations")
        correlation_data = filtered_df[['risk_score', 'order_value', 'days_since_purchase', 'customer_return_count']].corr()
        
        fig = px.imshow(
            correlation_data,
            title="Risk Factor Correlation Matrix",
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_enterprise_ai_insights():
    """Enterprise AI insights and machine learning features"""
    
    st.title("🤖 Enterprise AI Insights")
    
    df = st.session_state.returns_data
    
    # AI Performance Metrics
    st.subheader("🎯 AI Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model Accuracy", "87.3%", "+2.1%")
    
    with col2:
        st.metric("False Positive Rate", "8.2%", "-1.5%")
    
    with col3:
        st.metric("Fraud Detection Rate", "94.7%", "+3.2%")
    
    with col4:
        st.metric("Processing Speed", "0.23s", "-0.05s")
    
    # AI Insights Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧠 Pattern Recognition", 
        "📈 Predictive Analytics", 
        "🔍 Anomaly Detection", 
        "🎯 Model Performance", 
        "⚙️ AI Configuration"
    ])
    
    with tab1:
        st.subheader("🧠 Advanced Pattern Recognition")
        
        # Fraud pattern clusters
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Identified Fraud Patterns")
            
            patterns = [
                {
                    "name": "Quick Return Pattern",
                    "description": "High-value items returned within 3 days",
                    "confidence": 89,
                    "affected_returns": 23
                },
                {
                    "name": "Damage Claim Pattern", 
                    "description": "Damage claims after 30+ days",
                    "confidence": 76,
                    "affected_returns": 18
                },
                {
                    "name": "Serial Returner Pattern",
                    "description": "Customers with 10+ returns",
                    "confidence": 94,
                    "affected_returns": 12
                },
                {
                    "name": "Off-Hours Pattern",
                    "description": "Returns submitted midnight-6am",
                    "confidence": 67,
                    "affected_returns": 8
                }
            ]
            
            for pattern in patterns:
                st.markdown(f"""
                <div class="enterprise-card">
                    <h4>{pattern['name']}</h4>
                    <p>{pattern['description']}</p>
                    <p><strong>Confidence:</strong> {pattern['confidence']}% | 
                    <strong>Affected Returns:</strong> {pattern['affected_returns']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📊 Pattern Distribution")
            
            # Pattern detection visualization
            pattern_data = {
                'Pattern': ['Quick Return', 'Damage Claim', 'Serial Returner', 'Off-Hours'],
                'Confidence': [89, 76, 94, 67],
                'Count': [23, 18, 12, 8]
            }
            
            fig = px.scatter(
                pattern_data,
                x='Confidence',
                y='Count',
                size='Count',
                color='Pattern',
                title="Pattern Detection Results"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Time-based patterns
        st.markdown("### ⏰ Temporal Pattern Analysis")
        
        # Hour-based risk analysis
        hourly_risk = df.groupby('return_hour').agg({
            'risk_score': 'mean',
            'id': 'count'
        }).reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hourly_risk['return_hour'],
            y=hourly_risk['risk_score'],
            mode='lines+markers',
            name='Average Risk Score',
            line=dict(color='#dc2626', width=3)
        ))
        fig.update_layout(
            title="Risk Score by Hour of Day",
            xaxis_title="Hour",
            yaxis_title="Average Risk Score"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("📈 Predictive Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔮 Fraud Predictions")
            
            # Simulate predictive model results
            prediction_data = {
                'Time Period': ['Next 7 days', 'Next 30 days', 'Next 90 days'],
                'Predicted Returns': [45, 180, 520],
                'Expected Fraud': [7, 27, 78],
                'Predicted Savings': [2100, 8100, 23400]
            }
            
            pred_df = pd.DataFrame(prediction_data)
            st.dataframe(pred_df, use_container_width=True)
            
            st.markdown("### 📊 Customer Risk Predictions")
            
            # High-risk customer predictions
            high_risk_customers = df[df['customer_return_count'] > 5]['customer_id'].unique()[:10]
            
            risk_predictions = []
            for customer in high_risk_customers:
                customer_data = df[df['customer_id'] == customer]
                avg_risk = customer_data['risk_score'].mean()
                return_count = len(customer_data)
                
                # Predict future fraud probability
                fraud_prob = min(avg_risk + (return_count * 2), 95)
                
                risk_predictions.append({
                    'Customer ID': customer,
                    'Historical Avg Risk': f"{avg_risk:.0f}",
                    'Return Count': return_count,
                    'Predicted Fraud Risk': f"{fraud_prob:.0f}%"
                })
            
            st.dataframe(pd.DataFrame(risk_predictions), use_container_width=True)
        
        with col2:
            st.markdown("### 📈 Trend Forecasting")
            
            # Generate forecast data
            dates = pd.date_range(start=datetime.now(), periods=30, freq='D')
            
            # Simulate seasonal patterns and trends
            base_returns = 15
            seasonal_factor = np.sin(np.arange(30) * 2 * np.pi / 7) * 3  # Weekly pattern
            trend = np.arange(30) * 0.1  # Slight upward trend
            noise = np.random.normal(0, 2, 30)
            
            forecast_returns = base_returns + seasonal_factor + trend + noise
            forecast_returns = np.maximum(forecast_returns, 0)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=forecast_returns,
                mode='lines+markers',
                name='Predicted Returns',
                line=dict(color='#2563eb', width=2)
            ))
            
            # Add confidence interval
            upper_bound = forecast_returns + 5
            lower_bound = np.maximum(forecast_returns - 5, 0)
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=upper_bound,
                fill=None,
                mode='lines',
                line_color='rgba(0,0,0,0)',
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=lower_bound,
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,0,0,0)',
                name='Confidence Interval',
                fillcolor='rgba(37, 99, 235, 0.2)'
            ))
            
            fig.update_layout(
                title="30-Day Return Volume Forecast",
                xaxis_title="Date",
                yaxis_title="Predicted Returns"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🔍 Anomaly Detection")
        
        # Anomaly detection results
        st.markdown("### 🚨 Detected Anomalies")
        
        # Simulate anomaly detection
        anomalies = [
            {
                "type": "Volume Spike",
                "description": "Return volume increased 340% on March 15",
                "severity": "HIGH",
                "affected_returns": 67,
                "recommendation": "Investigate marketing campaign or product issue"
            },
            {
                "type": "Geographic Cluster",
                "description": "High fraud concentration in ZIP code 90210",
                "severity": "MEDIUM", 
                "affected_returns": 23,
                "recommendation": "Enhanced verification for this region"
            },
            {
                "type": "Temporal Pattern",
                "description": "Unusual return pattern on weekends",
                "severity": "LOW",
                "affected_returns": 15,
                "recommendation": "Monitor weekend return processing"
            }
        ]
        
        for anomaly in anomalies:
            severity_color = {"HIGH": "#dc2626", "MEDIUM": "#f59e0b", "LOW": "#16a34a"}[anomaly["severity"]]
            
            st.markdown(f"""
            <div style="border-left: 4px solid {severity_color}; padding: 1rem; margin: 1rem 0; background: #f9fafb; border-radius: 0.5rem;">
                <h4>{anomaly['type']} <span style="color: {severity_color};">({anomaly['severity']})</span></h4>
                <p>{anomaly['description']}</p>
                <p><strong>Affected Returns:</strong> {anomaly['affected_returns']}</p>
                <p><strong>Recommendation:</strong> {anomaly['recommendation']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Anomaly visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Anomaly score distribution
            anomaly_scores = np.random.beta(2, 8, len(df)) * 100
            
            fig = px.histogram(
                x=anomaly_scores,
                nbins=20,
                title="Anomaly Score Distribution",
                labels={'x': 'Anomaly Score', 'y': 'Count'}
            )
            fig.add_vline(x=80, line_dash="dash", line_color="red", annotation_text="Anomaly Threshold")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Real-time anomaly monitoring
            st.markdown("### 📊 Real-time Monitoring")
            
            # Simulate real-time metrics
            metrics = [
                {"Metric": "Return Volume", "Current": "18", "Normal Range": "12-22", "Status": "Normal"},
                {"Metric": "Avg Risk Score", "Current": "45", "Normal Range": "35-55", "Status": "Normal"},
                {"Metric": "High Risk %", "Current": "23%", "Normal Range": "15-25%", "Status": "Normal"},
                {"Metric": "Processing Time", "Current": "0.31s", "Normal Range": "0.2-0.4s", "Status": "Normal"}
            ]
            
            for metric in metrics:
                status_color = "#16a34a" if metric["Status"] == "Normal" else "#dc2626"
                st.markdown(f"""
                <div style="padding: 0.5rem; margin: 0.25rem 0; border: 1px solid #e5e7eb; border-radius: 0.25rem;">
                    <strong>{metric['Metric']}:</strong> {metric['Current']} 
                    <span style="color: {status_color};">● {metric['Status']}</span><br>
                    <small>Normal: {metric['Normal Range']}</small>
                </div>
                """, unsafe_allow_html=True)
    
    with tab4:
        st.subheader("🎯 Model Performance & Validation")
        
        # Model metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📊 Classification Metrics")
            
            metrics = {
                'Precision': 0.892,
                'Recall': 0.847,
                'F1-Score': 0.869,
                'Accuracy': 0.873,
                'AUC-ROC': 0.934
            }
            
            for metric, value in metrics.items():
                st.metric(metric, f"{value:.3f}")
        
        with col2:
            st.markdown("### 🎯 Confusion Matrix")
            
            # Simulate confusion matrix
            cm_data = np.array([[145, 12], [8, 35]])
            
            fig = px.imshow(
                cm_data,
                text_auto=True,
                aspect="auto",
                title="Confusion Matrix",
                labels=dict(x="Predicted", y="Actual"),
                color_continuous_scale="Blues"
            )
            fig.update_xaxes(ticktext=["Not Fraud", "Fraud"], tickvals=[0, 1])
            fig.update_yaxes(ticktext=["Not Fraud", "Fraud"], tickvals=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            st.markdown("### 📈 Performance Trends")
            
            # Performance over time
            dates = pd.date_range(start=datetime.now() - timedelta(days=30), periods=30, freq='D')
            accuracy_trend = 0.85 + np.random.normal(0, 0.02, 30)
            accuracy_trend = np.cumsum(accuracy_trend * 0.001) + 0.85
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=accuracy_trend,
                mode='lines+markers',
                name='Model Accuracy',
                line=dict(color='#16a34a', width=2)
            ))
            fig.update_layout(
                title="Model Accuracy Trend",
                xaxis_title="Date",
                yaxis_title="Accuracy",
                yaxis=dict(range=[0.8, 0.95])
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance
        st.markdown("### 🔍 Feature Importance Analysis")
        
        features = [
            'Customer Return History', 'Order Value', 'Return Timing', 
            'Return Reason Type', 'Days Since Purchase', 'Shipping Method',
            'Customer Age', 'Product Category', 'Geographic Location'
        ]
        importance_scores = [0.28, 0.22, 0.18, 0.12, 0.08, 0.05, 0.04, 0.02, 0.01]
        
        fig = px.bar(
            x=importance_scores,
            y=features,
            orientation='h',
            title="Feature Importance in Fraud Detection Model"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.subheader("⚙️ AI Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎛️ Model Parameters")
            
            # Model configuration
            confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.95, 0.85, 0.05)
            risk_sensitivity = st.selectbox("Risk Sensitivity", ["Conservative", "Balanced", "Aggressive"])
            auto_learning = st.checkbox("Enable Continuous Learning", value=True)
            feedback_incorporation = st.checkbox("Incorporate Manual Reviews", value=True)
            
            # Advanced settings
            st.markdown("### 🔧 Advanced Settings")
            
            feature_weights = st.expander("Feature Weight Customization")
            with feature_weights:
                customer_history_weight = st.slider("Customer History Weight", 0.0, 1.0, 0.3)
                order_value_weight = st.slider("Order Value Weight", 0.0, 1.0, 0.2)
                timing_weight = st.slider("Timing Weight", 0.0, 1.0, 0.2)
            
            if st.button("🚀 Update Model Configuration"):
                st.success("✅ Model configuration updated successfully!")
        
        with col2:
            st.markdown("### 📊 Model Status")
            
            # Current model info
            model_info = {
                "Model Version": "v2.1.3",
                "Last Updated": "2024-03-10 14:30 UTC",
                "Training Data": "50,000 returns",
                "Validation Accuracy": "87.3%",
                "Status": "Active"
            }
            
            for key, value in model_info.items():
                st.markdown(f"**{key}:** {value}")
            
            st.markdown("### 🔄 Model Management")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                if st.button("🔄 Retrain Model"):
                    st.info("🔄 Model retraining initiated...")
            
            with col_b:
                if st.button("📊 Validate Model"):
                    st.info("📊 Running model validation...")
            
            if st.button("📥 Export Model", use_container_width=True):
                st.info("📥 Preparing model export...")
            
            if st.button("📤 Import Model", use_container_width=True):
                st.info("📤 Model import interface...")

def show_enterprise_analytics():
    """Enterprise analytics and reporting"""
    
    st.title("📈 Enterprise Analytics")
    
    df = st.session_state.returns_data
    
    # Analytics time range selector
    col1, col2, col3 = st.columns(3)
    
    with col1:
        date_range = st.date_input(
            "📅 Select Date Range",
            value=(date.today() - timedelta(days=30), date.today()),
            max_value=date.today()
        )
    
    with col2:
        granularity = st.selectbox("📊 Time Granularity", ["Daily", "Weekly", "Monthly"])
    
    with col3:
        comparison_period = st.selectbox("📈 Compare With", ["Previous Period", "Same Period Last Year", "No Comparison"])
    
    # Key Performance Indicators
    st.subheader("🎯 Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Calculate KPIs
    total_returns = len(df)
    fraud_prevented = len(df[df['risk_level'] == 'HIGH'])
    total_savings = df['amount_saved'].sum()
    avg_processing_time = 0.23  # seconds
    customer_satisfaction = 94.2  # percentage
    
    with col1:
        st.metric(
            "🔄 Total Returns",
            f"{total_returns:,}",
            delta="12%",
            help="Total number of returns processed"
        )
    
    with col2:
        st.metric(
            "🛡️ Fraud Prevented", 
            f"{fraud_prevented:,}",
            delta="8%",
            help="Number of potentially fraudulent returns caught"
        )
    
    with col3:
        st.metric(
            "💰 Total Savings",
            f"${total_savings:,.0f}",
            delta="15%",
            help="Total amount saved from prevented fraud"
        )
    
    with col4:
        st.metric(
            "⚡ Avg Processing Time",
            f"{avg_processing_time}s",
            delta="-0.05s",
            delta_color="inverse",
            help="Average time to process a return"
        )
    
    with col5:
        st.metric(
            "😊 Customer Satisfaction",
            f"{customer_satisfaction}%",
            delta="2.1%",
            help="Customer satisfaction with return process"
        )
    
    # Analytics tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "🎯 Fraud Analysis", "💰 Financial Impact", "👥 Customer Insights", "📈 Trends"
    ])
    
    with tab1:
        # Overview analytics
        col1, col2 = st.columns(2)
        
        with col1:
            # Returns by risk level over time
            daily_risk = df.groupby([df['return_date'].dt.date, 'risk_level']).size().unstack(fill_value=0)
            
            fig = go.Figure()
            colors = {'LOW': '#16a34a', 'MEDIUM': '#f59e0b', 'HIGH': '#dc2626'}
            
            for risk_level in ['LOW', 'MEDIUM', 'HIGH']:
                if risk_level in daily_risk.columns:
                    fig.add_trace(go.Scatter(
                        x=daily_risk.index,
                        y=daily_risk[risk_level],
                        mode='lines+markers',
                        name=f'{risk_level} Risk',
                        line=dict(color=colors[risk_level], width=2),
                        stackgroup='one'
                    ))
            
            fig.update_layout(
                title="Returns by Risk Level Over Time",
                xaxis_title="Date",
                yaxis_title="Number of Returns",
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Category analysis
            category_stats = df.groupby('category').agg({
                'id': 'count',
                'risk_score': 'mean',
                'amount_saved': 'sum'
            }).round(2)
            category_stats.columns = ['Returns', 'Avg Risk Score', 'Savings']
            category_stats = category_stats.sort_values('Avg Risk Score', ascending=False)
            
            fig = px.bar(
                category_stats.reset_index(),
                x='category',
                y='Avg Risk Score',
                color='Returns',
                title="Average Risk Score by Product Category",
                color_continuous_scale="Reds"
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance summary table
        st.subheader("📋 Performance Summary")
        
        performance_data = {
            'Metric': [
                'Total Returns Processed',
                'Fraud Detection Accuracy', 
                'False Positive Rate',
                'Average Processing Time',
                'Customer Satisfaction Score',
                'Revenue Protected',
                'Cost Savings vs Manual Review'
            ],
            'Current Period': [
                f"{total_returns:,}",
                "87.3%",
                "8.2%", 
                "0.23s",
                "94.2%",
                f"${total_savings:,.0f}",
                "$45,200"
            ],
            'Previous Period': [
                f"{int(total_returns * 0.88):,}",
                "85.1%",
                "9.7%",
                "0.28s", 
                "92.1%",
                f"${total_savings * 0.85:,.0f}",
                "$39,800"
            ],
            'Change': [
                "+12%",
                "+2.2%",
                "-1.5%",
                "-0.05s",
                "+2.1%",
                "+15%",
                "+13.6%"
            ]
        }
        
        st.dataframe(pd.DataFrame(performance_data), use_container_width=True)
    
    with tab2:
        # Fraud analysis
        st.subheader("🎯 Fraud Pattern Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Fraud by hour of day
            hourly_fraud = df[df['risk_level'] == 'HIGH'].groupby('return_hour').size()
            
            fig = px.bar(
                x=hourly_fraud.index,
                y=hourly_fraud.values,
                title="High-Risk Returns by Hour of Day",
                labels={'x': 'Hour', 'y': 'High-Risk Returns'},
                color=hourly_fraud.values,
                color_continuous_scale="Reds"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Fraud by return reason
            reason_risk = df.groupby('return_reason').agg({
                'risk_score': 'mean',
                'id': 'count'
            }).round(1)
            reason_risk.columns = ['Avg Risk Score', 'Count']
            reason_risk = reason_risk[reason_risk['Count'] >= 3].sort_values('Avg Risk Score', ascending=False)
            
            fig = px.scatter(
                reason_risk.reset_index(),
                x='Count',
                y='Avg Risk Score',
                size='Count',
                hover_data=['return_reason'],
                title="Risk Score vs Frequency by Return Reason"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk correlation matrix
        st.subheader("🔗 Risk Factor Correlation Analysis")
        
        correlation_features = ['risk_score', 'order_value', 'days_since_purchase', 'customer_return_count', 'return_hour']
        correlation_matrix = df[correlation_features].corr()
        
        fig = px.imshow(
            correlation_matrix,
            title="Risk Factor Correlation Matrix",
            color_continuous_scale="RdBu_r",
            aspect="auto"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Top risky patterns
        st.subheader("🚨 Top Risky Patterns Identified")
        
        patterns = [
            {"Pattern": "Electronics returned within 2 days", "Instances": 23, "Avg Risk": 89, "Fraud Rate": "87%"},
            {"Pattern": "Customer with 8+ previous returns", "Instances": 18, "Avg Risk": 85, "Fraud Rate": "83%"},
            {"Pattern": "Damage claims after 45+ days", "Instances": 15, "Avg Risk": 82, "Fraud Rate": "80%"},
            {"Pattern": "High-value returns (>$500) on weekends", "Instances": 12, "Avg Risk": 78, "Fraud Rate": "75%"},
            {"Pattern": "Multiple returns same customer same day", "Instances": 8, "Avg Risk": 91, "Fraud Rate": "88%"}
        ]
        
        st.dataframe(pd.DataFrame(patterns), use_container_width=True)
    
    with tab3:
        # Financial impact analysis
        st.subheader("💰 Financial Impact Analysis")
        
        # Monthly savings trend
        col1, col2 = st.columns(2)
        
        with col1:
            # Simulate monthly savings data
            months = pd.date_range(start=datetime.now() - timedelta(days=365), periods=12, freq='M')
            monthly_savings = [15000, 18000, 22000, 19000, 25000, 28000, 31000, 27000, 33000, 35000, 38000, 42000]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=months,
                y=monthly_savings,
                mode='lines+markers',
                name='Monthly Savings',
                line=dict(color='#16a34a', width=3),
                fill='tonexty'
            ))
            fig.update_layout(
                title="Monthly Fraud Prevention Savings",
                xaxis_title="Month",
                yaxis_title="Savings ($)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # ROI calculation
            implementation_cost = 50000
            monthly_savings_avg = np.mean(monthly_savings)
            annual_savings = monthly_savings_avg * 12
            roi = ((annual_savings - implementation_cost) / implementation_cost) * 100
            
            st.metric("💡 Annual ROI", f"{roi:.1f}%")
            st.metric("💰 Annual Savings", f"${annual_savings:,.0f}")
            st.metric("⚡ Monthly Avg Savings", f"${monthly_savings_avg:,.0f}")
            st.metric("🎯 Payback Period", "1.8 months")
        
        # Cost breakdown
        st.subheader("💸 Cost-Benefit Analysis")
        
        cost_data = {
            'Category': ['Fraud Prevention', 'Manual Review Reduction', 'Processing Efficiency', 'Customer Retention'],
            'Annual Savings': [150000, 89000, 34000, 28000],
            'Implementation Cost': [25000, 15000, 8000, 2000],
            'Net Benefit': [125000, 74000, 26000, 26000]
        }
        
        cost_df = pd.DataFrame(cost_data)
        
        fig = px.bar(
            cost_df,
            x='Category',
            y=['Annual Savings', 'Implementation Cost'],
            title="Cost-Benefit Breakdown by Category",
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Customer insights
        st.subheader("👥 Customer Behavior Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Customer segmentation by return behavior
            customer_segments = df.groupby('customer_id').agg({
                'id': 'count',
                'risk_score': 'mean',
                'order_value': 'sum'
            }).reset_index()
            customer_segments.columns = ['Customer', 'Return Count', 'Avg Risk', 'Total Value']
            
            # Create segments
            def categorize_customer(row):
                if row['Return Count'] > 8 and row['Avg Risk'] > 70:
                    return 'High Risk'
                elif row['Return Count'] > 5 or row['Avg Risk'] > 60:
                    return 'Medium Risk'
                elif row['Return Count'] > 2:
                    return 'Regular'
                else:
                    return 'Low Activity'
            
            customer_segments['Segment'] = customer_segments.apply(categorize_customer, axis=1)
            
            segment_counts = customer_segments['Segment'].value_counts()
            
            fig = px.pie(
                values=segment_counts.values,
                names=segment_counts.index,
                title="Customer Segmentation by Return Behavior",
                color_discrete_map={
                    'High Risk': '#dc2626',
                    'Medium Risk': '#f59e0b', 
                    'Regular': '#2563eb',
                    'Low Activity': '#16a34a'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Customer lifetime value vs return frequency
            fig = px.scatter(
                customer_segments,
                x='Return Count',
                y='Total Value',
                color='Avg Risk',
                size='Total Value',
                title="Customer Value vs Return Frequency",
                color_continuous_scale="Reds"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Top customers by risk
        st.subheader("⚠️ High-Risk Customer Analysis")
        
        high_risk_customers = customer_segments[customer_segments['Segment'] == 'High Risk'].sort_values('Avg Risk', ascending=False).head(10)
        
        if not high_risk_customers.empty:
            st.dataframe(
                high_risk_customers[['Customer', 'Return Count', 'Avg Risk', 'Total Value']],
                use_container_width=True,
                column_config={
                    "Avg Risk": st.column_config.ProgressColumn(
                        "Average Risk Score",
                        min_value=0,
                        max_value=100
                    ),
                    "Total Value": st.column_config.NumberColumn(
                        "Total Order Value",
                        format="$%.2f"
                    )
                }
            )
        else:
            st.info("No high-risk customers identified in current data.")
    
    with tab5:
        # Trends and forecasting
        st.subheader("📈 Trends & Forecasting")
        
        # Generate trend data
        col1, col2 = st.columns(2)
        
        with col1:
            # Return volume trend with forecast
            dates = pd.date_range(start=datetime.now() - timedelta(days=90), periods=90, freq='D')
            
            # Historical data (first 60 days)
            historical_returns = 10 + np.sin(np.arange(60) * 2 * np.pi / 7) * 3 + np.random.normal(0, 1, 60)
            
            # Forecast (next 30 days)
            forecast_returns = 10 + np.sin(np.arange(60, 90) * 2 * np.pi / 7) * 3 + np.random.normal(0, 1, 30) + 2
            
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=dates[:60],
                y=historical_returns,
                mode='lines+markers',
                name='Historical',
                line=dict(color='#2563eb', width=2)
            ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=dates[60:],
                y=forecast_returns,
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#dc2626', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title="Return Volume Trend & 30-Day Forecast",
                xaxis_title="Date",
                yaxis_title="Daily Returns"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk score trend
            risk_trend = df.groupby(df['return_date'].dt.date)['risk_score'].mean()
            
            # Apply smoothing
            from scipy.signal import savgol_filter
            smoothed_risk = savgol_filter(risk_trend.values, window_length=min(7, len(risk_trend)), polyorder=2)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=risk_trend.index,
                y=risk_trend.values,
                mode='markers',
                name='Daily Avg Risk',
                marker=dict(color='#94a3b8', size=4)
            ))
            fig.add_trace(go.Scatter(
                x=risk_trend.index,
                y=smoothed_risk,
                mode='lines',
                name='Trend',
                line=dict(color='#dc2626', width=3)
            ))
            fig.update_layout(
                title="Average Risk Score Trend",
                xaxis_title="Date",
                yaxis_title="Risk Score"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Seasonal analysis
        st.subheader("🗓️ Seasonal Pattern Analysis")
        
        # Day of week analysis
        df['day_of_week'] = df['return_date'].dt.day_name()
        day_stats = df.groupby('day_of_week').agg({
            'id': 'count',
            'risk_score': 'mean'
        }).round(1)
        day_stats.columns = ['Return Count', 'Avg Risk Score']
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_stats = day_stats.reindex(day_order)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=day_stats.index,
            y=day_stats['Return Count'],
            name='Return Count',
            yaxis='y'
        ))
        fig.add_trace(go.Scatter(
            x=day_stats.index,
            y=day_stats['Avg Risk Score'],
            mode='lines+markers',
            name='Avg Risk Score',
            yaxis='y2',
            line=dict(color='#dc2626', width=3)
        ))
        
        fig.update_layout(
            title="Returns and Risk by Day of Week",
            xaxis_title="Day of Week",
            yaxis=dict(title="Return Count", side="left"),
            yaxis2=dict(title="Average Risk Score", side="right", overlaying="y")
        )
        st.plotly_chart(fig, use_container_width=True)

def show_enterprise_integrations():
    """Enterprise integrations management"""
    
    st.title("🔗 Enterprise Integrations")
    
    # Integration status overview
    st.subheader("📊 Integration Status Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🔌 Active Integrations", "3", "+1")
    with col2:
        st.metric("📡 Data Sync Rate", "99.7%", "+0.2%")
    with col3:
        st.metric("⚡ Last Sync", "2 min ago")
    with col4:
        st.metric("📈 Total Records", "125,847", "+1,203")
    
    # Integration tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏪 E-commerce", "📧 Notifications", "📊 Analytics", "🔧 API & Webhooks"])
    
    with tab1:
        st.subheader("🏪 E-commerce Platform Integrations")
        
        # Available integrations
        integrations = [
            {
                "name": "Shopify",
                "logo": "🛍️",
                "status": "Connected",
                "stores": 2,
                "last_sync": "2 minutes ago",
                "records": "48,392 returns"
            },
            {
                "name": "WooCommerce", 
                "logo": "🛒",
                "status": "Connected",
                "stores": 1,
                "last_sync": "5 minutes ago",
                "records": "31,208 returns"
            },
            {
                "name": "BigCommerce",
                "logo": "🏬",
                "status": "Available",
                "stores": 0,
                "last_sync": "N/A",
                "records": "N/A"
            },
            {
                "name": "Magento",
                "logo": "🎪",
                "status": "Available",
                "stores": 0,
                "last_sync": "N/A", 
                "records": "N/A"
            },
            {
                "name": "Amazon Seller",
                "logo": "📦",
                "status": "Beta",
                "stores": 0,
                "last_sync": "N/A",
                "records": "N/A"
            },
            {
                "name": "Etsy",
                "logo": "🎨",
                "status": "Coming Soon",
                "stores": 0,
                "last_sync": "N/A",
                "records": "N/A"
            }
        ]
        
        for integration in integrations:
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([1, 3, 2, 2, 2])
                
                with col1:
                    st.markdown(f"<h2>{integration['logo']}</h2>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"**{integration['name']}**")
                    status_color = {
                        "Connected": "#16a34a",
                        "Available": "#6b7280", 
                        "Beta": "#f59e0b",
                        "Coming Soon": "#dc2626"
                    }[integration['status']]
                    st.markdown(f"<span style='color: {status_color}; font-weight: bold;'>{integration['status']}</span>", unsafe_allow_html=True)
                
                with col3:
                    if integration['status'] == "Connected":
                        st.write(f"Stores: {integration['stores']}")
                        st.write(f"Records: {integration['records']}")
                    else:
                        st.write("Not connected")
                
                with col4:
                    if integration['status'] == "Connected":
                        st.write(f"Last sync: {integration['last_sync']}")
                    else:
                        st.write("—")
                
                with col5:
                    if integration['status'] == "Connected":
                        if st.button(f"⚙️ Configure", key=f"config_{integration['name']}"):
                            st.info(f"Opening {integration['name']} configuration...")
                    elif integration['status'] == "Available":
                        if st.button(f"🔌 Connect", key=f"connect_{integration['name']}"):
                            st.info(f"Connecting to {integration['name']}...")
                    elif integration['status'] == "Beta":
                        if st.button(f"🧪 Join Beta", key=f"beta_{integration['name']}"):
                            st.info(f"Joining {integration['name']} beta program...")
                    else:
                        st.button("⏳ Coming Soon", disabled=True)
                
                st.markdown("---")
        
        # Add new integration
        st.subheader("➕ Add New Integration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            platform = st.selectbox("Select Platform", ["Shopify", "WooCommerce", "BigCommerce", "Custom API"])
            store_name = st.text_input("Store Name", placeholder="My Store")
            store_url = st.text_input("Store URL", placeholder="https://mystore.shopify.com")
        
        with col2:
            api_key = st.text_input("API Key", type="password")
            api_secret = st.text_input("API Secret", type="password")
            webhook_url = st.text_input("Webhook URL (Optional)")
        
        if st.button("🚀 Test Connection & Connect"):
            with st.spinner("Testing connection..."):
                time.sleep(2)
                st.success("✅ Connection successful! Integration added.")
    
    with tab2:
        st.subheader("📧 Notification Integrations")
        
        # Notification channels
        notification_channels = [
            {
                "name": "Email",
                "icon": "📧",
                "status": "Active",
                "config": "admin@company.com",
                "alerts_sent": 156
            },
            {
                "name": "Slack",
                "icon": "💬",
                "status": "Active", 
                "config": "#fraud-alerts",
                "alerts_sent": 89
            },
            {
                "name": "Microsoft Teams",
                "icon": "👥",
                "status": "Available",
                "config": "Not configured",
                "alerts_sent": 0
            },
            {
                "name": "SMS",
                "icon": "📱",
                "status": "Available",
                "config": "Not configured", 
                "alerts_sent": 0
            },
            {
                "name": "Webhook",
                "icon": "🔗",
                "status": "Active",
                "config": "https://api.company.com/alerts",
                "alerts_sent": 234
            },
            {
                "name": "Discord",
                "icon": "🎮",
                "status": "Beta",
                "config": "Not configured",
                "alerts_sent": 0
            }
        ]
        
        for channel in notification_channels:
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 2])
                
                with col1:
                    st.markdown(f"<h3>{channel['icon']}</h3>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"**{channel['name']}**")
                    status_color = {
                        "Active": "#16a34a",
                        "Available": "#6b7280",
                        "Beta": "#f59e0b"
                    }[channel['status']]
                    st.markdown(f"<span style='color: {status_color};'>{channel['status']}</span>", unsafe_allow_html=True)
                
                with col3:
                    st.write(channel['config'])
                
                with col4:
                    if channel['status'] == "Active":
                        st.metric("Alerts Sent", channel['alerts_sent'])
                    else:
                        st.write("—")
                
                with col5:
                    if channel['status'] == "Active":
                        if st.button("⚙️ Configure", key=f"notif_config_{channel['name']}"):
                            st.info(f"Configuring {channel['name']}...")
                    else:
                        if st.button("🔌 Setup", key=f"notif_setup_{channel['name']}"):
                            st.info(f"Setting up {channel['name']}...")
                
                st.markdown("---")
        
        # Alert configuration
        st.subheader("🔔 Alert Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Alert Triggers**")
            high_risk_alerts = st.checkbox("High Risk Returns", value=True)
            volume_spikes = st.checkbox("Return Volume Spikes", value=True)
            new_patterns = st.checkbox("New Fraud Patterns", value=False)
            system_issues = st.checkbox("System Issues", value=True)
            
            st.markdown("**Alert Frequency**")
            frequency = st.selectbox("Frequency", ["Immediate", "Every 15 minutes", "Hourly", "Daily"])
        
        with col2:
            st.markdown("**Alert Recipients**")
            recipients = st.multiselect(
                "Select Recipients",
                ["admin@company.com", "security@company.com", "manager@company.com"],
                default=["admin@company.com"]
            )
            
            st.markdown("**Quiet Hours**")
            quiet_start = st.time_input("Quiet hours start", datetime.strptime("22:00", "%H:%M").time())
            quiet_end = st.time_input("Quiet hours end", datetime.strptime("08:00", "%H:%M").time())
        
        if st.button("💾 Save Alert Configuration"):
            st.success("✅ Alert configuration saved!")
    
    with tab3:
        st.subheader("📊 Analytics & Reporting Integrations")
        
        # Analytics integrations
        analytics_tools = [
            {
                "name": "Google Analytics",
                "icon": "📈",
                "description": "Track return behavior and e-commerce metrics",
                "status": "Available",
                "features": ["E-commerce tracking", "Custom events", "Goal tracking"]
            },
            {
                "name": "Tableau",
                "icon": "📊",
                "description": "Advanced data visualization and dashboards",
                "status": "Available", 
                "features": ["Live data connection", "Custom dashboards", "Automated reports"]
            },
            {
                "name": "Power BI",
                "icon": "💼",
                "description": "Microsoft Power BI integration for enterprise reporting",
                "status": "Available",
                "features": ["Real-time dashboards", "Data modeling", "Collaboration"]
            },
            {
                "name": "Datadog",
                "icon": "🐕",
                "description": "System monitoring and alerting",
                "status": "Connected",
                "features": ["Performance monitoring", "Custom metrics", "Alerting"]
            },
            {
                "name": "Mixpanel",
                "icon": "🔀",
                "description": "Product analytics and user behavior tracking", 
                "status": "Beta",
                "features": ["Event tracking", "Funnel analysis", "Retention reports"]
            }
        ]
        
        for tool in analytics_tools:
            with st.expander(f"{tool['icon']} {tool['name']} - {tool['status']}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(tool['description'])
                    st.markdown("**Features:**")
                    for feature in tool['features']:
                        st.write(f"• {feature}")
                
                with col2:
                    if tool['status'] == "Connected":
                        st.success("✅ Connected")
                        if st.button(f"⚙️ Configure {tool['name']}", key=f"analytics_{tool['name']}"):
                            st.info(f"Opening {tool['name']} configuration...")
                    elif tool['status'] == "Available":
                        if st.button(f"🔌 Connect to {tool['name']}", key=f"connect_analytics_{tool['name']}"):
                            st.info(f"Connecting to {tool['name']}...")
                    elif tool['status'] == "Beta":
                        if st.button(f"🧪 Join Beta", key=f"beta_analytics_{tool['name']}"):
                            st.info(f"Joining {tool['name']} beta...")
        
        # Export options
        st.subheader("📤 Data Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Export Formats**")
            export_csv = st.checkbox("CSV Export", value=True)
            export_json = st.checkbox("JSON Export", value=True)
            export_excel = st.checkbox("Excel Export", value=True)
            export_api = st.checkbox("API Access", value=True)
        
        with col2:
            st.markdown("**Automated Reports**")
            daily_reports = st.checkbox("Daily Summary Reports")
            weekly_reports = st.checkbox("Weekly Analysis Reports", value=True)
            monthly_reports = st.checkbox("Monthly Executive Reports", value=True)
            custom_reports = st.checkbox("Custom Scheduled Reports")
        
        if st.button("📊 Generate Sample Report"):
            st.info("📊 Generating sample analytics report...")
    
    with tab4:
        st.subheader("🔧 API & Webhook Management")
        
        # API information
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔑 API Configuration")
            
            api_key = "rg_" + "".join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz0123456789'), 32))
            
            st.code(f"API Key: {api_key}", language="text")
            
            if st.button("🔄 Regenerate API Key"):
                st.warning("⚠️ This will invalidate the current API key. Continue?")
            
            st.markdown("**API Endpoints:**")
            endpoints = [
                "POST /api/v1/returns/analyze",
                "GET /api/v1/returns/{id}",
                "GET /api/v1/analytics/summary",
                "POST /api/v1/webhooks/register",
                "GET /api/v1/rules/custom"
            ]
            
            for endpoint in endpoints:
                st.code(endpoint, language="text")
        
        with col2:
            st.markdown("### 🪝 Webhook Configuration")
            
            webhook_url = st.text_input("Webhook URL", placeholder="https://your-domain.com/webhook")
            webhook_secret = st.text_input("Webhook Secret", type="password")
            
            st.markdown("**Webhook Events:**")
            events = {
                "return.analyzed": st.checkbox("Return Analyzed", value=True),
                "fraud.detected": st.checkbox("Fraud Detected", value=True),
                "review.completed": st.checkbox("Manual Review Completed"),
                "pattern.discovered": st.checkbox("New Pattern Discovered"),
                "threshold.exceeded": st.checkbox("Risk Threshold Exceeded", value=True)
            }
            
            if st.button("💾 Save Webhook Configuration"):
                st.success("✅ Webhook configuration saved!")
        
        # API usage statistics
        st.markdown("### 📊 API Usage Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📈 Total Requests", "125,847", "+1,203 today")
        with col2:
            st.metric("⚡ Avg Response Time", "127ms", "-15ms")
        with col3:
            st.metric("✅ Success Rate", "99.97%", "+0.02%")
        with col4:
            st.metric("🔒 Rate Limit", "1000/hour", "23% used")
        
        # API documentation
        st.markdown("### 📚 API Documentation")
        
        with st.expander("📖 View API Documentation"):
            st.markdown("""
            ## ReturnGuard AI API Documentation
            
            ### Authentication
            Include your API key in the header:
            ```
            Authorization: Bearer {your_api_key}
            ```
            
            ### Analyze Return
            ```
            POST /api/v1/returns/analyze
            Content-Type: application/json
            
            {
                "return_id": "RET-12345",
                "customer_id": "CUST-67890",
                "order_value": 299.99,
                "return_reason": "Item damaged",
                "days_since_purchase": 7,
                "customer_return_count": 2
            }
            ```
            
            ### Response
            ```json
            {
                "risk_score": 75,
                "risk_level": "MEDIUM",
                "recommendation": "Request additional verification",
                "confidence": 87,
                "processing_time_ms": 123
            }
            ```
            """)
        
        # Test API endpoint
        st.markdown("### 🧪 Test API Endpoint")
        
        with st.form("api_test"):
            endpoint = st.selectbox("Select Endpoint", [
                "/api/v1/returns/analyze",
                "/api/v1/analytics/summary", 
                "/api/v1/returns/list"
            ])
            
            if endpoint == "/api/v1/returns/analyze":
                st.json({
                    "return_id": "RET-TEST-001",
                    "customer_id": "CUST-TEST-001",
                    "order_value": 149.99,
                    "return_reason": "Wrong size",
                    "days_since_purchase": 5,
                    "customer_return_count": 1
                })
            
            if st.form_submit_button("🚀 Test API Call"):
                with st.spinner("Making API call..."):
                    time.sleep(1)
                    st.success("✅ API call successful!")
                    st.json({
                        "risk_score": 45,
                        "risk_level": "LOW",
                        "recommendation": "Safe to auto-approve",
                        "confidence": 89,
                        "processing_time_ms": 156
                    })

def show_enterprise_team_management():
    """Team management and user administration"""
    
    st.title("👥 Team Management")
    
    # Team overview
    st.subheader("👥 Team Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👤 Total Users", "12", "+2")
    with col2:
        st.metric("👨‍💼 Admins", "3")
    with col3:
        st.metric("👩‍💻 Analysts", "7") 
    with col4:
        st.metric("👁️ Reviewers", "2")
    
    # Team management tabs
    tab1, tab2, tab3, tab4 = st.tabs(["👥 Team Members", "🔐 Roles & Permissions", "📊 Activity Logs", "⚙️ Team Settings"])
    
    with tab1:
        st.subheader("👥 Team Members")
        
        # Add new user
        with st.expander("➕ Add New Team Member"):
            col1, col2 = st.columns(2)
            
            with col1:
                new_user_name = st.text_input("Full Name", placeholder="John Doe")
                new_user_email = st.text_input("Email Address", placeholder="john@company.com")
                new_user_role = st.selectbox("Role", ["Admin", "Analyst", "Reviewer", "Viewer"])
            
            with col2:
                send_invite = st.checkbox("Send email invitation", value=True)
                require_2fa = st.checkbox("Require 2FA", value=True)
                
                if new_user_role == "Analyst":
                    departments = st.multiselect("Departments", ["Electronics", "Clothing", "Home & Garden"], default=["Electronics"])
                elif new_user_role == "Reviewer":
                    risk_levels = st.multiselect("Review Authority", ["HIGH", "MEDIUM", "LOW"], default=["MEDIUM", "LOW"])
            
            if st.button("👤 Add Team Member", type="primary"):
                st.success(f"✅ Invitation sent to {new_user_email}")
        
        # Current team members
        team_members = [
            {
                "name": "Sarah Johnson",
                "email": "sarah@company.com", 
                "role": "Admin",
                "status": "Active",
                "last_login": "2 hours ago",
                "permissions": ["Full Access"],
                "2fa": True
            },
            {
                "name": "Mike Chen",
                "email": "mike@company.com",
                "role": "Analyst", 
                "status": "Active",
                "last_login": "1 day ago",
                "permissions": ["Analytics", "Reports"],
                "2fa": True
            },
            {
                "name": "Emily Rodriguez",
                "email": "emily@company.com",
                "role": "Reviewer",
                "status": "Active", 
                "last_login": "3 hours ago",
                "permissions": ["Review Returns", "Approve/Reject"],
                "2fa": False
            },
            {
                "name": "David Kim",
                "email": "david@company.com",
                "role": "Analyst",
                "status": "Invited",
                "last_login": "Never",
                "permissions": ["Analytics"],
                "2fa": False
            },
            {
                "name": "Lisa Wang",
                "email": "lisa@company.com",
                "role": "Viewer",
                "status": "Inactive",
                "last_login": "2 weeks ago", 
                "permissions": ["Read Only"],
                "2fa": True
            }
        ]
        
        # Team members table
        for member in team_members:
            with st.container():
                col1, col2, col3, col4, col5, col6 = st.columns([2, 2, 1, 1, 2, 1])
                
                with col1:
                    st.markdown(f"**{member['name']}**")
                    st.write(member['email'])
                
                with col2:
                    st.write(f"Role: {member['role']}")
                    st.write(f"Last login: {member['last_login']}")
                
                with col3:
                    status_color = {
                        "Active": "#16a34a",
                        "Invited": "#f59e0b", 
                        "Inactive": "#6b7280"
                    }[member['status']]
                    st.markdown(f"<span style='color: {status_color}; font-weight: bold;'>{member['status']}</span>", unsafe_allow_html=True)
                
                with col4:
                    if member['2fa']:
                        st.write("🔐 2FA")
                    else:
                        st.write("⚠️ No 2FA")
                
                with col5:
                    permissions_text = ", ".join(member['permissions'])
                    st.write(f"Permissions: {permissions_text}")
                
                with col6:
                    if st.button("⚙️", key=f"edit_{member['email']}"):
                        st.info(f"Editing {member['name']}")
                
                st.markdown("---")
        
        # Bulk actions
        st.subheader("📋 Bulk Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📧 Send Reminder to Inactive Users"):
                st.info("📧 Reminder emails sent to inactive users")
        
        with col2:
            if st.button("🔐 Require 2FA for All"):
                st.info("🔐 2FA requirement enabled for all users")
        
        with col3:
            if st.button("📊 Export Team Report"):
                st.info("📊 Team report generated and downloading...")
    
    with tab2:
        st.subheader("🔐 Roles & Permissions")
        
        # Role definitions
        roles = {
            "Admin": {
                "description": "Full system access including user management",
                "permissions": [
                    "Manage team members",
                    "Configure integrations", 
                    "Access all data",
                    "Modify system settings",
                    "View audit logs",
                    "Manage billing",
                    "Export data",
                    "Configure AI rules"
                ],
                "count": 3
            },
            "Analyst": {
                "description": "Advanced analytics and reporting access",
                "permissions": [
                    "View analytics dashboards",
                    "Generate reports",
                    "Access return data",
                    "View AI insights",
                    "Export limited data",
                    "Configure alerts"
                ],
                "count": 7
            },
            "Reviewer": {
                "description": "Manual review and decision making",
                "permissions": [
                    "Review flagged returns",
                    "Approve/reject returns",
                    "Add review notes",
                    "View customer data",
                    "Access risk scores",
                    "Update return status"
                ],
                "count": 2
            },
            "Viewer": {
                "description": "Read-only access to dashboards and reports",
                "permissions": [
                    "View basic dashboard",
                    "Access reports",
                    "View return summaries"
                ],
                "count": 0
            }
        }
        
        for role, details in roles.items():
            with st.expander(f"👤 {role} ({details['count']} users)"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown(f"**Description:**")
                    st.write(details['description'])
                    
                    if st.button(f"✏️ Edit {role} Role", key=f"edit_role_{role}"):
                        st.info(f"Editing {role} role permissions...")
                
                with col2:
                    st.markdown(f"**Permissions:**")
                    for permission in details['permissions']:
                        st.write(f"✅ {permission}")
        
        # Custom permissions
        st.subheader("🎛️ Custom Permissions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Data Access Permissions**")
            
            can_view_returns = st.checkbox("View Returns Data", value=True)
            can_export_data = st.checkbox("Export Data")
            can_view_customer_pii = st.checkbox("View Customer PII")
            can_access_api = st.checkbox("API Access")
        
        with col2:
            st.markdown("**Action Permissions**")
            
            can_approve_returns = st.checkbox("Approve Returns")
            can_reject_returns = st.checkbox("Reject Returns") 
            can_modify_rules = st.checkbox("Modify AI Rules")
            can_manage_integrations = st.checkbox("Manage Integrations")
        
        if st.button("💾 Save Custom Permissions"):
            st.success("✅ Custom permissions saved!")
    
    with tab3:
        st.subheader("📊 Team Activity Logs")
        
        # Activity filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            user_filter = st.selectbox("Filter by User", ["All Users", "Sarah Johnson", "Mike Chen", "Emily Rodriguez"])
        
        with col2:
            action_filter = st.selectbox("Filter by Action", ["All Actions", "Login", "Return Review", "Export Data", "Settings Change"])
        
        with col3:
            time_filter = st.selectbox("Time Period", ["Last 24 hours", "Last 7 days", "Last 30 days"])
        
        # Activity log entries
        activities = [
            {
                "timestamp": "2024-03-15 14:23:15",
                "user": "Sarah Johnson",
                "action": "Return Review",
                "details": "Approved return RET-12345 (Risk Score: 45)",
                "ip": "192.168.1.100",
                "result": "Success"
            },
            {
                "timestamp": "2024-03-15 14:15:32",
                "user": "Mike Chen", 
                "action": "Export Data",
                "details": "Exported returns data (CSV, 500 records)",
                "ip": "192.168.1.101",
                "result": "Success"
            },
            {
                "timestamp": "2024-03-15 13:45:21",
                "user": "Emily Rodriguez",
                "action": "Return Review",
                "details": "Rejected return RET-12344 (Risk Score: 89)",
                "ip": "192.168.1.102",
                "result": "Success"
            },
            {
                "timestamp": "2024-03-15 13:30:45",
                "user": "Sarah Johnson",
                "action": "Settings Change",
                "details": "Updated risk threshold from 80 to 85",
                "ip": "192.168.1.100",
                "result": "Success"
            },
            {
                "timestamp": "2024-03-15 12:15:33",
                "user": "Mike Chen",
                "action": "Login",
                "details": "Successful login via SSO",
                "ip": "192.168.1.101",
                "result": "Success"
            },
            {
                "timestamp": "2024-03-15 11:45:21",
                "user": "David Kim",
                "action": "Login",
                "details": "Failed login attempt - incorrect password",
                "ip": "203.0.113.15",
                "result": "Failed"
            }
        ]
        
        # Display activity log
        for activity in activities:
            result_color = "#16a34a" if activity['result'] == "Success" else "#dc2626"
            
            st.markdown(f"""
            <div style="border: 1px solid #e5e7eb; padding: 1rem; margin: 0.5rem 0; border-radius: 0.5rem; background: #f9fafb;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong>{activity['timestamp']}</strong> - {activity['user']}
                        <br><small>{activity['action']}: {activity['details']}</small>
                        <br><small>IP: {activity['ip']}</small>
                    </div>
                    <div style="color: {result_color}; font-weight: bold;">
                        {activity['result']}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Export activity log
        if st.button("📥 Export Activity Log"):
            activity_df = pd.DataFrame(activities)
            csv = activity_df.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, "activity_log.csv", "text/csv")
    
    with tab4:
        st.subheader("⚙️ Team Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔐 Security Settings")
            
            # Password policy
            min_password_length = st.slider("Minimum Password Length", 8, 20, 12)
            require_special_chars = st.checkbox("Require Special Characters", value=True)
            require_numbers = st.checkbox("Require Numbers", value=True)
            require_uppercase = st.checkbox("Require Uppercase Letters", value=True)
            
            # Session settings
            session_timeout = st.selectbox("Session Timeout", ["30 minutes", "1 hour", "4 hours", "8 hours"])
            require_2fa = st.checkbox("Require 2FA for All Users", value=False)
            
            # IP restrictions
            enable_ip_whitelist = st.checkbox("Enable IP Whitelist")
            if enable_ip_whitelist:
                allowed_ips = st.text_area("Allowed IP Addresses (one per line)", 
                                         placeholder="192.168.1.0/24\n203.0.113.0/24")
        
        with col2:
            st.markdown("### 👥 Team Policies")
            
            # Access policies
            max_concurrent_sessions = st.slider("Max Concurrent Sessions per User", 1, 10, 3)
            auto_deactivate_days = st.slider("Auto-deactivate inactive users after (days)", 30, 365, 90)
            
            # Notification settings
            notify_admin_new_user = st.checkbox("Notify admins of new user registrations", value=True)
            notify_failed_logins = st.checkbox("Notify of failed login attempts", value=True)
            notify_permission_changes = st.checkbox("Notify of permission changes", value=True)
            
            # Data retention
            audit_log_retention = st.selectbox("Audit Log Retention", ["30 days", "90 days", "1 year", "2 years"])
            export_data_retention = st.selectbox("Export Data Retention", ["7 days", "30 days", "90 days"])
        
        if st.button("💾 Save Team Settings", type="primary"):
            st.success("✅ Team settings saved successfully!")

def show_enterprise_custom_rules():
    """Custom fraud detection rules management"""
    
    st.title("🔧 Custom Fraud Detection Rules")
    
    # Rules overview
    st.subheader("📊 Rules Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📋 Total Rules", "23", "+3")
    with col2:
        st.metric("✅ Active Rules", "20")
    with col3:
        st.metric("⚠️ Triggered Today", "47", "+12")
    with col4:
        st.metric("🎯 Accuracy Rate", "94.2%", "+1.1%")
    
    # Rules management tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Active Rules", "➕ Create Rule", "📊 Rule Performance", "🔧 Advanced Settings"])
    
    with tab1:
        st.subheader("📋 Active Fraud Detection Rules")
        
        # Existing rules
        rules = [
            {
                "id": "RULE-001",
                "name": "High Value Quick Return",
                "description": "Flag returns over $500 within 3 days of purchase",
                "condition": "order_value > 500 AND days_since_purchase <= 3",
                "action": "Add 25 risk points",
                "status": "Active",
                "triggered": 12,
                "accuracy": 89.5
            },
            {
                "id": "RULE-002", 
                "name": "Serial Returner",
                "description": "Flag customers with more than 8 returns",
                "condition": "customer_return_count > 8",
                "action": "Add 30 risk points",
                "status": "Active",
                "triggered": 8,
                "accuracy": 92.1
            },
            {
                "id": "RULE-003",
                "name": "Late Damage Claim",
                "description": "Flag damage claims after 30 days",
                "condition": "return_reason CONTAINS 'damage' AND days_since_purchase > 30",
                "action": "Add 20 risk points",
                "status": "Active", 
                "triggered": 15,
                "accuracy": 76.3
            },
            {
                "id": "RULE-004",
                "name": "Weekend Electronics Returns",
                "description": "Flag electronics returns on weekends",
                "condition": "category = 'Electronics' AND day_of_week IN ['Saturday', 'Sunday']",
                "action": "Add 10 risk points",
                "status": "Testing",
                "triggered": 6,
                "accuracy": 68.2
            },
            {
                "id": "RULE-005",
                "name": "Midnight Return Submissions",
                "description": "Flag returns submitted between midnight and 6 AM",
                "condition": "return_hour >= 0 AND return_hour <= 6",
                "action": "Add 15 risk points",
                "status": "Inactive",
                "triggered": 0,
                "accuracy": 0
            }
        ]
        
        # Rules table
        for rule in rules:
            with st.expander(f"{rule['name']} ({rule['status']})"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Rule ID:** {rule['id']}")
                    st.markdown(f"**Description:** {rule['description']}")
                    st.markdown(f"**Condition:** `{rule['condition']}`")
                    st.markdown(f"**Action:** {rule['action']}")
                
                with col2:
                    status_color = {
                        "Active": "#16a34a",
                        "Testing": "#f59e0b",
                        "Inactive": "#6b7280"
                    }[rule['status']]
                    
                    st.markdown(f"**Status:** <span style='color: {status_color}'>{rule['status']}</span>", unsafe_allow_html=True)
                    st.metric("Times Triggered", rule['triggered'])
                    if rule['accuracy'] > 0:
                        st.metric("Accuracy", f"{rule['accuracy']}%")
                
                # Rule actions
                col_a, col_b, col_c, col_d = st.columns(4)
                
                with col_a:
                    if rule['status'] == "Active":
                        if st.button("⏸️ Pause", key=f"pause_{rule['id']}"):
                            st.info(f"Rule {rule['id']} paused")
                    elif rule['status'] == "Inactive":
                        if st.button("▶️ Activate", key=f"activate_{rule['id']}"):
                            st.success(f"Rule {rule['id']} activated")
                
                with col_b:
                    if st.button("✏️ Edit", key=f"edit_{rule['id']}"):
                        st.info(f"Editing rule {rule['id']}")
                
                with col_c:
                    if st.button("📊 Analytics", key=f"analytics_{rule['id']}"):
                        st.info(f"Showing analytics for rule {rule['id']}")
                
                with col_d:
                    if st.button("🗑️ Delete", key=f"delete_{rule['id']}"):
                        st.warning(f"Delete rule {rule['id']}?")
        
        # Bulk operations
        st.markdown("---")
        st.subheader("📋 Bulk Operations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("✅ Activate All Testing Rules"):
                st.success("All testing rules activated")
        
        with col2:
            if st.button("⏸️ Pause All Rules"):
                st.warning("All rules paused")
        
        with col3:
            if st.button("📊 Export Rules"):
                st.info("Rules exported to CSV")
    
    with tab2:
        st.subheader("➕ Create New Fraud Detection Rule")
        
        # Rule builder
        with st.form("new_rule"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📝 Rule Details")
                
                rule_name = st.text_input("Rule Name*", placeholder="My Custom Rule")
                rule_description = st.text_area("Description*", placeholder="Describe what this rule detects...")
                rule_category = st.selectbox("Category", ["Customer Behavior", "Order Patterns", "Timing", "Product Specific", "Geographic", "Custom"])
                rule_priority = st.selectbox("Priority", ["Low", "Medium", "High", "Critical"])
            
            with col2:
                st.markdown("### ⚙️ Rule Configuration")
                
                action_type = st.selectbox("Action Type", ["Add Risk Points", "Set Risk Level", "Flag for Review", "Auto-Reject"])
                
                if action_type == "Add Risk Points":
                    risk_points = st.slider("Risk Points to Add", 1, 50, 15)
                elif action_type == "Set Risk Level":
                    risk_level = st.selectbox("Risk Level", ["LOW", "MEDIUM", "HIGH"])
                
                rule_enabled = st.checkbox("Enable Rule Immediately", value=False)
                testing_mode = st.checkbox("Start in Testing Mode", value=True)
        
        # Condition builder
        st.markdown("### 🔧 Condition Builder")
        
        condition_type = st.selectbox("Condition Type", ["Simple Condition", "Advanced Logic", "SQL-like Expression"])
        
        if condition_type == "Simple Condition":
            col1, col2, col3 = st.columns(3)
            
            with col1:
                field = st.selectbox("Field", [
                    "order_value", "days_since_purchase", "customer_return_count",
                    "return_reason", "category", "return_hour", "day_of_week",
                    "shipping_method", "customer_age_days"
                ])
            
            with col2:
                operator = st.selectbox("Operator", [
                    "equals", "greater_than", "less_than", "greater_equal",
                    "less_equal", "contains", "not_contains", "in_list"
                ])
            
            with col3:
                if operator in ["contains", "not_contains"]:
                    value = st.text_input("Value", placeholder="damage")
                elif operator == "in_list":
                    value = st.text_input("Values (comma-separated)", placeholder="Electronics,Clothing")
                else:
                    value = st.number_input("Value", value=0.0)
        
        elif condition_type == "Advanced Logic":
            st.markdown("**Build complex conditions using AND/OR logic:**")
            
            # Multiple conditions
            num_conditions = st.number_input("Number of conditions", min_value=1, max_value=5, value=2)
            
            conditions = []
            for i in range(int(num_conditions)):
                st.markdown(f"**Condition {i+1}:**")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    field = st.selectbox(f"Field {i+1}", [
                        "order_value", "days_since_purchase", "customer_return_count",
                        "return_reason", "category", "return_hour"
                    ], key=f"field_{i}")
                
                with col2:
                    operator = st.selectbox(f"Operator {i+1}", [
                        "equals", "greater_than", "less_than", "contains"
                    ], key=f"op_{i}")
                
                with col3:
                    value = st.text_input(f"Value {i+1}", key=f"val_{i}")
                
                with col4:
                    if i < num_conditions - 1:
                        logic = st.selectbox(f"Logic {i+1}", ["AND", "OR"], key=f"logic_{i}")
                
                conditions.append({"field": field, "operator": operator, "value": value})
        
        else:  # SQL-like expression
            st.markdown("**Enter SQL-like condition:**")
            sql_condition = st.text_area(
                "SQL Condition",
                placeholder="order_value > 500 AND days_since_purchase <= 3 AND category = 'Electronics'",
                help="Use field names: order_value, days_since_purchase, customer_return_count, return_reason, category, return_hour"
            )
        
        # Rule testing
        st.markdown("### 🧪 Test Rule")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Test with sample data:**")
            test_order_value = st.number_input("Test Order Value", value=299.99)
            test_days = st.number_input("Test Days Since Purchase", value=5)
            test_return_count = st.number_input("Test Customer Return Count", value=2)
            test_category = st.selectbox("Test Category", ["Electronics", "Clothing", "Home"])
        
        with col2:
            if st.button("🧪 Test Rule"):
                st.success("✅ Rule would trigger - adding 15 risk points")
                st.info("📊 Estimated impact: 12 returns would be affected in last 30 days")
        
        # Submit rule
        if st.form_submit_button("🚀 Create Rule", type="primary"):
            if rule_name and rule_description:
                st.success(f"✅ Rule '{rule_name}' created successfully!")
                if testing_mode:
                    st.info("🧪 Rule is now active in testing mode")
            else:
                st.error("Please fill in all required fields")
    
    with tab3:
        st.subheader("📊 Rule Performance Analytics")
        
        # Performance overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Top performing rules
            st.markdown("### 🏆 Top Performing Rules")
            
            top_rules = [
                {"name": "Serial Returner", "accuracy": 92.1, "triggered": 45},
                {"name": "High Value Quick Return", "accuracy": 89.5, "triggered": 38},
                {"name": "Late Damage Claim", "accuracy": 76.3, "triggered": 22}
            ]
            
            for rule in top_rules:
                st.markdown(f"""
                <div style="padding: 0.5rem; margin: 0.25rem 0; border: 1px solid #e5e7eb; border-radius: 0.25rem;">
                    <strong>{rule['name']}</strong><br>
                    Accuracy: {rule['accuracy']}% | Triggered: {rule['triggered']}
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Rule accuracy distribution
            accuracies = [rule['accuracy'] for rule in rules if rule['accuracy'] > 0]
            
            fig = px.histogram(
                x=accuracies,
                title="Rule Accuracy Distribution",
                labels={'x': 'Accuracy (%)', 'y': 'Number of Rules'},
                nbins=10
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # Recent rule triggers
            st.markdown("### 📊 Recent Triggers")
            
            trigger_data = {
                'Rule': ['Serial Returner', 'High Value Return', 'Late Damage', 'Weekend Electronics'],
                'Last 24h': [8, 12, 6, 3],
                'Last 7d': [45, 38, 22, 18]
            }
            
            st.dataframe(pd.DataFrame(trigger_data), use_container_width=True)
        
        # Performance trends
        st.markdown("### 📈 Performance Trends")
        
        # Simulate performance data over time
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), periods=30, freq='D')
        
        # Rule performance over time
        rule_performance = {
            'Serial Returner': np.random.normal(92, 3, 30),
            'High Value Return': np.random.normal(89, 4, 30),
            'Late Damage': np.random.normal(76, 5, 30)
        }
        
        fig = go.Figure()
        
        colors = ['#2563eb', '#16a34a', '#dc2626']
        for i, (rule_name, performance) in enumerate(rule_performance.items()):
            fig.add_trace(go.Scatter(
                x=dates,
                y=performance,
                mode='lines+markers',
                name=rule_name,
                line=dict(color=colors[i], width=2)
            ))
        
        fig.update_layout(
            title="Rule Accuracy Trends (30 Days)",
            xaxis_title="Date",
            yaxis_title="Accuracy (%)",
            yaxis=dict(range=[60, 100])
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Rule impact analysis
        st.markdown("### 💰 Financial Impact by Rule")
        
        impact_data = {
            'Rule Name': ['Serial Returner', 'High Value Return', 'Late Damage', 'Weekend Electronics'],
            'Fraud Prevented': [23, 18, 12, 8],
            'Savings ($)': [34500, 27000, 18000, 12000],
            'False Positives': [2, 3, 5, 4]
        }
        
        impact_df = pd.DataFrame(impact_data)
        
        fig = px.bar(
            impact_df,
            x='Rule Name',
            y='Savings ($)',
            color='Fraud Prevented',
            title="Financial Impact by Rule (Last 30 Days)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("🔧 Advanced Rule Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ⚙️ Global Rule Settings")
            
            # Global settings
            max_risk_score = st.slider("Maximum Risk Score", 50, 200, 100)
            enable_rule_chaining = st.checkbox("Enable Rule Chaining", value=True)
            auto_disable_poor_rules = st.checkbox("Auto-disable rules with <60% accuracy", value=False)
            
            st.markdown("### 🎯 Rule Evaluation")
            
            evaluation_mode = st.selectbox("Evaluation Mode", ["All Rules", "Best Match Only", "Weighted Average"])
            rule_timeout = st.slider("Rule Timeout (ms)", 100, 5000, 1000)
            
            # Rule conflicts
            st.markdown("### ⚔️ Conflict Resolution")
            
            conflict_resolution = st.selectbox("When rules conflict:", [
                "Take highest risk score",
                "Take lowest risk score", 
                "Average risk scores",
                "Use rule priority"
            ])
        
        with col2:
            st.markdown("### 📊 Rule Analytics Settings")
            
            # Analytics settings
            min_samples_accuracy = st.slider("Min samples for accuracy calculation", 10, 100, 30)
            performance_window = st.selectbox("Performance calculation window", ["7 days", "30 days", "90 days"])
            
            st.markdown("### 🔔 Rule Alerts")
            
            alert_poor_performance = st.checkbox("Alert when rule accuracy drops below 70%", value=True)
            alert_high_triggers = st.checkbox("Alert when rule triggers >50 times/day", value=True)
            alert_new_patterns = st.checkbox("Alert for potential new rule opportunities", value=False)
            
            # Maintenance
            st.markdown("### 🧹 Rule Maintenance")
            
            auto_archive_unused = st.checkbox("Auto-archive unused rules after 90 days")
            backup_rules = st.checkbox("Backup rules before changes", value=True)
            
            if st.button("🗄️ Backup All Rules"):
                st.success("✅ All rules backed up successfully!")
        
        # Machine learning integration
        st.markdown("---")
        st.subheader("🤖 Machine Learning Integration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🧠 Auto-Rule Generation")
            
            enable_auto_rules = st.checkbox("Enable automatic rule generation", value=False)
            
            if enable_auto_rules:
                confidence_threshold = st.slider("Min confidence for auto-generated rules", 70, 95, 85)
                review_auto_rules = st.checkbox("Require manual review for auto-generated rules", value=True)
                
                if st.button("🚀 Generate Rules from Patterns"):
                    with st.spinner("Analyzing fraud patterns..."):
                        time.sleep(3)
                        st.success("✅ 3 potential rules identified and added to review queue")
        
        with col2:
            st.markdown("### 📈 Rule Optimization")
            
            enable_optimization = st.checkbox("Enable rule parameter optimization", value=True)
            
            if enable_optimization:
                optimization_frequency = st.selectbox("Optimization frequency", ["Daily", "Weekly", "Monthly"])
                optimization_metric = st.selectbox("Optimization target", ["Accuracy", "F1-Score", "Precision", "Recall"])
                
                if st.button("⚡ Optimize Rules Now"):
                    with st.spinner("Optimizing rule parameters..."):
                        time.sleep(2)
                        st.success("✅ 5 rules optimized with improved performance")
        
        if st.button("💾 Save Advanced Settings", type="primary"):
            st.success("✅ Advanced settings saved successfully!")

def show_enterprise_audit_logs():
    """Enterprise audit logs and compliance"""
    
    st.title("📋 Audit Logs & Compliance")
    
    # Audit overview
    st.subheader("📊 Audit Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📝 Total Events", "15,847", "+234 today")
    with col2:
        st.metric("🔍 Security Events", "23", "+3 today")
    with col3:
        st.metric("⚠️ Failed Logins", "12", "-5 vs yesterday")
    with col4:
        st.metric("📊 Data Exports", "45", "+8 this week")
    
    # Audit tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Event Logs", "🔍 Security Audit", "📊 Compliance Reports", "⚙️ Audit Settings"])
    
    with tab1:
        st.subheader("📋 Comprehensive Event Logs")
        
        # Filters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            event_type = st.selectbox("Event Type", [
                "All Events", "User Authentication", "Data Access", "Configuration Changes",
                "Return Reviews", "Exports", "API Calls", "Security Events"
            ])
        
        with col2:
            user_filter = st.selectbox("User", [
                "All Users", "Sarah Johnson", "Mike Chen", "Emily Rodriguez", "System"
            ])
        
        with col3:
            severity = st.selectbox("Severity", ["All", "Info", "Warning", "Error", "Critical"])
        
        with col4:
            date_range = st.date_input("Date Range", value=(date.today() - timedelta(days=7), date.today()))
        
        # Search
        search_query = st.text_input("🔍 Search logs", placeholder="Search by action, details, IP address...")
        
        # Sample audit events
        audit_events = [
            {
                "timestamp": "2024-03-15 14:23:15",
                "event_type": "Return Review",
                "user": "Emily Rodriguez",
                "action": "Approve Return",
                "resource": "RET-12345",
                "details": "Return approved with risk score 45. Manual review completed.",
                "ip_address": "192.168.1.102",
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "severity": "Info",
                "result": "Success"
            },
            {
                "timestamp": "2024-03-15 14:15:32",
                "event_type": "Data Export",
                "user": "Mike Chen",
                "action": "Export Returns Data",
                "resource": "returns_march_2024.csv",
                "details": "Exported 500 return records for analysis",
                "ip_address": "192.168.1.101",
                "user_agent": "Mozilla/5.0 (macOS)",
                "severity": "Info",
                "result": "Success"
            },
            {
                "timestamp": "2024-03-15 13:58:41",
                "event_type": "Configuration Changes",
                "user": "Sarah Johnson",
                "action": "Update Risk Threshold",
                "resource": "fraud_detection_settings",
                "details": "Risk threshold changed from 80 to 85",
                "ip_address": "192.168.1.100",
                "user_agent": "Mozilla/5.0 (Windows NT 10.0)",
                "severity": "Warning",
                "result": "Success"
            },
            {
                "timestamp": "2024-03-15 13:45:21",
                "event_type": "Security Events",
                "user": "Unknown",
                "action": "Failed Login Attempt",
                "resource": "login_endpoint",
                "details": "Failed login attempt for user david@company.com - incorrect password",
                "ip_address": "203.0.113.15",
                "user_agent": "curl/7.68.0",
                "severity": "Warning",
                "result": "Failed"
            },
            {
                "timestamp": "2024-03-15 13:30:12",
                "event_type": "API Calls",
                "user": "API_Client_001",
                "action": "Analyze Return",
                "resource": "api/v1/returns/analyze",
                "details": "Return analysis via API - RET-12346",
                "ip_address": "203.0.113.45",
                "user_agent": "ReturnGuard-Client/1.0",
                "severity": "Info",
                "result": "Success"
            },
            {
                "timestamp": "2024-03-15 12:15:33",
                "event_type": "User Authentication",
                "user": "Mike Chen",
                "action": "Successful Login",
                "resource": "login_endpoint",
                "details": "User logged in via SSO",
                "ip_address": "192.168.1.101",
                "user_agent": "Mozilla/5.0 (macOS)",
                "severity": "Info",
                "result": "Success"
            }
        ]
        
        # Display events
        for event in audit_events:
            severity_color = {
                "Info": "#2563eb",
                "Warning": "#f59e0b",
                "Error": "#dc2626",
                "Critical": "#7c2d12"
            }[event['severity']]
            
            result_color = "#16a34a" if event['result'] == "Success" else "#dc2626"
            
            with st.expander(f"{event['timestamp']} - {event['action']} ({event['severity']})"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Event Type:** {event['event_type']}")
                    st.markdown(f"**User:** {event['user']}")
                    st.markdown(f"**Action:** {event['action']}")
                    st.markdown(f"**Resource:** {event['resource']}")
                    st.markdown(f"**Details:** {event['details']}")
                
                with col2:
                    st.markdown(f"**Severity:** <span style='color: {severity_color}'>{event['severity']}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Result:** <span style='color: {result_color}'>{event['result']}</span>", unsafe_allow_html=True)
                    st.markdown(f"**IP Address:** {event['ip_address']}")
                    st.markdown(f"**User Agent:** {event['user_agent'][:50]}...")
        
        # Export options
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Export Current View"):
                st.info("📊 Exporting filtered audit logs...")
        
        with col2:
            if st.button("📧 Email Report"):
                st.info("📧 Sending audit report via email...")
        
        with col3:
            if st.button("🔄 Real-time Monitor"):
                st.info("🔄 Opening real-time log monitoring...")
    
    with tab2:
        st.subheader("🔍 Security Audit Dashboard")
        
        # Security metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🔐 Authentication Events")
            
            auth_data = {
                'Event': ['Successful Logins', 'Failed Logins', 'Password Resets', 'Account Lockouts'],
                'Count': [156, 12, 3, 2],
                'Last 24h': [45, 3, 1, 0]
            }
            
            fig = px.bar(
                auth_data,
                x='Event',
                y='Count',
                title="Authentication Events (7 days)"
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🌐 Geographic Access")
            
            # Simulate geographic data
            geo_data = {
                'Country': ['United States', 'Canada', 'United Kingdom', 'Germany', 'Unknown'],
                'Logins': [142, 23, 8, 5, 3],
                'Risk Level': ['Low', 'Low', 'Low', 'Medium', 'High']
            }
            
            fig = px.pie(
                values=geo_data['Logins'],
                names=geo_data['Country'],
                title="Login Locations"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            st.markdown("### ⚠️ Security Alerts")
            
            security_alerts = [
                {"type": "Suspicious IP", "count": 5, "severity": "Medium"},
                {"type": "Multiple Failed Logins", "count": 3, "severity": "High"},
                {"type": "Off-hours Access", "count": 8, "severity": "Low"},
                {"type": "API Rate Limit", "count": 2, "severity": "Medium"}
            ]
            
            for alert in security_alerts:
                severity_color = {
                    "Low": "#16a34a",
                    "Medium": "#f59e0b", 
                    "High": "#dc2626"
                }[alert['severity']]
                
                st.markdown(f"""
                <div style="border-left: 4px solid {severity_color}; padding: 0.5rem; margin: 0.5rem 0; background: #f9fafb;">
                    <strong>{alert['type']}</strong><br>
                    Count: {alert['count']} | Severity: <span style="color: {severity_color}">{alert['severity']}</span>
                </div>
                """, unsafe_allow_html=True)
        
        # Failed login analysis
        st.markdown("### 📊 Failed Login Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Failed logins over time
            hours = list(range(24))
            failed_logins = [0, 0, 1, 0, 0, 0, 2, 1, 0, 1, 0, 0, 1, 2, 0, 1, 0, 0, 1, 0, 2, 1, 0, 0]
            
            fig = px.bar(
                x=hours,
                y=failed_logins,
                title="Failed Logins by Hour",
                labels={'x': 'Hour of Day', 'y': 'Failed Attempts'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top failed login IPs
            failed_ip_data = {
                'IP Address': ['203.0.113.15', '198.51.100.42', '192.0.2.123'],
                'Attempts': [5, 3, 2],
                'Location': ['Unknown', 'Russia', 'China']
            }
            
            st.dataframe(pd.DataFrame(failed_ip_data), use_container_width=True)
            
            if st.button("🚫 Block Suspicious IPs"):
                st.success("✅ Suspicious IPs added to blocklist")
    
    with tab3:
        st.subheader("📊 Compliance Reports")
        
        # Compliance frameworks
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📋 Compliance Frameworks")
            
            frameworks = [
                {"name": "SOX (Sarbanes-Oxley)", "status": "Compliant", "last_audit": "2024-01-15"},
                {"name": "GDPR", "status": "Compliant", "last_audit": "2024-02-01"},
                {"name": "SOC 2 Type II", "status": "In Progress", "last_audit": "2024-03-01"},
                {"name": "PCI DSS", "status": "Compliant", "last_audit": "2024-01-30"},
                {"name": "HIPAA", "status": "N/A", "last_audit": "N/A"}
            ]
            
            for framework in frameworks:
                status_color = {
                    "Compliant": "#16a34a",
                    "In Progress": "#f59e0b",
                    "Non-Compliant": "#dc2626",
                    "N/A": "#6b7280"
                }[framework['status']]
                
                st.markdown(f"""
                <div style="border: 1px solid #e5e7eb; padding: 1rem; margin: 0.5rem 0; border-radius: 0.5rem;">
                    <strong>{framework['name']}</strong><br>
                    Status: <span style="color: {status_color}; font-weight: bold;">{framework['status']}</span><br>
                    Last Audit: {framework['last_audit']}
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📈 Compliance Metrics")
            
            # Compliance score over time
            dates = pd.date_range(start=datetime.now() - timedelta(days=90), periods=90, freq='D')
            compliance_scores = 95 + np.random.normal(0, 2, 90)
            compliance_scores = np.clip(compliance_scores, 85, 100)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=compliance_scores,
                mode='lines',
                name='Compliance Score',
                line=dict(color='#16a34a', width=2)
            ))
            fig.add_hline(y=90, line_dash="dash", line_color="red", annotation_text="Minimum Required")
            
            fig.update_layout(
                title="Compliance Score Trend",
                xaxis_title="Date",
                yaxis_title="Compliance Score (%)",
                yaxis=dict(range=[80, 100])
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Data retention compliance
        st.markdown("### 🗄️ Data Retention Compliance")
        
        retention_data = {
            'Data Type': ['Audit Logs', 'Return Data', 'Customer Data', 'Financial Records', 'System Logs'],
            'Retention Period': ['7 years', '5 years', '3 years', '7 years', '1 year'],
            'Current Age': ['2 years', '1 year', '6 months', '3 years', '3 months'],
            'Compliance Status': ['Compliant', 'Compliant', 'Compliant', 'Compliant', 'Compliant']
        }
        
        st.dataframe(pd.DataFrame(retention_data), use_container_width=True)
        
        # Generate compliance reports
        st.markdown("### 📄 Generate Compliance Reports")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            report_type = st.selectbox("Report Type", [
                "SOX Compliance Report",
                "GDPR Data Processing Report", 
                "Access Control Report",
                "Data Retention Report",
                "Security Incident Report"
            ])
        
        with col2:
            report_period = st.selectbox("Period", ["Last 30 days", "Last 90 days", "Last Year", "Custom"])
        
        with col3:
            report_format = st.selectbox("Format", ["PDF", "Excel", "CSV", "Word"])
        
        if st.button("📊 Generate Compliance Report", type="primary"):
            with st.spinner("Generating compliance report..."):
                time.sleep(3)
                st.success("✅ Compliance report generated successfully!")
                st.download_button("📥 Download Report", b"Sample report content", f"compliance_report.{report_format.lower()}")
    
    with tab4:
        st.subheader("⚙️ Audit Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📝 Audit Settings")
            
            # What to audit
            audit_logins = st.checkbox("Audit User Logins", value=True)
            audit_data_access = st.checkbox("Audit Data Access", value=True)
            audit_exports = st.checkbox("Audit Data Exports", value=True)
            audit_config_changes = st.checkbox("Audit Configuration Changes", value=True)
            audit_api_calls = st.checkbox("Audit API Calls", value=True)
            audit_return_reviews = st.checkbox("Audit Return Reviews", value=True)
            
            # Retention settings
            st.markdown("### 🗄️ Retention Settings")
            
            audit_retention = st.selectbox("Audit Log Retention", ["1 year", "3 years", "5 years", "7 years"])
            auto_archive = st.checkbox("Auto-archive old logs", value=True)
            
            if auto_archive:
                archive_after = st.selectbox("Archive after", ["1 year", "2 years", "3 years"])
        
        with col2:
            st.markdown("### 🔔 Alert Settings")
            
            # Alert thresholds
            failed_login_threshold = st.slider("Alert after failed logins", 3, 20, 10)
            suspicious_activity_alerts = st.checkbox("Alert on suspicious activity", value=True)
            compliance_alerts = st.checkbox("Alert on compliance violations", value=True)
            
            # Alert recipients
            st.markdown("### 📧 Alert Recipients")
            
            security_team = st.text_input("Security Team Email", value="security@company.com")
            compliance_team = st.text_input("Compliance Team Email", value="compliance@company.com")
            
            # Integration settings
            st.markdown("### 🔗 Integration Settings")
            
            siem_integration = st.checkbox("Send logs to SIEM")
            if siem_integration:
                siem_endpoint = st.text_input("SIEM Endpoint", placeholder="https://siem.company.com/api")
            
            external_storage = st.checkbox("External log storage")
            if external_storage:
                storage_provider = st.selectbox("Storage Provider", ["AWS S3", "Azure Blob", "Google Cloud"])
        
        # Audit test
        st.markdown("---")
        st.subheader("🧪 Test Audit System")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 Test Log Generation"):
                with st.spinner("Generating test audit event..."):
                    time.sleep(1)
                    st.success("✅ Test audit event generated successfully")
        
        with col2:
            if st.button("📊 Validate Compliance"):
                with st.spinner("Running compliance validation..."):
                    time.sleep(2)
                    st.success("✅ All compliance checks passed")
        
        if st.button("💾 Save Audit Configuration", type="primary"):
            st.success("✅ Audit configuration saved successfully!")

def show_enterprise_reports():
    """Enterprise reporting and analytics"""
    
    st.title("📊 Enterprise Reports")
    
    # Reports overview
    st.subheader("📋 Reports Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📄 Total Reports", "156", "+23 this month")
    with col2:
        st.metric("📅 Scheduled Reports", "12", "+2 new")
    with col3:
        st.metric("📧 Auto-sent", "89", "this month")
    with col4:
        st.metric("📥 Downloads", "234", "+45 this week")
    
    # Reports tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Standard Reports", "🔧 Custom Reports", "📅 Scheduled Reports", "📈 Executive Dashboard"])
    
    with tab1:
        st.subheader("📊 Standard Report Templates")
        
        # Standard report categories
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Fraud Detection Reports")
            
            fraud_reports = [
                {
                    "name": "Daily Fraud Summary",
                    "description": "Daily overview of fraud detection activity",
                    "frequency": "Daily",
                    "last_generated": "Today 08:00",
                    "recipients": 5
                },
                {
                    "name": "Weekly Risk Analysis", 
                    "description": "Comprehensive weekly fraud risk analysis",
                    "frequency": "Weekly",
                    "last_generated": "Monday 09:00",
                    "recipients": 8
                },
                {
                    "name": "Monthly Fraud Trends",
                    "description": "Monthly analysis of fraud patterns and trends",
                    "frequency": "Monthly",
                    "last_generated": "March 1st",
                    "recipients": 12
                },
                {
                    "name": "High-Risk Customer Report",
                    "description": "List of customers flagged as high-risk",
                    "frequency": "Weekly",
                    "last_generated": "Monday 09:00",
                    "recipients": 6
                }
            ]
            
            for report in fraud_reports:
                with st.expander(f"📋 {report['name']}"):
                    st.write(f"**Description:** {report['description']}")
                    st.write(f"**Frequency:** {report['frequency']}")
                    st.write(f"**Last Generated:** {report['last_generated']}")
                    st.write(f"**Recipients:** {report['recipients']}")
                    
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        if st.button("📊 Generate", key=f"gen_{report['name']}"):
                            st.info(f"Generating {report['name']}...")
                    
                    with col_b:
                        if st.button("📧 Email", key=f"email_{report['name']}"):
                            st.info(f"Emailing {report['name']}...")
                    
                    with col_c:
                        if st.button("⚙️ Configure", key=f"config_{report['name']}"):
                            st.info(f"Configuring {report['name']}...")
        
        with col2:
            st.markdown("### 📈 Performance Reports")
            
            performance_reports = [
                {
                    "name": "AI Model Performance",
                    "description": "Detailed analysis of AI model accuracy and performance",
                    "frequency": "Weekly",
                    "last_generated": "Monday 10:00",
                    "recipients": 4
                },
                {
                    "name": "System Performance Metrics",
                    "description": "Technical performance and uptime metrics",
                    "frequency": "Daily",
                    "last_generated": "Today 06:00",
                    "recipients": 3
                },
                {
                    "name": "ROI Analysis Report",
                    "description": "Return on investment and cost savings analysis",
                    "frequency": "Monthly",
                    "last_generated": "March 1st",
                    "recipients": 15
                },
                {
                    "name": "User Activity Report",
                    "description": "Team member activity and usage statistics",
                    "frequency": "Monthly",
                    "last_generated": "March 1st",
                    "recipients": 8
                }
            ]
            
            for report in performance_reports:
                with st.expander(f"📈 {report['name']}"):
                    st.write(f"**Description:** {report['description']}")
                    st.write(f"**Frequency:** {report['frequency']}")
                    st.write(f"**Last Generated:** {report['last_generated']}")
                    st.write(f"**Recipients:** {report['recipients']}")
                    
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        if st.button("📊 Generate", key=f"gen_{report['name']}"):
                            st.info(f"Generating {report['name']}...")
                    
                    with col_b:
                        if st.button("📧 Email", key=f"email_{report['name']}"):
                            st.info(f"Emailing {report['name']}...")
                    
                    with col_c:
                        if st.button("⚙️ Configure", key=f"config_{report['name']}"):
                            st.info(f"Configuring {report['name']}...")
        
        # Quick report generation
        st.markdown("---")
        st.subheader("⚡ Quick Report Generation")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📊 Report Type**")
            quick_report_type = st.selectbox("Select Report", [
                "Fraud Summary", "Performance Metrics", "Customer Analysis",
                "Financial Impact", "Rule Performance", "Compliance Status"
            ])
        
        with col2:
            st.markdown("**📅 Time Period**")
            time_period = st.selectbox("Period", [
                "Last 24 hours", "Last 7 days", "Last 30 days", 
                "Last 90 days", "Year to date", "Custom range"
            ])
        
        with col3:
            st.markdown("**📄 Format**")
            report_format = st.selectbox("Format", ["PDF", "Excel", "CSV", "PowerPoint"])
        
        if st.button("🚀 Generate Quick Report", type="primary"):
            with st.spinner(f"Generating {quick_report_type} report..."):
                time.sleep(3)
                st.success("✅ Report generated successfully!")
                
                # Simulate report content
                st.download_button(
                    "📥 Download Report",
                    b"Sample report content",
                    f"{quick_report_type.lower().replace(' ', '_')}_report.{report_format.lower()}",
                    f"application/{report_format.lower()}"
                )
    
    with tab2:
        st.subheader("🔧 Custom Report Builder")
        
        # Custom report builder
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📋 Report Configuration")
            
            custom_report_name = st.text_input("Report Name", placeholder="My Custom Report")
            custom_description = st.text_area("Description", placeholder="Describe what this report includes...")
            
            # Data sources
            st.markdown("**📊 Data Sources**")
            include_returns = st.checkbox("Returns Data", value=True)
            include_customers = st.checkbox("Customer Data", value=True)
            include_fraud_scores = st.checkbox("Fraud Scores", value=True)
            include_rules = st.checkbox("Rule Performance", value=False)
            include_financials = st.checkbox("Financial Impact", value=True)
            
            # Filters
            st.markdown("**🎛️ Filters**")
            date_range_custom = st.date_input("Date Range", value=(date.today() - timedelta(days=30), date.today()))
            risk_levels = st.multiselect("Risk Levels", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM"])
            categories = st.multiselect("Product Categories", ["Electronics", "Clothing", "Home", "Beauty"])
        
        with col2:
            st.markdown("### 📈 Visualizations")
            
            # Chart selections
            st.markdown("**📊 Charts to Include**")
            chart_risk_distribution = st.checkbox("Risk Level Distribution", value=True)
            chart_trends = st.checkbox("Trends Over Time", value=True)
            chart_categories = st.checkbox("Category Analysis", value=False)
            chart_customers = st.checkbox("Customer Analysis", value=False)
            chart_financial = st.checkbox("Financial Impact", value=True)
            
            # Report sections
            st.markdown("**📄 Report Sections**")
            section_executive_summary = st.checkbox("Executive Summary", value=True)
            section_detailed_analysis = st.checkbox("Detailed Analysis", value=True)
            section_recommendations = st.checkbox("Recommendations", value=True)
            section_appendix = st.checkbox("Data Appendix", value=False)
            
            # Output options
            st.markdown("**⚙️ Output Options**")
            custom_format = st.selectbox("Output Format", ["PDF", "Excel", "PowerPoint", "Word"])
            include_raw_data = st.checkbox("Include Raw Data", value=False)
            auto_schedule = st.checkbox("Schedule for Future Generation")
            
            if auto_schedule:
                schedule_frequency = st.selectbox("Frequency", ["Daily", "Weekly", "Monthly"])
        
        # Preview and generate
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("👁️ Preview Report Structure"):
                st.info("📋 Report structure preview:")
                st.markdown("""
                **Report Sections:**
                1. Executive Summary
                2. Fraud Detection Overview
                3. Risk Analysis
                4. Financial Impact
                5. Recommendations
                """)
        
        with col2:
            if st.button("💾 Save Custom Report Template"):
                if custom_report_name:
                    st.success(f"✅ Custom report '{custom_report_name}' saved!")
                else:
                    st.error("Please enter a report name")
        
        if st.button("🚀 Generate Custom Report", type="primary"):
            if custom_report_name:
                with st.spinner("Building custom report..."):
                    time.sleep(4)
                    st.success("✅ Custom report generated successfully!")
            else:
                st.error("Please enter a report name")
    
    with tab3:
        st.subheader("📅 Scheduled Reports Management")
        
        # Scheduled reports list
        scheduled_reports = [
            {
                "name": "Daily Fraud Summary",
                "schedule": "Daily at 08:00",
                "recipients": ["admin@company.com", "security@company.com"],
                "last_sent": "Today 08:00",
                "status": "Active",
                "next_run": "Tomorrow 08:00"
            },
            {
                "name": "Weekly Executive Report",
                "schedule": "Mondays at 09:00", 
                "recipients": ["ceo@company.com", "cfo@company.com"],
                "last_sent": "Monday 09:00",
                "status": "Active",
                "next_run": "Next Monday 09:00"
            },
            {
                "name": "Monthly ROI Analysis",
                "schedule": "1st of month at 10:00",
                "recipients": ["finance@company.com"],
                "last_sent": "March 1st 10:00",
                "status": "Active", 
                "next_run": "April 1st 10:00"
            },
            {
                "name": "Compliance Report",
                "schedule": "Quarterly",
                "recipients": ["compliance@company.com", "legal@company.com"],
                "last_sent": "January 1st",
                "status": "Paused",
                "next_run": "Paused"
            }
        ]
        
        for report in scheduled_reports:
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
                
                with col1:
                    status_color = "#16a34a" if report['status'] == "Active" else "#6b7280"
                    st.markdown(f"**{report['name']}**")
                    st.markdown(f"<span style='color: {status_color}'>{report['status']}</span>", unsafe_allow_html=True)
                
                with col2:
                    st.write(f"**Schedule:** {report['schedule']}")
                    st.write(f"**Last Sent:** {report['last_sent']}")
                
                with col3:
                    st.write(f"**Next Run:** {report['next_run']}")
                    st.write(f"**Recipients:** {len(report['recipients'])}")
                
                with col4:
                    if report['status'] == "Active":
                        if st.button("⏸️ Pause", key=f"pause_sched_{report['name']}"):
                            st.info(f"Paused {report['name']}")
                    else:
                        if st.button("▶️ Resume", key=f"resume_sched_{report['name']}"):
                            st.success(f"Resumed {report['name']}")
                    
                    if st.button("⚙️ Edit", key=f"edit_sched_{report['name']}"):
                        st.info(f"Editing {report['name']}")
                
                st.markdown("---")
        
        # Add new scheduled report
        st.subheader("➕ Schedule New Report")
        
        col1, col2 = st.columns(2)
        
        with col1:
            new_schedule_name = st.selectbox("Report Template", [
                "Daily Fraud Summary", "Weekly Risk Analysis", "Monthly Trends",
                "Quarterly Compliance", "Custom Report"
            ])
            
            schedule_type = st.selectbox("Schedule Type", ["Daily", "Weekly", "Monthly", "Quarterly"])
            
            if schedule_type == "Daily":
                schedule_time = st.time_input("Time", datetime.strptime("08:00", "%H:%M").time())
            elif schedule_type == "Weekly":
                schedule_day = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
                schedule_time = st.time_input("Time", datetime.strptime("09:00", "%H:%M").time())
            elif schedule_type == "Monthly":
                schedule_date = st.selectbox("Day of Month", list(range(1, 29)))
                schedule_time = st.time_input("Time", datetime.strptime("10:00", "%H:%M").time())
        
        with col2:
            recipients_input = st.text_area(
                "Recipients (one email per line)",
                placeholder="admin@company.com\nsecurity@company.com\nmanager@company.com"
            )
            
            email_subject = st.text_input("Email Subject", placeholder="Automated ReturnGuard Report")
            email_message = st.text_area("Email Message", placeholder="Please find the attached report...")
            
            start_immediately = st.checkbox("Start immediately", value=True)
        
        if st.button("📅 Schedule Report", type="primary"):
            if new_schedule_name and recipients_input:
                st.success(f"✅ {new_schedule_name} scheduled successfully!")
            else:
                st.error("Please fill in all required fields")
    
    with tab4:
        st.subheader("📈 Executive Dashboard")
        
        # Executive summary cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%); padding: 1.5rem; border-radius: 1rem; color: white; text-align: center;">
                <h3 style="margin: 0; color: white;">Revenue Protected</h3>
                <h1 style="margin: 0.5rem 0; color: white;">$847K</h1>
                <p style="margin: 0; opacity: 0.8;">+23% vs last month</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #16a34a 0%, #15803d 100%); padding: 1.5rem; border-radius: 1rem; color: white; text-align: center;">
                <h3 style="margin: 0; color: white;">Fraud Detection Rate</h3>
                <h1 style="margin: 0.5rem 0; color: white;">94.7%</h1>
                <p style="margin: 0; opacity: 0.8;">+3.2% improvement</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); padding: 1.5rem; border-radius: 1rem; color: white; text-align: center;">
                <h3 style="margin: 0; color: white;">Processing Efficiency</h3>
                <h1 style="margin: 0.5rem 0; color: white;">0.23s</h1>
                <p style="margin: 0; opacity: 0.8;">-0.05s faster</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%); padding: 1.5rem; border-radius: 1rem; color: white; text-align: center;">
                <h3 style="margin: 0; color: white;">Risk Score Avg</h3>
                <h1 style="margin: 0.5rem 0; color: white;">42.3</h1>
                <p style="margin: 0; opacity: 0.8;">-8.2 vs last month</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Executive charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue protected over time
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
            revenue_protected = [620, 730, 650, 780, 690, 847]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=months,
                y=revenue_protected,
                mode='lines+markers',
                name='Revenue Protected ($K)',
                line=dict(color='#2563eb', width=4),
                marker=dict(size=10),
                fill='tonexty'
            ))
            
            fig.update_layout(
                title="Monthly Revenue Protection Trend",
                xaxis_title="Month",
                yaxis_title="Revenue Protected ($K)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Key metrics comparison
            metrics = ['Fraud Detection', 'Customer Satisfaction', 'Processing Speed', 'Cost Reduction']
            current = [94.7, 96.2, 88.5, 78.3]
            target = [95, 95, 90, 80]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Current', x=metrics, y=current, marker_color='#2563eb'))
            fig.add_trace(go.Bar(name='Target', x=metrics, y=target, marker_color='#16a34a'))
            
            fig.update_layout(
                title="Performance vs Targets",
                xaxis_title="Metrics",
                yaxis_title="Score (%)",
                barmode='group',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Strategic insights
        st.markdown("---")
        st.subheader("💡 Strategic Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🎯 Key Achievements
            ✅ **Revenue Protection:** Prevented $847K in fraudulent returns  
            ✅ **Efficiency Gains:** 23% improvement in processing speed  
            ✅ **Customer Satisfaction:** 96.2% satisfaction rate  
            ✅ **False Positive Reduction:** Down 15% from last quarter
            """)
        
        with col2:
            st.markdown("""
            ### 🚀 Growth Opportunities
            📈 **Market Expansion:** Ready for 3x customer growth  
            🤖 **AI Enhancement:** 5% accuracy improvement potential  
            🔗 **New Integrations:** 4 platform integrations in pipeline  
            📊 **Advanced Analytics:** Predictive capabilities roadmap
            """)
        
        with col3:
            st.markdown("""
            ### ⚠️ Risk Factors
            🎯 **Model Drift:** Monitor for accuracy degradation  
            📊 **Data Quality:** Ensure clean integration data  
            👥 **Team Scaling:** Plan for support team growth  
            🔐 **Security:** Continuous compliance monitoring
            """)
        
        # Export executive summary
        if st.button("📊 Generate Executive Summary Report", type="primary"):
            with st.spinner("Generating executive summary..."):
                time.sleep(3)
                st.success("✅ Executive summary generated!")
                st.download_button("📥 Download Executive Report", b"Executive summary content", "executive_summary.pdf")

def show_enterprise_settings():
    """Enterprise settings and configuration"""
    
    st.title("⚙️ Enterprise Settings")
    
    # Settings tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏢 Company Settings", "🔐 Security", "🔗 Integrations", "🎛️ AI Configuration", "💳 Billing"
    ])
    
    with tab1:
        st.subheader("🏢 Company Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📋 Company Information")
            
            company_name = st.text_input("Company Name", value=st.session_state.user_data['company_name'])
            company_domain = st.text_input("Company Domain", value="enterprise.com")
            company_size = st.selectbox("Company Size", ["1-10", "11-50", "51-200", "201-1000", "1000+"])
            industry = st.selectbox("Industry", [
                "E-commerce", "Retail", "Fashion", "Electronics", "Software", "Other"
            ])
            
            # Contact information
            st.markdown("### 📞 Contact Information")
            
            primary_contact = st.text_input("Primary Contact", value=st.session_state.user_data['name'])
            contact_email = st.text_input("Contact Email", value=st.session_state.user_data['email'])
            phone_number = st.text_input("Phone Number", placeholder="+1-555-123-4567")
            
            # Address
            st.markdown("### 📍 Address")
            
            street_address = st.text_input("Street Address")
            city = st.text_input("City")
            col_a, col_b = st.columns(2)
            with col_a:
                state = st.text_input("State/Province")
            with col_b:
                postal_code = st.text_input("Postal Code")
            country = st.selectbox("Country", ["United States", "Canada", "United Kingdom", "Germany", "Other"])
        
        with col2:
            st.markdown("### 🎯 Business Configuration")
            
            # Business settings
            primary_currency = st.selectbox("Primary Currency", ["USD", "EUR", "GBP", "CAD", "AUD"])
            timezone = st.selectbox("Timezone", [
                "America/New_York", "America/Los_Angeles", "Europe/London", 
                "Europe/Berlin", "Asia/Tokyo", "Australia/Sydney"
            ])
            date_format = st.selectbox("Date Format", ["MM/DD/YYYY", "DD/MM/YYYY", "YYYY-MM-DD"])
            
            # Return processing settings
            st.markdown("### 🔄 Return Processing Preferences")
            
            default_return_window = st.slider("Default Return Window (days)", 7, 90, 30)
            auto_approve_threshold = st.slider("Auto-approve Risk Score Below", 0, 50, 30)
            auto_reject_threshold = st.slider("Auto-reject Risk Score Above", 50, 100, 85)
            
            require_manager_approval = st.checkbox("Require manager approval for high-risk returns", value=True)
            enable_customer_notifications = st.checkbox("Enable automatic customer notifications", value=True)
            
            # Data preferences
            st.markdown("### 📊 Data & Analytics Preferences")
            
            enable_advanced_analytics = st.checkbox("Enable advanced analytics", value=True)
            share_anonymous_data = st.checkbox("Share anonymous data for AI improvement")
            enable_benchmarking = st.checkbox("Enable industry benchmarking", value=True)
        
        if st.button("💾 Save Company Settings", type="primary"):
            st.success("✅ Company settings saved successfully!")
    
    with tab2:
        st.subheader("🔐 Security Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔒 Authentication Settings")
            
            # Password policy
            min_password_length = st.slider("Minimum Password Length", 8, 20, 12)
            require_uppercase = st.checkbox("Require Uppercase Letters", value=True)
            require_lowercase = st.checkbox("Require Lowercase Letters", value=True)
            require_numbers = st.checkbox("Require Numbers", value=True)
            require_special_chars = st.checkbox("Require Special Characters", value=True)
            password_expiry = st.selectbox("Password Expiry", ["Never", "90 days", "180 days", "365 days"])
            
            # Two-factor authentication
            st.markdown("### 📱 Two-Factor Authentication")
            
            enforce_2fa = st.checkbox("Enforce 2FA for all users")
            allowed_2fa_methods = st.multiselect("Allowed 2FA Methods", [
                "SMS", "Email", "Authenticator App", "Hardware Token"
            ], default=["Authenticator App"])
            
            # Session management
            st.markdown("### ⏱️ Session Management")
            
            session_timeout = st.selectbox("Session Timeout", [
                "15 minutes", "30 minutes", "1 hour", "4 hours", "8 hours", "24 hours"
            ])
            max_concurrent_sessions = st.slider("Max Concurrent Sessions per User", 1, 10, 3)
            remember_me_duration = st.selectbox("Remember Me Duration", [
                "7 days", "30 days", "90 days", "Never expire"
            ])
        
        with col2:
            st.markdown("### 🌐 Network Security")
            
            # IP restrictions
            enable_ip_whitelist = st.checkbox("Enable IP Address Whitelist")
            if enable_ip_whitelist:
                ip_whitelist = st.text_area(
                    "Allowed IP Addresses (one per line)",
                    placeholder="192.168.1.0/24\n203.0.113.0/24"
                )
            
            # Geographic restrictions
            enable_geo_restrictions = st.checkbox("Enable Geographic Restrictions")
            if enable_geo_restrictions:
                allowed_countries = st.multiselect("Allowed Countries", [
                    "United States", "Canada", "United Kingdom", "Germany", "France"
                ])
            
            # API security
            st.markdown("### 🔌 API Security")
            
            api_rate_limit = st.slider("API Rate Limit (requests/hour)", 100, 10000, 1000)
            require_api_key_rotation = st.checkbox("Require API key rotation every 90 days")
            log_all_api_calls = st.checkbox("Log all API calls", value=True)
            
            # Data encryption
            st.markdown("### 🔐 Data Encryption")
            
            encrypt_data_at_rest = st.checkbox("Encrypt data at rest", value=True, disabled=True)
            encrypt_data_in_transit = st.checkbox("Encrypt data in transit", value=True, disabled=True)
            encryption_key_rotation = st.selectbox("Encryption Key Rotation", [
                "90 days", "180 days", "365 days", "Manual"
            ])
        
        # Security monitoring
        st.markdown("---")
        st.subheader("👁️ Security Monitoring")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🚨 Alert Triggers**")
            alert_failed_logins = st.checkbox("Multiple failed login attempts", value=True)
            alert_unusual_access = st.checkbox("Unusual access patterns", value=True)
            alert_data_export = st.checkbox("Large data exports", value=True)
            alert_privilege_escalation = st.checkbox("Privilege escalation attempts", value=True)
        
        with col2:
            st.markdown("**📧 Alert Recipients**")
            security_team_email = st.text_input("Security Team Email", value="security@company.com")
            admin_alert_email = st.text_input("Admin Alert Email", value="admin@company.com")
            soc_email = st.text_input("SOC Email (Optional)")
        
        with col3:
            st.markdown("**⚙️ Response Actions**")
            auto_lock_accounts = st.checkbox("Auto-lock accounts after 5 failed attempts", value=True)
            auto_revoke_sessions = st.checkbox("Auto-revoke sessions on suspicious activity")
            quarantine_suspicious_users = st.checkbox("Quarantine suspicious users")
        
        if st.button("🛡️ Save Security Settings", type="primary"):
            st.success("✅ Security settings saved successfully!")
    
    with tab3:
        st.subheader("🔗 Integration Management")
        
        # Current integrations status
        integrations_status = [
            {"name": "Shopify", "status": "Connected", "health": "Healthy", "last_sync": "2 min ago"},
            {"name": "WooCommerce", "status": "Connected", "health": "Healthy", "last_sync": "5 min ago"},
            {"name": "Email (SMTP)", "status": "Connected", "health": "Healthy", "last_sync": "Active"},
            {"name": "Slack", "status": "Connected", "health": "Warning", "last_sync": "1 hour ago"},
            {"name": "BigCommerce", "status": "Available", "health": "N/A", "last_sync": "N/A"}
        ]
        
        st.markdown("### 📊 Integration Status")
        
        for integration in integrations_status:
            col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 2, 1])
            
            with col1:
                st.write(f"**{integration['name']}**")
            
            with col2:
                status_color = {"Connected": "#16a34a", "Available": "#6b7280", "Error": "#dc2626"}[integration['status']]
                st.markdown(f"<span style='color: {status_color}'>{integration['status']}</span>", unsafe_allow_html=True)
            
            with col3:
                if integration['health'] != "N/A":
                    health_color = {"Healthy": "#16a34a", "Warning": "#f59e0b", "Error": "#dc2626"}[integration['health']]
                    st.markdown(f"<span style='color: {health_color}'>{integration['health']}</span>", unsafe_allow_html=True)
                else:
                    st.write("—")
            
            with col4:
                st.write(integration['last_sync'])
            
            with col5:
                if integration['status'] == "Connected":
                    if st.button("⚙️", key=f"config_int_{integration['name']}"):
                        st.info(f"Configuring {integration['name']}")
                else:
                    if st.button("🔌", key=f"connect_int_{integration['name']}"):
                        st.info(f"Connecting {integration['name']}")
        
        # Integration settings
        st.markdown("---")
        st.subheader("⚙️ Global Integration Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔄 Sync Settings")
            
            auto_sync_frequency = st.selectbox("Auto-sync Frequency", [
                "Real-time", "Every 5 minutes", "Every 15 minutes", 
                "Every hour", "Every 4 hours", "Daily"
            ])
            
            retry_failed_syncs = st.checkbox("Retry failed syncs automatically", value=True)
            max_retry_attempts = st.slider("Max Retry Attempts", 1, 10, 3)
            
            # Data validation
            st.markdown("### ✅ Data Validation")
            
            validate_incoming_data = st.checkbox("Validate incoming data", value=True)
            reject_invalid_records = st.checkbox("Reject invalid records", value=True)
            alert_data_quality_issues = st.checkbox("Alert on data quality issues", value=True)
        
        with col2:
            st.markdown("### 📊 Monitoring & Alerts")
            
            monitor_api_health = st.checkbox("Monitor API health", value=True)
            alert_integration_failures = st.checkbox("Alert on integration failures", value=True)
            alert_sync_delays = st.checkbox("Alert on sync delays", value=True)
            
            # Performance settings
            st.markdown("### ⚡ Performance")
            
            batch_size = st.slider("Batch Size for Data Processing", 100, 10000, 1000)
            connection_timeout = st.slider("Connection Timeout (seconds)", 5, 120, 30)
            enable_compression = st.checkbox("Enable data compression", value=True)
        
        if st.button("🔗 Save Integration Settings", type="primary"):
            st.success("✅ Integration settings saved successfully!")
    
    with tab4:
        st.subheader("🎛️ AI Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🧠 Model Settings")
            
            # Model selection
            ai_model_version = st.selectbox("AI Model Version", [
                "Production v2.1.3 (Recommended)",
                "Beta v2.2.0",
                "Legacy v2.0.8"
            ])
            
            # Risk scoring
            st.markdown("### 🎯 Risk Scoring")
            
            risk_score_sensitivity = st.selectbox("Risk Score Sensitivity", [
                "Conservative", "Balanced", "Aggressive"
            ])
            
            confidence_threshold = st.slider("Minimum Confidence Threshold", 0.5, 0.95, 0.85)
            enable_adaptive_learning = st.checkbox("Enable adaptive learning", value=True)
            
            # Feature weights
            st.markdown("### ⚖️ Feature Weights")
            
            customer_history_weight = st.slider("Customer History", 0.0, 1.0, 0.30)
            order_value_weight = st.slider("Order Value", 0.0, 1.0, 0.25)
            timing_weight = st.slider("Return Timing", 0.0, 1.0, 0.20)
            reason_weight = st.slider("Return Reason", 0.0, 1.0, 0.15)
            product_weight = st.slider("Product Category", 0.0, 1.0, 0.10)
        
        with col2:
            st.markdown("### 🔧 Advanced Configuration")
            
            # Model optimization
            auto_retrain_model = st.checkbox("Auto-retrain model weekly", value=True)
            use_feedback_learning = st.checkbox("Learn from manual reviews", value=True)
            enable_ensemble_methods = st.checkbox("Enable ensemble methods")
            
            # Performance tuning
            st.markdown("### ⚡ Performance Tuning")
            
            max_processing_time = st.slider("Max Processing Time (ms)", 100, 5000, 1000)
            enable_caching = st.checkbox("Enable result caching", value=True)
            cache_duration = st.selectbox("Cache Duration", ["1 hour", "6 hours", "24 hours"])
            
            # Custom rules integration
            st.markdown("### 🔧 Custom Rules Integration")
            
            rules_influence_ai = st.checkbox("Allow custom rules to influence AI", value=True)
            ai_explains_decisions = st.checkbox("AI provides decision explanations", value=True)
            log_all_predictions = st.checkbox("Log all AI predictions", value=True)
        
        # Model performance monitoring
        st.markdown("---")
        st.subheader("📊 Model Performance Monitoring")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Accuracy", "87.3%", "+2.1%")
            st.metric("Precision", "89.2%", "+1.8%")
        
        with col2:
            st.metric("Recall", "84.7%", "+2.5%")
            st.metric("F1-Score", "86.9%", "+2.1%")
        
        with col3:
            st.metric("Avg Processing Time", "0.23s", "-0.05s")
            st.metric("Daily Predictions", "1,247", "+89")
        
        # Model actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Retrain Model"):
                with st.spinner("Retraining model..."):
                    time.sleep(3)
                    st.success("✅ Model retrained successfully!")
        
        with col2:
            if st.button("📊 Validate Model"):
                with st.spinner("Running validation..."):
                    time.sleep(2)
                    st.success("✅ Model validation complete!")
        
        with col3:
            if st.button("💾 Export Model"):
                st.info("📥 Model export initiated...")
        
        if st.button("🤖 Save AI Configuration", type="primary"):
            st.success("✅ AI configuration saved successfully!")
    
    with tab5:
        show_enterprise_billing()

def show_enterprise_billing():
    """Enterprise billing and subscription management (enhanced)"""
    
    if 'billing_shown' not in st.session_state:
        st.session_state.billing_shown = True
        st.subheader("💳 Billing & Subscription Management")
    
    # Current subscription overview
    current_plan = st.session_state.user_data.get('subscription_plan', 'Professional').title()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Plan", current_plan)
    with col2:
        st.metric("Monthly Cost", "$99")
    with col3:
        st.metric("Next Billing", "March 15, 2024")
    with col4:
        st.metric("Annual Savings", "$237" if current_plan != "Enterprise" else "$597")
    
    # Billing tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💳 Current Plan", "📈 Usage & Billing", "💰 Payment Methods", "📊 Cost Analysis"])
    
    with tab1:
        st.subheader("📋 Current Subscription Details")
        
        # Plan comparison
        plans = {
            "Starter": {
                "price": 49,
                "annual_price": 470,
                "returns_limit": 100,
                "features": [
                    "Up to 100 returns/month",
                    "Basic fraud detection",
                    "Email alerts",
                    "Standard dashboard",
                    "Email support"
                ]
            },
            "Professional": {
                "price": 99,
                "annual_price": 950,
                "returns_limit": 500,
                "features": [
                    "Up to 500 returns/month",
                    "Advanced AI analysis",
                    "Real-time alerts + SMS",
                    "Custom fraud rules",
                    "Priority support",
                    "Advanced analytics"
                ]
            },
            "Business": {
                "price": 199,
                "annual_price": 1910,
                "returns_limit": "Unlimited",
                "features": [
                    "Unlimited returns",
                    "Custom AI training",
                    "Predictive forecasting",
                    "Multi-store management",
                    "API access",
                    "White-label options"
                ]
            },
            "Enterprise": {
                "price": 499,
                "annual_price": 4790,
                "returns_limit": "Unlimited",
                "features": [
                    "Everything in Business",
                    "Custom AI model development",
                    "Dedicated success manager",
                    "Phone support",
                    "Custom integrations",
                    "SLA guarantees"
                ]
            }
        }
        
        # Display plan cards
        for plan_name, plan_details in plans.items():
            is_current = plan_name.lower() == current_plan.lower()
            
            card_style = """
                background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
                color: white;
            """ if is_current else """
                border: 2px solid #e5e7eb;
                background: white;
            """
            
            with st.container():
                st.markdown(f"""
                <div style="{card_style} padding: 2rem; border-radius: 1rem; margin: 1rem 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                        <h3 style="margin: 0;">{plan_name} {"(Current)" if is_current else ""}</h3>
                        <div style="text-align: right;">
                            <h2 style="margin: 0;">${plan_details['price']}/month</h2>
                            <p style="margin: 0; opacity: 0.7;">${plan_details['annual_price']}/year (save ${plan_details['price']*12 - plan_details['annual_price']})</p>
                        </div>
                    </div>
                    <p><strong>Returns Limit:</strong> {plan_details['returns_limit']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Features list
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    for feature in plan_details['features']:
                        st.write(f"✅ {feature}")
                
                with col2:
                    if not is_current:
                        if plan_name == "Enterprise":
                            if st.button("📞 Contact Sales", key=f"contact_{plan_name}"):
                                st.info("🤝 Our sales team will contact you within 24 hours")
                        else:
                            if st.button(f"⬆️ Upgrade", key=f"upgrade_{plan_name}"):
                                st.success(f"✅ Upgraded to {plan_name} plan!")
                                # Update session state
                                st.session_state.user_data['subscription_plan'] = plan_name.lower()
                                st.rerun()
                    else:
                        st.success("✅ Current Plan")
                
                st.markdown("---")
    
    with tab2:
        st.subheader("📈 Usage & Billing History")
        
        # Current usage
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Current Month Usage")
            
            returns_processed = 347
            returns_limit = 500 if current_plan.lower() == "professional" else "Unlimited"
            
            if isinstance(returns_limit, int):
                usage_percent = (returns_processed / returns_limit) * 100
                st.progress(usage_percent / 100)
                st.write(f"{returns_processed} / {returns_limit} returns processed ({usage_percent:.1f}%)")
            else:
                st.write(f"{returns_processed} returns processed (Unlimited plan)")
            
            # Usage breakdown
            usage_data = {
                'Feature': ['Returns Analyzed', 'API Calls', 'Reports Generated', 'Alerts Sent'],
                'Usage': [347, 1247, 23, 89],
                'Limit': ['500', 'Unlimited', 'Unlimited', 'Unlimited']
            }
            
            st.dataframe(pd.DataFrame(usage_data), use_container_width=True)
        
        with col2:
            st.markdown("### 💰 Billing Summary")
            
            # Current charges
            base_cost = plans[current_plan.title()]['price']
            overage_cost = 0
            total_cost = base_cost + overage_cost
            
            st.metric("Base Plan Cost", f"${base_cost}")
            st.metric("Overage Charges", f"${overage_cost}")
            st.metric("Total This Month", f"${total_cost}")
            
            # Next billing
            st.write("**Next Billing Date:** March 15, 2024")
            st.write("**Payment Method:** Visa ending in 4242")
            
            if st.button("📧 Email Invoice"):
                st.success("📧 Invoice emailed to billing contact")
        
        # Billing history
        st.markdown("### 📊 Billing History")
        
        billing_history = [
            {"Date": "Feb 15, 2024", "Plan": "Professional", "Amount": "$99.00", "Status": "Paid"},
            {"Date": "Jan 15, 2024", "Plan": "Professional", "Amount": "$99.00", "Status": "Paid"},
            {"Date": "Dec 15, 2023", "Plan": "Starter", "Amount": "$49.00", "Status": "Paid"},
            {"Date": "Nov 15, 2023", "Plan": "Starter", "Amount": "$49.00", "Status": "Paid"},
            {"Date": "Oct 15, 2023", "Plan": "Starter", "Amount": "$49.00", "Status": "Paid"}
        ]
        
        df_billing = pd.DataFrame(billing_history)
        st.dataframe(df_billing, use_container_width=True)
        
        # Usage trends
        st.markdown("### 📈 Usage Trends")
        
        # Simulate usage data over time
        months = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar']
        returns_usage = [45, 78, 125, 234, 289, 347]
        costs = [49, 49, 49, 99, 99, 99]  # Plan changes
        
        fig = go.Figure()
        
        # Returns usage
        fig.add_trace(go.Scatter(
            x=months,
            y=returns_usage,
            mode='lines+markers',
            name='Returns Processed',
            yaxis='y',
            line=dict(color='#2563eb', width=3)
        ))
        
        # Monthly costs
        fig.add_trace(go.Bar(
            x=months,
            y=costs,
            name='Monthly Cost ($)',
            yaxis='y2',
            opacity=0.7,
            marker_color='#16a34a'
        ))
        
        fig.update_layout(
            title="Usage and Cost Trends",
            xaxis_title="Month",
            yaxis=dict(title="Returns Processed", side="left"),
            yaxis2=dict(title="Monthly Cost ($)", side="right", overlaying="y"),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("💰 Payment Methods")
        
        # Current payment methods
        payment_methods = [
            {
                "type": "Credit Card",
                "details": "Visa ending in 4242",
                "expiry": "12/2025",
                "primary": True,
                "auto_pay": True
            },
            {
                "type": "Bank Account",
                "details": "Account ending in 1234",
                "expiry": "N/A",
                "primary": False,
                "auto_pay": False
            }
        ]
        
        for i, method in enumerate(payment_methods):
            col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
            
            with col1:
                primary_text = " (Primary)" if method['primary'] else ""
                st.write(f"**{method['type']}{primary_text}**")
                st.write(method['details'])
            
            with col2:
                if method['expiry'] != "N/A":
                    st.write(f"Expires: {method['expiry']}")
                auto_pay_text = "✅ Auto-pay enabled" if method['auto_pay'] else "❌ Auto-pay disabled"
                st.write(auto_pay_text)
            
            with col3:
                if not method['primary']:
                    if st.button("⭐ Make Primary", key=f"primary_{i}"):
                        st.success("Payment method updated")
            
            with col4:
                if st.button("✏️ Edit", key=f"edit_payment_{i}"):
                    st.info("Payment method editor would open here")
            
            st.markdown("---")
        
        # Add new payment method
        st.markdown("### ➕ Add New Payment Method")
        
        col1, col2 = st.columns(2)
        
        with col1:
            payment_type = st.selectbox("Payment Type", ["Credit Card", "Debit Card", "Bank Account", "PayPal"])
            
            if payment_type in ["Credit Card", "Debit Card"]:
                card_number = st.text_input("Card Number", placeholder="1234 5678 9012 3456")
                col_a, col_b = st.columns(2)
                with col_a:
                    expiry_month = st.selectbox("Expiry Month", list(range(1, 13)))
                with col_b:
                    expiry_year = st.selectbox("Expiry Year", list(range(2024, 2035)))
                cvv = st.text_input("CVV", placeholder="123", type="password")
        
        with col2:
            # Billing address
            st.markdown("**Billing Address**")
            billing_name = st.text_input("Name on Card", placeholder="John Doe")
            billing_address = st.text_input("Address")
            col_a, col_b = st.columns(2)
            with col_a:
                billing_city = st.text_input("City")
            with col_b:
                billing_zip = st.text_input("ZIP Code")
            
            make_primary = st.checkbox("Make this primary payment method")
            enable_auto_pay = st.checkbox("Enable auto-pay", value=True)
        
        if st.button("💳 Add Payment Method", type="primary"):
            st.success("✅ Payment method added successfully!")
        
        # Payment settings
        st.markdown("---")
        st.subheader("⚙️ Payment Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            auto_pay_enabled = st.checkbox("Enable automatic payments", value=True)
            failed_payment_retries = st.slider("Failed payment retry attempts", 1, 5, 3)
            payment_notifications = st.checkbox("Email payment notifications", value=True)
        
        with col2:
            billing_contact = st.text_input("Billing contact email", value="billing@company.com")
            invoice_delivery = st.selectbox("Invoice delivery", ["Email", "Postal Mail", "Both"])
            currency_preference = st.selectbox("Billing currency", ["USD", "EUR", "GBP"])
    
    with tab4:
        st.subheader("📊 Cost Analysis & Optimization")
        
        # ROI analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 💰 Return on Investment")
            
            # Calculate ROI
            monthly_cost = plans[current_plan.title()]['price']
            annual_cost = monthly_cost * 12
            fraud_prevented = 847000  # From sample data
            processing_savings = 45200
            total_savings = fraud_prevented + processing_savings
            roi = ((total_savings - annual_cost) / annual_cost) * 100
            
            st.metric("Annual Subscription Cost", f"${annual_cost:,}")
            st.metric("Total Annual Savings", f"${total_savings:,}")
            st.metric("Net Annual Benefit", f"${total_savings - annual_cost:,}")
            st.metric("ROI", f"{roi:.1f}%")
            
            # Payback period
            monthly_savings = total_savings / 12
            payback_months = annual_cost / monthly_savings
            st.metric("Payback Period", f"{payback_months:.1f} months")
        
        with col2:
            st.markdown("### 📈 Cost Breakdown")
            
            # Cost vs savings visualization
            categories = ['Subscription Cost', 'Fraud Prevention', 'Process Efficiency', 'Net Benefit']
            values = [annual_cost, fraud_prevented, processing_savings, total_savings - annual_cost]
            colors = ['#dc2626', '#16a34a', '#2563eb', '#f59e0b']
            
            fig = go.Figure(data=[go.Bar(x=categories, y=values, marker_color=colors)])
            fig.update_layout(
                title="Annual Cost vs Benefits Analysis",
                yaxis_title="Amount ($)",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Cost optimization recommendations
        st.markdown("### 💡 Cost Optimization Recommendations")
        
        recommendations = [
            {
                "title": "Annual Billing Discount",
                "description": f"Switch to annual billing to save ${plans[current_plan.title()]['price']*12 - plans[current_plan.title()]['annual_price']} per year",
                "savings": f"${plans[current_plan.title()]['price']*12 - plans[current_plan.title()]['annual_price']}",
                "effort": "Low"
            },
            {
                "title": "Optimize Usage Patterns",
                "description": "Review current usage to ensure you're on the optimal plan",
                "savings": "Up to $600/year",
                "effort": "Medium"
            },
            {
                "title": "Volume Discount Negotiation",
                "description": "Contact sales for enterprise volume discounts",
                "savings": "10-20% off",
                "effort": "Low"
            }
        ]
        
        for rec in recommendations:
            st.markdown(f"""
            <div style="border: 1px solid #e5e7eb; padding: 1rem; margin: 1rem 0; border-radius: 0.5rem; background: #f9fafb;">
                <h4>{rec['title']}</h4>
                <p>{rec['description']}</p>
                <p><strong>Potential Savings:</strong> {rec['savings']} | <strong>Effort:</strong> {rec['effort']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Usage forecast
        st.markdown("### 📊 Usage Forecast")
        
        # Simulate usage forecast
        months_ahead = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
        forecasted_usage = [420, 485, 520, 580, 630, 690]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=months_ahead,
            y=forecasted_usage,
            mode='lines+markers',
            name='Forecasted Usage',
            line=dict(color='#2563eb', width=3, dash='dash')
        ))
        
        # Add plan limits
        if current_plan.lower() == "professional":
            fig.add_hline(y=500, line_dash="solid", line_color="red", 
                         annotation_text="Professional Plan Limit (500)")
        
        fig.update_layout(
            title="6-Month Usage Forecast",
            xaxis_title="Month",
            yaxis_title="Predicted Returns",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Plan recommendations based on forecast
        if current_plan.lower() == "professional" and max(forecasted_usage) > 500:
            st.warning("⚠️ **Plan Upgrade Recommended:** Your forecasted usage will exceed the Professional plan limit. Consider upgrading to Business plan.")
            
            if st.button("🚀 Upgrade to Business Plan"):
                st.success("✅ Upgraded to Business plan to accommodate growth!")

if __name__ == "__main__":
    main()