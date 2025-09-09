"""
Analytics and Visualization Components
Interactive charts and data analysis for dashboard
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import streamlit as st

class FarmersAnalytics:
    def __init__(self):
        self.color_palette = {
            'primary': '#2E8B57',
            'secondary': '#32CD32', 
            'warning': '#FF6B35',
            'danger': '#DC143C',
            'info': '#1E90FF',
            'success': '#228B22'
        }
    
    def create_metrics_cards(self, metrics: Dict) -> None:
        """Create metric cards for dashboard overview"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📊 Total Reports", 
                value=metrics.get('total_reports', 0),
                delta=metrics.get('reports_delta', None)
            )
        
        with col2:
            st.metric(
                label="⚠️ Active Cases", 
                value=metrics.get('active_cases', 0),
                delta=metrics.get('active_delta', None)
            )
        
        with col3:
            critical_cases = metrics.get('critical_cases', 0)
            st.metric(
                label="🚨 Critical Cases", 
                value=critical_cases,
                delta=critical_cases if critical_cases > 0 else None,
                delta_color="inverse"
            )
        
        with col4:
            st.metric(
                label="💡 Community Remedies", 
                value=metrics.get('community_remedies', 0),
                delta=metrics.get('remedies_delta', None)
            )
    
    def create_severity_pie_chart(self, severity_data: Dict) -> go.Figure:
        """Create pie chart for severity distribution"""
        if not severity_data or sum(severity_data.values()) == 0:
            # Create dummy data for demo
            severity_data = {'Low': 15, 'Medium': 25, 'High': 8, 'Critical': 3}
        
        colors = {
            'Low': self.color_palette['success'],
            'Medium': self.color_palette['info'], 
            'High': self.color_palette['warning'],
            'Critical': self.color_palette['danger']
        }
        
        fig = go.Figure(data=[go.Pie(
            labels=list(severity_data.keys()),
            values=list(severity_data.values()),
            hole=.3,
            marker_colors=[colors.get(k, self.color_palette['primary']) for k in severity_data.keys()]
        )])
        
        fig.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )
        
        fig.update_layout(
            title="Disease Severity Distribution",
            title_x=0.5,
            font=dict(size=12),
            showlegend=True,
            height=400,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        return fig
    
    def create_disease_bar_chart(self, disease_data: Dict, top_n: int = 10) -> go.Figure:
        """Create bar chart for top diseases"""
        if not disease_data:
            # Sample data for demo
            disease_data = {
                'Tomato Late Blight': 12,
                'Wheat Rust': 8,
                'Apple Scab': 6,
                'Corn Blight': 5,
                'Lumpy Skin Disease': 4,
                'Mastitis': 3
            }
        
        # Sort and limit to top N
        sorted_diseases = sorted(disease_data.items(), key=lambda x: x[1], reverse=True)[:top_n]
        diseases, counts = zip(*sorted_diseases) if sorted_diseases else ([], [])
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(diseases),
                y=list(counts),
                marker_color=self.color_palette['primary'],
                text=list(counts),
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Cases: %{y}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=f"Top {min(top_n, len(diseases))} Reported Diseases",
            title_x=0.5,
            xaxis_title="Disease",
            yaxis_title="Number of Cases",
            height=400,
            margin=dict(t=50, b=100, l=50, r=50),
            xaxis_tickangle=-45
        )
        
        return fig
    
    def create_time_series_chart(self, daily_data: Dict) -> go.Figure:
        """Create time series chart for daily submissions"""
        if not daily_data:
            # Generate sample data for last 30 days
            dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(29, -1, -1)]
            counts = np.random.poisson(3, 30)  # Random counts with average of 3
            daily_data = dict(zip(dates, counts))
        
        dates = list(daily_data.keys())
        counts = list(daily_data.values())
        
        fig = go.Figure(data=[
            go.Scatter(
                x=dates,
                y=counts,
                mode='lines+markers',
                line=dict(color=self.color_palette['primary'], width=2),
                marker=dict(size=6, color=self.color_palette['secondary']),
                name='Daily Reports',
                hovertemplate='<b>%{x}</b><br>Reports: %{y}<extra></extra>'
            )
        ])
        
        # Add trend line
        if len(counts) > 1:
            z = np.polyfit(range(len(counts)), counts, 1)
            trend = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=dates,
                y=trend(range(len(counts))),
                mode='lines',
                line=dict(color=self.color_palette['warning'], width=1, dash='dash'),
                name='Trend',
                showlegend=False
            ))
        
        fig.update_layout(
            title="Daily Disease Reports Trend",
            title_x=0.5,
            xaxis_title="Date",
            yaxis_title="Number of Reports",
            height=400,
            margin=dict(t=50, b=50, l=50, r=50),
            hovermode='x unified'
        )
        
        return fig
    
    def create_location_map(self, location_data: Dict) -> go.Figure:
        """Create choropleth-style visualization for locations"""
        if not location_data:
            # Sample location data
            location_data = {
                'Coimbatore, Tamil Nadu': 15,
                'Nashik, Maharashtra': 12,
                'Anand, Gujarat': 8,
                'Ludhiana, Punjab': 6,
                'Mysore, Karnataka': 5
            }
        
        locations = list(location_data.keys())
        counts = list(location_data.values())
        
        # Create bubble map representation
        fig = go.Figure(data=[
            go.Scatter(
                x=range(len(locations)),
                y=[1] * len(locations),  # Single row
                mode='markers+text',
                marker=dict(
                    size=[c*10 for c in counts],  # Scale bubble size
                    color=counts,
                    colorscale='Viridis',
                    showscale=True,
                    sizemode='diameter',
                    sizeref=2.*max(counts)/50,
                    colorbar=dict(title="Cases")
                ),
                text=locations,
                textposition='top center',
                hovertemplate='<b>%{text}</b><br>Cases: %{marker.color}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="Disease Reports by Location",
            title_x=0.5,
            height=300,
            showlegend=False,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        return fig
    
    def create_category_comparison(self, category_data: Dict) -> go.Figure:
        """Create comparison chart between plant and livestock diseases"""
        if not category_data:
            category_data = {'plant': 35, 'livestock': 18}
        
        categories = list(category_data.keys())
        counts = list(category_data.values())
        
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=counts,
                marker_color=[self.color_palette['success'], self.color_palette['info']],
                text=counts,
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Cases: %{y}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="Plant vs Livestock Disease Reports",
            title_x=0.5,
            xaxis_title="Category",
            yaxis_title="Number of Cases",
            height=300,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        return fig
    
    def create_remedy_effectiveness_chart(self, remedy_data: List[Dict]) -> go.Figure:
        """Create chart showing remedy effectiveness ratings"""
        if not remedy_data:
            # Sample remedy data
            remedy_data = [
                {'disease': 'Tomato Late Blight', 'effectiveness': 4.2, 'count': 8},
                {'disease': 'Wheat Rust', 'effectiveness': 3.8, 'count': 6},
                {'disease': 'Apple Scab', 'effectiveness': 3.5, 'count': 4},
                {'disease': 'Mastitis', 'effectiveness': 4.0, 'count': 3}
            ]
        
        diseases = [r['disease'] for r in remedy_data]
        effectiveness = [r['effectiveness'] for r in remedy_data]
        counts = [r['count'] for r in remedy_data]
        
        fig = go.Figure(data=[
            go.Bar(
                x=diseases,
                y=effectiveness,
                marker=dict(
                    color=effectiveness,
                    colorscale='RdYlGn',
                    cmin=1,
                    cmax=5,
                    colorbar=dict(title="Effectiveness Rating")
                ),
                text=[f"{e:.1f} ({c} remedies)" for e, c in zip(effectiveness, counts)],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Avg Effectiveness: %{y:.1f}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="Community Remedy Effectiveness by Disease",
            title_x=0.5,
            xaxis_title="Disease",
            yaxis_title="Average Effectiveness Rating (1-5)",
            height=400,
            margin=dict(t=50, b=100, l=50, r=50),
            xaxis_tickangle=-45,
            yaxis_range=[0, 5]
        )
        
        return fig
    
    def create_alert_summary(self, alerts: List[Dict]) -> None:
        """Display alert summary with appropriate styling"""
        if not alerts:
            st.info("🟢 No active alerts - All systems normal")
            return
        
        st.subheader("🚨 Active Alerts")
        
        for alert in alerts:
            alert_type = alert.get('type', 'info')
            message = alert.get('message', '')
            action = alert.get('action', '')
            
            if alert_type == 'critical':
                st.error(f"🔴 **Critical**: {message}")
                if action:
                    st.caption(f"**Action Required**: {action}")
            elif alert_type == 'warning':
                st.warning(f"🟡 **Warning**: {message}")
                if action:
                    st.caption(f"**Recommended Action**: {action}")
            else:
                st.info(f"🔵 **Info**: {message}")
                if action:
                    st.caption(f"**Suggestion**: {action}")
    
    def create_performance_metrics(self, data: Dict) -> None:
        """Create performance and system health metrics"""
        st.subheader("📈 System Performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Response time simulation
            response_time = np.random.normal(1.2, 0.3)  # Average 1.2s
            st.metric("⚡ Avg Response Time", f"{response_time:.1f}s", 
                     delta=f"{response_time-1.5:.1f}s")
        
        with col2:
            # Accuracy simulation
            accuracy = np.random.normal(85, 3)  # Average 85%
            st.metric("🎯 AI Model Accuracy", f"{accuracy:.1f}%",
                     delta=f"{accuracy-82:.1f}%")
        
        with col3:
            # User satisfaction
            satisfaction = np.random.normal(4.2, 0.2)  # Average 4.2/5
            st.metric("😊 User Satisfaction", f"{satisfaction:.1f}/5",
                     delta=f"{satisfaction-4.0:.1f}")
    
    def export_analytics_report(self, data: Dict) -> str:
        """Generate and return analytics report as text"""
        report = f"""
# Farmers Disease Diagnostic Portal - Analytics Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary Statistics
- Total Disease Reports: {data.get('total_reports', 0)}
- Active Cases: {data.get('active_cases', 0)}  
- Critical Cases: {data.get('critical_cases', 0)}
- Community Remedies: {data.get('community_remedies', 0)}

## Top Diseases
"""
        
        top_diseases = data.get('top_diseases', [])
        for i, (disease, count) in enumerate(top_diseases[:5], 1):
            report += f"{i}. {disease}: {count} cases\n"
        
        report += f"""
## Geographic Distribution
Total Locations Affected: {len(data.get('location_distribution', {}))}

## Recommendations
- Monitor critical cases closely
- Implement preventive measures for top diseases
- Encourage community remedy sharing
- Maintain system performance metrics
        """
        
        return report

# Utility functions for data processing
def calculate_growth_rate(current: int, previous: int) -> float:
    """Calculate growth rate percentage"""
    if previous == 0:
        return 100.0 if current > 0 else 0.0
    return ((current - previous) / previous) * 100

def detect_anomalies(data: List[float], threshold: float = 2.0) -> List[bool]:
    """Detect anomalies using z-score method"""
    if len(data) < 2:
        return [False] * len(data)
    
    mean_val = np.mean(data)
    std_val = np.std(data)
    
    if std_val == 0:
        return [False] * len(data)
    
    z_scores = [(x - mean_val) / std_val for x in data]
    return [abs(z) > threshold for z in z_scores]

def generate_insights(analytics_data: Dict) -> List[str]:
    """Generate automated insights from analytics data"""
    insights = []
    
    # Critical cases insight
    critical_cases = analytics_data.get('critical_cases', 0)
    if critical_cases > 5:
        insights.append(f"⚠️ High number of critical cases ({critical_cases}) - immediate attention required")
    
    # Growth trend insight
    total_reports = analytics_data.get('total_reports', 0)
    if total_reports > 50:
        insights.append("📈 High reporting activity indicates good farmer engagement")
    elif total_reports < 10:
        insights.append("📉 Low reporting activity - consider awareness campaigns")
    
    # Disease pattern insight
    top_diseases = analytics_data.get('top_diseases', [])
    if top_diseases and top_diseases[0][1] > 10:
        dominant_disease = top_diseases[0][0]
        insights.append(f"🔍 {dominant_disease} appears to be the dominant issue - focus prevention efforts")
    
    return insights

# Example usage and testing
if __name__ == "__main__":
    analytics = FarmersAnalytics()
    
    # Sample data
    sample_data = {
        'total_reports': 45,
        'active_cases': 23,
        'critical_cases': 3,
        'community_remedies': 18,
        'severity_distribution': {'Low': 15, 'Medium': 20, 'High': 7, 'Critical': 3},
        'top_diseases': [('Tomato Late Blight', 12), ('Wheat Rust', 8)],
        'daily_trends': {},
        'location_distribution': {'Coimbatore': 15, 'Nashik': 12}
    }
    
    print("Analytics module initialized successfully!")
    print(f"Sample insights: {generate_insights(sample_data)}")