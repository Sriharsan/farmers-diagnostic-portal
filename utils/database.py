"""
Database Management for Farmers Portal
JSON-based lightweight database for demo/development
"""

import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import uuid
import pandas as pd

class FarmersDatabase:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.ensure_data_directory()
        
        # Database files
        self.submissions_file = os.path.join(data_dir, "disease_submissions.json")
        self.remedies_file = os.path.join(data_dir, "community_remedies.json")
        self.analytics_file = os.path.join(data_dir, "analytics_data.json")
        self.users_file = os.path.join(data_dir, "users.json")
        
        # Initialize files if they don't exist
        self.initialize_database()
    
    def ensure_data_directory(self):
        """Create data directory if it doesn't exist"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def initialize_database(self):
        """Initialize JSON database files with empty structures"""
        
        # Disease submissions
        if not os.path.exists(self.submissions_file):
            initial_submissions = {
                "submissions": [],
                "last_updated": datetime.now().isoformat(),
                "total_count": 0
            }
            self.save_json(self.submissions_file, initial_submissions)
        
        # Community remedies
        if not os.path.exists(self.remedies_file):
            initial_remedies = {
                "remedies": [],
                "last_updated": datetime.now().isoformat(),
                "total_count": 0
            }
            self.save_json(self.remedies_file, initial_remedies)
        
        # Analytics data
        if not os.path.exists(self.analytics_file):
            initial_analytics = {
                "daily_stats": {},
                "disease_counts": {},
                "location_stats": {},
                "severity_stats": {"Low": 0, "Medium": 0, "High": 0, "Critical": 0},
                "last_updated": datetime.now().isoformat()
            }
            self.save_json(self.analytics_file, initial_analytics)
        
        # Users (basic info)
        if not os.path.exists(self.users_file):
            initial_users = {
                "users": [],
                "total_count": 0,
                "last_updated": datetime.now().isoformat()
            }
            self.save_json(self.users_file, initial_users)
    
    def load_json(self, filepath: str) -> Dict:
        """Load JSON data from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def save_json(self, filepath: str, data: Dict):
        """Save data to JSON file"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving to {filepath}: {e}")
    
    # Disease Submissions
    def add_disease_submission(self, farmer_name: str, location: str, disease_name: str, 
                             category: str, description: str, severity: str, 
                             image_data: Optional[str] = None) -> str:
        """Add new disease submission"""
        
        submission_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()
        
        new_submission = {
            "id": submission_id,
            "farmer_name": farmer_name,
            "location": location,
            "disease_name": disease_name,
            "category": category,  # 'plant' or 'livestock'
            "description": description,
            "severity": severity,
            "image_data": image_data,
            "timestamp": timestamp,
            "status": "Open",
            "expert_verified": False,
            "treatment_applied": None,
            "outcome": None
        }
        
        # Load existing data
        data = self.load_json(self.submissions_file)
        data["submissions"].append(new_submission)
        data["total_count"] = len(data["submissions"])
        data["last_updated"] = timestamp
        
        # Save updated data
        self.save_json(self.submissions_file, data)
        
        # Update analytics
        self.update_analytics_on_submission(new_submission)
        
        return submission_id
    
    def get_disease_submissions(self, limit: Optional[int] = None, 
                               status: Optional[str] = None) -> List[Dict]:
        """Get disease submissions with optional filtering"""
        data = self.load_json(self.submissions_file)
        submissions = data.get("submissions", [])
        
        # Filter by status if specified
        if status:
            submissions = [s for s in submissions if s.get("status") == status]
        
        # Sort by timestamp (most recent first)
        submissions = sorted(submissions, key=lambda x: x.get("timestamp", ""), reverse=True)
        
        # Limit results if specified
        if limit:
            submissions = submissions[:limit]
        
        return submissions
    
    def update_submission_status(self, submission_id: str, status: str, 
                               treatment_applied: Optional[str] = None,
                               outcome: Optional[str] = None) -> bool:
        """Update submission status"""
        data = self.load_json(self.submissions_file)
        
        for submission in data["submissions"]:
            if submission["id"] == submission_id:
                submission["status"] = status
                submission["last_updated"] = datetime.now().isoformat()
                if treatment_applied:
                    submission["treatment_applied"] = treatment_applied
                if outcome:
                    submission["outcome"] = outcome
                
                self.save_json(self.submissions_file, data)
                return True
        
        return False
    
    # Community Remedies
    def add_community_remedy(self, farmer_name: str, disease_name: str, 
                           remedy_description: str, effectiveness_rating: int,
                           category: str) -> str:
        """Add community remedy"""
        
        remedy_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()
        
        new_remedy = {
            "id": remedy_id,
            "farmer_name": farmer_name,
            "disease_name": disease_name,
            "remedy_description": remedy_description,
            "effectiveness_rating": effectiveness_rating,  # 1-5 scale
            "category": category,
            "timestamp": timestamp,
            "verified": False,
            "upvotes": 0,
            "downvotes": 0,
            "comments": []
        }
        
        # Load existing data
        data = self.load_json(self.remedies_file)
        data["remedies"].append(new_remedy)
        data["total_count"] = len(data["remedies"])
        data["last_updated"] = timestamp
        
        # Save updated data
        self.save_json(self.remedies_file, data)
        
        return remedy_id
    
    def get_community_remedies(self, disease_name: Optional[str] = None,
                              category: Optional[str] = None,
                              limit: Optional[int] = None) -> List[Dict]:
        """Get community remedies with optional filtering"""
        data = self.load_json(self.remedies_file)
        remedies = data.get("remedies", [])
        
        # Filter by disease name
        if disease_name:
            remedies = [r for r in remedies if disease_name.lower() in r.get("disease_name", "").lower()]
        
        # Filter by category
        if category:
            remedies = [r for r in remedies if r.get("category") == category]
        
        # Sort by effectiveness rating and upvotes
        remedies = sorted(remedies, 
                         key=lambda x: (x.get("effectiveness_rating", 0), x.get("upvotes", 0)), 
                         reverse=True)
        
        # Limit results
        if limit:
            remedies = remedies[:limit]
        
        return remedies
    
    def vote_remedy(self, remedy_id: str, vote_type: str) -> bool:
        """Vote on a remedy (upvote/downvote)"""
        data = self.load_json(self.remedies_file)
        
        for remedy in data["remedies"]:
            if remedy["id"] == remedy_id:
                if vote_type == "up":
                    remedy["upvotes"] = remedy.get("upvotes", 0) + 1
                elif vote_type == "down":
                    remedy["downvotes"] = remedy.get("downvotes", 0) + 1
                
                self.save_json(self.remedies_file, data)
                return True
        
        return False
    
    # Analytics Functions
    def update_analytics_on_submission(self, submission: Dict):
        """Update analytics when new submission is added"""
        data = self.load_json(self.analytics_file)
        
        # Update daily stats
        date_key = submission["timestamp"][:10]  # YYYY-MM-DD
        if date_key not in data["daily_stats"]:
            data["daily_stats"][date_key] = 0
        data["daily_stats"][date_key] += 1
        
        # Update disease counts
        disease = submission["disease_name"]
        if disease not in data["disease_counts"]:
            data["disease_counts"][disease] = 0
        data["disease_counts"][disease] += 1
        
        # Update location stats
        location = submission["location"]
        if location not in data["location_stats"]:
            data["location_stats"][location] = 0
        data["location_stats"][location] += 1
        
        # Update severity stats
        severity = submission["severity"]
        if severity in data["severity_stats"]:
            data["severity_stats"][severity] += 1
        
        data["last_updated"] = datetime.now().isoformat()
        self.save_json(self.analytics_file, data)
    
    def get_analytics_summary(self) -> Dict:
        """Get comprehensive analytics summary"""
        submissions = self.get_disease_submissions()
        remedies = self.get_community_remedies()
        analytics = self.load_json(self.analytics_file)
        
        # Calculate summary metrics
        total_submissions = len(submissions)
        total_remedies = len(remedies)
        active_cases = len([s for s in submissions if s.get("status") == "Open"])
        critical_cases = len([s for s in submissions if s.get("severity") == "Critical"])
        
        # Recent activity (last 7 days)
        recent_date = (datetime.now() - timedelta(days=7)).isoformat()
        recent_submissions = len([s for s in submissions if s.get("timestamp", "") > recent_date])
        
        # Top diseases
        disease_counts = {}
        for submission in submissions:
            disease = submission.get("disease_name", "Unknown")
            disease_counts[disease] = disease_counts.get(disease, 0) + 1
        
        top_diseases = sorted(disease_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Location distribution
        location_counts = {}
        for submission in submissions:
            location = submission.get("location", "Unknown")
            location_counts[location] = location_counts.get(location, 0) + 1
        
        return {
            "total_submissions": total_submissions,
            "total_remedies": total_remedies,
            "active_cases": active_cases,
            "critical_cases": critical_cases,
            "recent_submissions_7d": recent_submissions,
            "top_diseases": top_diseases,
            "location_distribution": location_counts,
            "severity_distribution": analytics.get("severity_stats", {}),
            "daily_trends": analytics.get("daily_stats", {}),
            "last_updated": analytics.get("last_updated")
        }
    
    def get_dashboard_data(self) -> Dict:
        """Get formatted data for dashboard visualization"""
        analytics = self.get_analytics_summary()
        submissions = self.get_disease_submissions(limit=20)
        
        # Format for charts
        dashboard_data = {
            "metrics": {
                "total_reports": analytics["total_submissions"],
                "active_cases": analytics["active_cases"],
                "critical_cases": analytics["critical_cases"],
                "community_remedies": analytics["total_remedies"]
            },
            "charts": {
                "severity_pie": analytics["severity_distribution"],
                "top_diseases_bar": dict(analytics["top_diseases"]),
                "location_map": analytics["location_distribution"],
                "daily_trend": analytics["daily_trends"]
            },
            "recent_submissions": submissions[:10],
            "alerts": self.generate_alerts(analytics)
        }
        
        return dashboard_data
    
    def generate_alerts(self, analytics: Dict) -> List[Dict]:
        """Generate alerts based on analytics data"""
        alerts = []
        
        # Critical cases alert
        if analytics["critical_cases"] > 0:
            alerts.append({
                "type": "critical",
                "message": f"{analytics['critical_cases']} critical cases require immediate attention",
                "action": "Contact veterinarian/agricultural expert"
            })
        
        # Disease outbreak alert
        for disease, count in analytics["top_diseases"][:3]:
            if count > 5:  # Threshold for outbreak
                alerts.append({
                    "type": "warning",
                    "message": f"Potential {disease} outbreak detected ({count} cases)",
                    "action": "Implement preventive measures in affected areas"
                })
        
        # High activity alert
        if analytics["recent_submissions_7d"] > 20:
            alerts.append({
                "type": "info",
                "message": f"High activity: {analytics['recent_submissions_7d']} reports in last 7 days",
                "action": "Monitor trends and prepare resources"
            })
        
        return alerts
    
    # Backup and Maintenance
    def backup_database(self) -> str:
        """Create backup of entire database"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"backup_{timestamp}"
        
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
        
        # Copy all JSON files
        import shutil
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.json'):
                shutil.copy2(
                    os.path.join(self.data_dir, filename),
                    os.path.join(backup_dir, filename)
                )
        
        return backup_dir
    
    def cleanup_old_data(self, days_old: int = 90):
        """Clean up old submissions (optional maintenance)"""
        cutoff_date = (datetime.now() - timedelta(days=days_old)).isoformat()
        
        # Clean submissions
        data = self.load_json(self.submissions_file)
        original_count = len(data["submissions"])
        data["submissions"] = [s for s in data["submissions"] 
                             if s.get("timestamp", "") > cutoff_date]
        
        if len(data["submissions"]) < original_count:
            data["total_count"] = len(data["submissions"])
            data["last_updated"] = datetime.now().isoformat()
            self.save_json(self.submissions_file, data)
            
            return original_count - len(data["submissions"])
        
        return 0

# Utility functions
def export_to_csv(database: FarmersDatabase, output_dir: str = "exports"):
    """Export database data to CSV files"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Export submissions
    submissions = database.get_disease_submissions()
    if submissions:
        df_submissions = pd.DataFrame(submissions)
        df_submissions.to_csv(os.path.join(output_dir, "disease_submissions.csv"), index=False)
    
    # Export remedies
    remedies = database.get_community_remedies()
    if remedies:
        df_remedies = pd.DataFrame(remedies)
        df_remedies.to_csv(os.path.join(output_dir, "community_remedies.csv"), index=False)

def import_sample_data(database: FarmersDatabase):
    """Import sample data for demo purposes"""
    sample_submissions = [
        {
            "farmer_name": "Ravi Kumar",
            "location": "Coimbatore, Tamil Nadu",
            "disease_name": "Tomato Late Blight",
            "category": "plant",
            "description": "Dark spots on leaves with rapid spreading",
            "severity": "High"
        },
        {
            "farmer_name": "Priya Sharma",
            "location": "Nashik, Maharashtra", 
            "disease_name": "Wheat Rust",
            "category": "plant",
            "description": "Orange pustules on wheat leaves",
            "severity": "Medium"
        },
        {
            "farmer_name": "Suresh Patel",
            "location": "Anand, Gujarat",
            "disease_name": "Lumpy Skin Disease",
            "category": "livestock",
            "description": "Skin nodules on cattle with fever",
            "severity": "Critical"
        }
    ]
    
    for submission in sample_submissions:
        database.add_disease_submission(**submission)
    
    # Add sample remedies
    sample_remedies = [
        {
            "farmer_name": "Lakshmi Devi",
            "disease_name": "Tomato Late Blight",
            "remedy_description": "Mix neem oil with garlic water and spray early morning",
            "effectiveness_rating": 4,
            "category": "plant"
        },
        {
            "farmer_name": "Mohan Singh",
            "disease_name": "Wheat Rust",
            "remedy_description": "Apply turmeric powder mixed with mustard oil on affected areas",
            "effectiveness_rating": 3,
            "category": "plant"
        }
    ]
    
    for remedy in sample_remedies:
        database.add_community_remedy(**remedy)

# Example usage
if __name__ == "__main__":
    db = FarmersDatabase()
    
    # Import sample data
    import_sample_data(db)
    
    # Test analytics
    analytics = db.get_analytics_summary()
    print(f"Total submissions: {analytics['total_submissions']}")
    print(f"Active cases: {analytics['active_cases']}")
    print(f"Top diseases: {analytics['top_diseases']}")
    
    # Test dashboard data
    dashboard = db.get_dashboard_data()
    print(f"Dashboard alerts: {len(dashboard['alerts'])}")
    
    print("Database initialized successfully!")