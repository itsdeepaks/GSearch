import streamlit as st
import requests
import json
import pandas as pd
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass
import time
import os
from dotenv import load_dotenv
from io import BytesIO

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class InstagramProfile:
    """Data class for Instagram profile information"""
    name: str = ""
    username: str = ""
    instagram_url: str = ""
    followers: Optional[int] = None
    following: Optional[int] = None
    posts: Optional[int] = None
    followers_raw: str = ""
    following_raw: str = ""
    posts_raw: str = ""
    email: str = ""
    description: str = ""
    raw_snippet: str = ""
    location: str = ""
    niche_keywords: List[str] = None

class InstagramDataExtractor:
    """Advanced Instagram data extractor with robust parsing capabilities"""
    
    def __init__(self):
        # Enhanced email patterns for various providers
        self.email_patterns = [
            r'\b[A-Za-z0-9._%+-]+@(?:gmail|yahoo|outlook|hotmail|live|msn|aol|icloud|protonmail|zoho)\.com\b',
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        ]
        
        # Multiple patterns for follower/following/posts extraction
        self.stats_patterns = [
            # Pattern: "1.2K Followers, 500 Following, 234 Posts"
            r'(\d+(?:\.\d+)?[KMB]?)\s*(?:Followers?),?\s*(\d+(?:\.\d+)?[KMB]?)\s*(?:Following?),?\s*(\d+(?:\.\d+)?[KMB]?)\s*(?:Posts?)',
            # Pattern: "Followers: 1.2K Following: 500 Posts: 234"
            r'(?:Followers?:?\s*)(\d+(?:\.\d+)?[KMB]?)\s*(?:Following?:?\s*)(\d+(?:\.\d+)?[KMB]?)\s*(?:Posts?:?\s*)(\d+(?:\.\d+)?[KMB]?)',
            # Pattern: "1234 followers • 567 following • 89 posts"
            r'(\d+(?:\.\d+)?[KMB]?)\s*followers?\s*[•·]\s*(\d+(?:\.\d+)?[KMB]?)\s*following?\s*[•·]\s*(\d+(?:\.\d+)?[KMB]?)\s*posts?',
            # Pattern: Individual extraction
            r'(\d+(?:\.\d+)?[KMB]?)\s*(?:Followers?)|(\d+(?:\.\d+)?[KMB]?)\s*(?:Following?)|(\d+(?:\.\d+)?[KMB]?)\s*(?:Posts?)'
        ]
        
        self.niche_keywords = {
            'beauty': ['makeup', 'cosmetics', 'skincare', 'beauty', 'salon', 'spa'],
            'tech': ['technology', 'software', 'coding', 'developer', 'programmer', 'IT'],
            'travel': ['travel', 'wanderlust', 'explore', 'adventure', 'vacation'],
            'food': ['food', 'chef', 'cooking', 'recipe', 'restaurant', 'cuisine'],
            'fitness': ['fitness', 'gym', 'workout', 'health', 'training', 'yoga'],
            'fashion': ['fashion', 'style', 'designer', 'model', 'clothing', 'outfit']
        }

    def extract_emails(self, text: str) -> List[str]:
        """Extract all emails from text using multiple patterns"""
        emails = []
        for pattern in self.email_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            emails.extend(matches)
        
        # Remove duplicates and return unique emails
        return list(set(emails))

    def convert_count_to_number(self, count_str: str) -> Optional[int]:
        """Convert follower count string to number (e.g., '1.2K' -> 1200)"""
        if not count_str:
            return None
            
        count_str = count_str.strip().upper()
        
        try:
            if count_str.endswith('K'):
                return int(float(count_str[:-1]) * 1000)
            elif count_str.endswith('M'):
                return int(float(count_str[:-1]) * 1000000)
            elif count_str.endswith('B'):
                return int(float(count_str[:-1]) * 1000000000)
            else:
                return int(count_str)
        except (ValueError, TypeError):
            return None

    def extract_stats(self, text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Extract followers, following, and posts from text"""
        text = text.replace(',', ' ').replace('\n', ' ')
        
        for pattern in self.stats_patterns:
            matches = re.search(pattern, text, re.IGNORECASE)
            if matches:
                groups = matches.groups()
                if len(groups) >= 3:
                    return groups[0], groups[1], groups[2]
        
        # Individual extraction as fallback
        followers = re.search(r'(\d+(?:\.\d+)?[KMB]?)\s*(?:followers?)', text, re.IGNORECASE)
        following = re.search(r'(\d+(?:\.\d+)?[KMB]?)\s*(?:following?)', text, re.IGNORECASE)
        posts = re.search(r'(\d+(?:\.\d+)?[KMB]?)\s*(?:posts?)', text, re.IGNORECASE)
        
        return (
            followers.group(1) if followers else None,
            following.group(1) if following else None,
            posts.group(1) if posts else None
        )

    def extract_username(self, title: str) -> str:
        """Extract Instagram username from title"""
        match = re.search(r'@([a-zA-Z0-9_.]+)', title)
        return match.group(1) if match else ""

    def extract_name(self, title: str) -> str:
        """Extract name from title (before username)"""
        match = re.search(r'^([^@(]+)', title)
        if match:
            return match.group(1).strip()
        return ""

    def clean_description(self, snippet: str, email: str = "", stats_removed: bool = True) -> str:
        """Clean description by removing email and stats"""
        description = snippet
        
        # Remove email
        if email:
            description = description.replace(email, "").strip()
        
        # Remove common Instagram artifacts
        artifacts = [
            r"'s profile picture\.",
            r"profile picture\.",
            r"Instagram photos and videos",
            r"Instagram photos and \.\.\.",
            r"\u00b7",  # middle dot
            r"[•·]",
            r"\s+", # multiple spaces
        ]
        
        for artifact in artifacts:
            description = re.sub(artifact, " ", description)
        
        # Clean up extra spaces
        description = re.sub(r'\s+', ' ', description).strip()
        
        return description

    def detect_niche(self, text: str) -> List[str]:
        """Detect niche keywords in text"""
        detected_niches = []
        text_lower = text.lower()
        
        for niche, keywords in self.niche_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_niches.append(niche)
        
        return detected_niches

    def extract_username_and_name(self, title: str) -> Tuple[str, str]:
        """
        Extracts name and username from a title like 'Praveen Rana (@iampraveenrana)'
        Returns (name, username)
        """
        match = re.match(r'^(.*?)\s*\(@([a-zA-Z0-9_.]+)\)', title)
        if match:
            name = match.group(1).strip()
            username = match.group(2).strip()
            return name, username
        return "", ""

    def extract_stats_from_text(self, text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Extract followers, following, and posts from text"""
        # ...existing code for stats extraction...
        # (keep your current implementation)
        return super().extract_stats(text)

    def parse_instagram_result(self, item: Dict) -> InstagramProfile:
        """Parse a single Instagram search result"""
        profile = InstagramProfile()
        try:
            # Prefer og:description for stats, fallback to snippet
            title = item.get('title', '')
            snippet = item.get('snippet', '')
            og_desc = item.get('og:description', '')

            profile.raw_snippet = snippet
            profile.instagram_url = item.get('link', '')

            # Extract name and username from title
            name, username = self.extract_username_and_name(title)
            profile.name = name
            profile.username = username

            # Extract emails
            emails = self.extract_emails(snippet)
            profile.email = emails[0] if emails else ""

            # Extract stats from og:description or snippet
            stats_source = og_desc if og_desc else snippet
            followers_raw, following_raw, posts_raw = self.extract_stats(stats_source)
            profile.followers_raw = followers_raw or ""
            profile.following_raw = following_raw or ""
            profile.posts_raw = posts_raw or ""

            # Convert to numbers
            profile.followers = self.convert_count_to_number(followers_raw)
            profile.following = self.convert_count_to_number(following_raw)
            profile.posts = self.convert_count_to_number(posts_raw)

            # Clean description
            profile.description = self.clean_description(snippet, profile.email)

            # Detect niche
            profile.niche_keywords = self.detect_niche(snippet + " " + title)

        except Exception as e:
            logger.error(f"Error parsing Instagram result: {str(e)}")

        return profile

    def parse_instagram_results(self, results: List[Dict]) -> List[InstagramProfile]:
        """Parse multiple Instagram search results"""
        profiles = []
        
        for item in results:
            profile = self.parse_instagram_result(item)
            profiles.append(profile)
            
        return profiles

    def to_dataframe(self, profiles: List[InstagramProfile]) -> pd.DataFrame:
        """Convert profiles to pandas DataFrame"""
        data = []
        
        for profile in profiles:
            row = {
                'Name': profile.name,
                'Username': profile.username,
                'Instagram URL': profile.instagram_url,
                'Email': profile.email,
                'Followers': profile.followers,
                'Following': profile.following,
                'Posts': profile.posts,
                'Followers (Raw)': profile.followers_raw,
                'Following (Raw)': profile.following_raw,
                'Posts (Raw)': profile.posts_raw,
                'Description': profile.description,
                'Niche Keywords': ', '.join(profile.niche_keywords) if profile.niche_keywords else '',
                'Raw Snippet': profile.raw_snippet
            }
            data.append(row)
            
        return pd.DataFrame(data)

class GoogleSearchAPI:
    """Google Custom Search API wrapper"""
    
    def __init__(self, api_key: str, search_engine_id: str):
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"

    def search_instagram_profiles(self, niche: str, location: str, 
                                email_domains: List[str] = None, 
                                pages: int = 1) -> List[Dict]:
        """Search for Instagram profiles"""
        if email_domains is None:
            email_domains = ['gmail.com', 'yahoo.com', 'outlook.com']
        
        all_results = []
        
        for domain in email_domains:
            for page in range(pages):
                start = page * 10 + 1
                
                # Construct search query
                domain_query = f'@{domain}' if domain else ''
                query = f'{niche} {location} {domain_query} site:instagram.com'
                
                params = {
                    'key': self.api_key,
                    'cx': self.search_engine_id,
                    'q': query,
                    'num': 10,
                    'start': start
                }
                
                try:
                    response = requests.get(self.base_url, params=params)
                    if response.status_code == 200:
                        search_results = response.json()
                        items = search_results.get('items', [])
                        all_results.extend(items)
                        
                        # Add delay to respect rate limits
                        time.sleep(0.1)
                        
                    else:
                        logger.warning(f"API request failed with status {response.status_code}")
                        
                except Exception as e:
                    logger.error(f"Search error: {str(e)}")
        
        # Remove duplicates based on URL
        unique_results = []
        seen_urls = set()
        
        for result in all_results:
            url = result.get('link', '')
            if url not in seen_urls:
                unique_results.append(result)
                seen_urls.add(url)
        
        return unique_results

def validate_dcs_key(input_key: str) -> bool:
    """Validate DCS access key"""
    return input_key.upper() == "DCS"

def create_env_file():
    """Create a sample .env file if it doesn't exist"""
    env_content = """# Google API Configuration
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id_here

# DCS Configuration
DCS_ACCESS_KEY=DCS
"""
    
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write(env_content)
        return True
    return False

def get_downloadable_data(df: pd.DataFrame, selected_columns: List[str], file_format: str) -> bytes:
    """Generate downloadable data in specified format"""
    # Filter DataFrame to selected columns
    filtered_df = df[selected_columns] if selected_columns else df
    
    if file_format == 'csv':
        return filtered_df.to_csv(index=False).encode('utf-8')
    elif file_format == 'json':
        return filtered_df.to_json(orient='records', indent=2).encode('utf-8')
    elif file_format == 'excel':
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            filtered_df.to_excel(writer, sheet_name='Instagram Profiles', index=False)
        return buffer.getvalue()

# Streamlit App
def main():
    st.set_page_config(
        page_title="Instagram Profile Extractor",
        page_icon="📸",
        layout="wide"
    )
    
    st.title("🔍 DCS-Instagram Influencer Search Tool")
    st.markdown("Extract detailed information from Instagram profiles including follower counts, emails, and more!")
    
    # Create .env file if it doesn't exist
    if create_env_file():
        st.info("📄 Created sample .env file. Please update it with your API credentials.")
    
    # Load API credentials from environment
    api_key_from_env = os.getenv('GOOGLE_API_KEY', '')
    search_engine_id_from_env = os.getenv('GOOGLE_SEARCH_ENGINE_ID', '')
    
    # Fix: assign to variables used later
    api_key = api_key_from_env
    search_engine_id = search_engine_id_from_env
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🔐 Authentication")
        
        # DCS Validation Key
        dcs_key = st.text_input("Enter Access Code", type="password", placeholder="Code")
        dcs_valid = validate_dcs_key(dcs_key) if dcs_key else False
        
        if dcs_key and not dcs_valid:
            st.error("❌ Invalid access code")
        elif dcs_valid:
            st.success("✅ Access granted")
        
        st.header("⚙️ 404 Not Found… Just Kidding!")
        
        
        st.header("🎯 Search Parameters")
        
        # Only show search parameters if DCS is valid
        if dcs_valid:
            niche = st.text_input("Niche/Industry", placeholder="e.g., beauty, tech, travel")
            location = st.text_input("Location", placeholder="e.g., Delhi, Mumbai, India")
            
            email_domains = st.multiselect(
                "Email Domains",
                options=['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'live.com'],
                default=['gmail.com', 'yahoo.com', 'outlook.com']
            )
            
            pages = st.slider("Number of Pages", min_value=1, max_value=10, value=1)
            
            search_button = st.button("🚀 Start Search", type="primary", disabled=not dcs_valid)
        else:
            st.info("Enter valid access code to unlock search parameters")
            search_button = False
            niche = location = ""
            email_domains = []
            pages = 1
    
    # Main content area
    if not dcs_valid:
        st.warning("🔒 Please enter the access code in the sidebar to continue")
        
        with st.expander("📖 Setup Instructions"):
            st.markdown("""
            ### 🔧 Setup Steps:
            
            1. **Get Access Code:**
               - Contact administrator for the access code
            
            2. **Follow all the steps above:**
               
            """)
        return
    
    if search_button:
        if not api_key or not search_engine_id:
            st.error("Please provide Google API Key and Search Engine ID")
            return
        
        if not niche or not location:
            st.error("Please provide niche and location")
            return
        
        # Initialize components
        search_api = GoogleSearchAPI(api_key, search_engine_id)
        extractor = InstagramDataExtractor()
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Search for profiles
            status_text.text("🔍 Searching Instagram profiles...")
            progress_bar.progress(25)
            
            search_results = search_api.search_instagram_profiles(
                niche=niche,
                location=location,
                email_domains=email_domains,
                pages=pages
            )
            
            if not search_results:
                st.warning("No results found. Try different search parameters.")
                return
            
            # Parse results
            status_text.text("📊 Parsing profile data...")
            progress_bar.progress(50)
            
            profiles = extractor.parse_instagram_results(search_results)
            
            # Convert to DataFrame
            status_text.text("📋 Preparing data table...")
            progress_bar.progress(75)
            
            df = extractor.to_dataframe(profiles)
            
            progress_bar.progress(100)
            status_text.text("✅ Data extraction completed!")
            
            # Store DataFrame in session state
            st.session_state.df = df
            st.session_state.profiles = profiles
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"Application error: {str(e)}")
            return
    
    # Display results if data exists in session state
    if 'df' in st.session_state:
        df = st.session_state.df
        profiles = st.session_state.profiles
        
        # Display results
        st.success(f"Found {len(profiles)} Instagram profiles!")
        
        # Show summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Profiles", len(profiles))
        
        with col2:
            profiles_with_email = len([p for p in profiles if p.email])
            st.metric("Profiles with Email", profiles_with_email)
        
        with col3:
            profiles_with_stats = len([p for p in profiles if p.followers is not None])
            st.metric("Profiles with Stats", profiles_with_stats)
        
        with col4:
            avg_followers = df['Followers'].dropna().mean() if not df['Followers'].isna().all() else 0
            st.metric("Avg Followers", f"{avg_followers:,.0f}" if avg_followers > 0 else "N/A")
        
        # Column Selection Section
        st.subheader("📋 Column Selection")

        col1, col2 = st.columns([2, 1])

        all_columns = list(df.columns)
        # Set default if not in session_state
        if "selected_columns" not in st.session_state:
            st.session_state.selected_columns = all_columns

        # --- Move Quick Selection Buttons ABOVE the multiselect ---
        with col2:
            st.markdown("**Quick Selections:**")

            def set_essential():
                st.session_state.selected_columns = ['Name', 'Username', 'Instagram URL', 'Email', 'Followers', 'Description']

            def set_with_stats():
                st.session_state.selected_columns = ['Name', 'Username', 'Email', 'Followers', 'Following', 'Posts', 'Instagram URL']

            def set_all():
                st.session_state.selected_columns = all_columns

            def clear_all():
                st.session_state.selected_columns = []

            st.button("📧 Essential Only", on_click=set_essential)
            st.button("📊 With Stats", on_click=set_with_stats)
            st.button("🔄 Select All", on_click=set_all)
            st.button("❌ Clear All", on_click=clear_all)

        with col1:
            selected_columns = st.multiselect(
                "Select columns to display and download:",
                options=all_columns,
                default=st.session_state.selected_columns,
                help="Choose which columns to include in the table and downloads",
                key="selected_columns"
            )
    
        # Display data table with selected columns
        if selected_columns:
            st.subheader("📊 Extracted Data")
            
            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                email_filter = st.selectbox("Filter by Email Domain", 
                                          options=["All"] + [f"@{domain}" for domain in ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com']])
            
            with col2:
                min_followers = st.number_input("Minimum Followers", min_value=0, value=0)
            
            # Apply filters
            filtered_df = df[selected_columns].copy()
            
            if 'Email' in selected_columns and email_filter != "All":
                filtered_df = filtered_df[filtered_df['Email'].str.contains(email_filter.replace("@", ""), na=False)]
            
            if 'Followers' in selected_columns and min_followers > 0:
                filtered_df = filtered_df[filtered_df['Followers'].fillna(0) >= min_followers]
            
            # Display filtered data
            st.dataframe(filtered_df, use_container_width=True)
            
            # Download options
            st.subheader("💾 Download Selected Data")
            
            if selected_columns:
                col1, col2, col3 = st.columns(3)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                with col1:
                    csv_data = get_downloadable_data(filtered_df, selected_columns, 'csv')
                    st.download_button(
                        label="📄 Download CSV",
                        data=csv_data,
                        file_name=f"instagram_profiles_{timestamp}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    json_data = get_downloadable_data(filtered_df, selected_columns, 'json')
                    st.download_button(
                        label="📋 Download JSON",
                        data=json_data,
                        file_name=f"instagram_profiles_{timestamp}.json",
                        mime="application/json"
                    )
                
                with col3:
                    excel_data = get_downloadable_data(filtered_df, selected_columns, 'excel')
                    st.download_button(
                        label="📊 Download Excel",
                        data=excel_data,
                        file_name=f"instagram_profiles_{timestamp}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                # Display selected columns info
                st.info(f"Selected {len(selected_columns)} out of {len(all_columns)} columns for download")
            else:
                st.warning("Please select at least one column to download data")
        else:
            st.warning("No columns selected. Please choose columns to display.")
    
    elif dcs_valid:
        # Show instructions when no search is performed but access is granted
        st.info("👈 Configure your search parameters in the sidebar and click 'Start Search' to begin!")
        
        with st.expander("📖 How to use this tool"):
            st.markdown("""
            ### 🚀 Quick Start Guide:
            
            1. **✅ Access Granted** - You've entered the correct access code
            
            2. **Configure search parameters:**
               - Enter your niche/industry (e.g., beauty, tech, travel)
               - Specify location (e.g., Delhi, Mumbai, India)
               - Select email domains to search for
               - Choose number of pages to search
            
            3. **Start extraction:**
               - Click "Start Search" to begin
               - Wait for results to be processed
               - Select columns you want to keep
               - Review and download the extracted data
            
            ### 🎯 Features:
            - 📧 **Multi-domain email extraction**: Gmail, Yahoo, Outlook, etc.
            - 📊 **Smart stats parsing**: Followers, Following, Posts with K/M conversion
            - 🎨 **Column selection**: Choose exactly what data to export
            - 💾 **Multiple formats**: CSV, JSON, Excel downloads
            - 🔍 **Advanced filtering**: Email domain, follower count filters
            - 🛡️ **Secure access**: DCS validation and .env credentials
            
            ### 📋 Column Options:
            - **Essential**: Name, Username, URL, Email, Followers, Description
            - **With Stats**: Includes Following and Posts counts
            - **Complete**: All available data including raw snippets
            - **Custom**: Select any combination you need
            """)

if __name__ == "__main__":
    main()
