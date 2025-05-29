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

@dataclass
class YouTubeProfile:
    name: str = ""
    channel_url: str = ""
    subscribers: Optional[int] = None
    videos: Optional[int] = None
    views: Optional[int] = None
    subscribers_raw: str = ""
    videos_raw: str = ""
    views_raw: str = ""
    email: str = ""
    description: str = ""
    raw_snippet: str = ""
    niche_keywords: List[str] = None

class InstagramDataExtractor:
    def __init__(self):
        self.email_patterns = [
            r'\b[A-Za-z0-9._%+-]+@(?:gmail|yahoo|outlook|hotmail|live|msn|aol|icloud|protonmail|zoho)\.com\b',
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        ]
        self.stats_patterns = [
            r'(\d+(?:\.\d+)?[KMB]?)\s*(?:Followers?),?\s*(\d+(?:\.\d+)?[KMB]?)\s*(?:Following?),?\s*(\d+(?:\.\d+)?[KMB]?)\s*(?:Posts?)',
            r'(?:Followers?:?\s*)(\d+(?:\.\d+)?[KMB]?)\s*(?:Following?:?\s*)(\d+(?:\.\d+)?[KMB]?)\s*(?:Posts?:?\s*)(\d+(?:\.\d+)?[KMB]?)',
            r'(\d+(?:\.\d+)?[KMB]?)\s*followers?\s*[•·]\s*(\d+(?:\.\d+)?[KMB]?)\s*following?\s*[•·]\s*(\d+(?:\.\d+)?[KMB]?)\s*posts?',
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
        emails = []
        for pattern in self.email_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            emails.extend(matches)
        return list(set(emails))

    def convert_count_to_number(self, count_str: str) -> Optional[int]:
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
        text = text.replace(',', ' ').replace('\n', ' ')
        for pattern in self.stats_patterns:
            matches = re.search(pattern, text, re.IGNORECASE)
            if matches:
                groups = matches.groups()
                if len(groups) >= 3:
                    return groups[0], groups[1], groups[2]
        followers = re.search(r'(\d+(?:\.\d+)?[KMB]?)\s*(?:followers?)', text, re.IGNORECASE)
        following = re.search(r'(\d+(?:\.\d+)?[KMB]?)\s*(?:following?)', text, re.IGNORECASE)
        posts = re.search(r'(\d+(?:\.\d+)?[KMB]?)\s*(?:posts?)', text, re.IGNORECASE)
        return (
            followers.group(1) if followers else None,
            following.group(1) if following else None,
            posts.group(1) if posts else None
        )

    def extract_username_and_name(self, title: str) -> Tuple[str, str]:
        match = re.match(r'^(.*?)\s*\(@([a-zA-Z0-9_.]+)\)', title)
        if match:
            name = match.group(1).strip()
            username = match.group(2).strip()
            return name, username
        return "", ""

    def clean_description(self, snippet: str, email: str = "", stats_removed: bool = True) -> str:
        description = snippet
        if email:
            description = description.replace(email, "").strip()
        artifacts = [
            r"'s profile picture\.",
            r"profile picture\.",
            r"Instagram photos and videos",
            r"Instagram photos and \.\.\.",
            r"\u00b7",
            r"[•·]",
            r"\s+"
        ]
        for artifact in artifacts:
            description = re.sub(artifact, " ", description)
        description = re.sub(r'\s+', ' ', description).strip()
        return description

    def detect_niche(self, text: str) -> List[str]:
        detected_niches = []
        text_lower = text.lower()
        for niche, keywords in self.niche_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_niches.append(niche)
        return detected_niches

    def parse_instagram_result(self, item: Dict) -> InstagramProfile:
        profile = InstagramProfile()
        try:
            title = item.get('title', '')
            snippet = item.get('snippet', '')
            og_desc = item.get('og:description', '')
            profile.raw_snippet = snippet
            profile.instagram_url = item.get('link', '')
            name, username = self.extract_username_and_name(title)
            profile.name = name
            profile.username = username
            emails = self.extract_emails(snippet)
            profile.email = emails[0] if emails else ""
            stats_source = og_desc if og_desc else snippet
            followers_raw, following_raw, posts_raw = self.extract_stats(stats_source)
            profile.followers_raw = followers_raw or ""
            profile.following_raw = following_raw or ""
            profile.posts_raw = posts_raw or ""
            profile.followers = self.convert_count_to_number(followers_raw)
            profile.following = self.convert_count_to_number(following_raw)
            profile.posts = self.convert_count_to_number(posts_raw)
            profile.description = self.clean_description(snippet, profile.email)
            profile.niche_keywords = self.detect_niche(snippet + " " + title)
        except Exception as e:
            logger.error(f"Error parsing Instagram result: {str(e)}")
        return profile

    def parse_instagram_results(self, results: List[Dict]) -> List[InstagramProfile]:
        return [self.parse_instagram_result(item) for item in results]

    def to_dataframe(self, profiles: List[InstagramProfile]) -> pd.DataFrame:
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

class YouTubeDataExtractor:
    def __init__(self):
        self.email_patterns = [
            r'\b[A-Za-z0-9._%+-]+@(?:gmail|yahoo|outlook|hotmail|live|msn|aol|icloud|protonmail|zoho)\.com\b',
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        ]
        self.stats_patterns = [
            r'(\d+(?:\.\d+)?[KMB]?)\s*(?:subscribers?)',
            r'(\d+(?:\.\d+)?[KMB]?)\s*(?:videos?)',
            r'(\d+(?:\.\d+)?[KMB]?)\s*(?:views?)'
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
        emails = []
        for pattern in self.email_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            emails.extend(matches)
        return list(set(emails))

    def convert_count_to_number(self, count_str: str) -> Optional[int]:
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
        text = text.replace(',', ' ').replace('\n', ' ')
        subscribers = re.search(r'(\d+(?:\.\d+)?[KMB]?)\s*(?:subscribers?)', text, re.IGNORECASE)
        videos = re.search(r'(\d+(?:\.\d+)?[KMB]?)\s*(?:videos?)', text, re.IGNORECASE)
        views = re.search(r'(\d+(?:\.\d+)?[KMB]?)\s*(?:views?)', text, re.IGNORECASE)
        return (
            subscribers.group(1) if subscribers else None,
            videos.group(1) if videos else None,
            views.group(1) if views else None
        )

    def clean_description(self, snippet: str, email: str = "") -> str:
        description = snippet
        if email:
            description = description.replace(email, "").strip()
        artifacts = [
            r"YouTube channel\.",
            r"channel\.",
            r"\u00b7",
            r"[•·]",
            r"\s+"
        ]
        for artifact in artifacts:
            description = re.sub(artifact, " ", description)
        description = re.sub(r'\s+', ' ', description).strip()
        return description

    def detect_niche(self, text: str) -> List[str]:
        detected_niches = []
        text_lower = text.lower()
        for niche, keywords in self.niche_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_niches.append(niche)
        return detected_niches

    def parse_youtube_result(self, item: Dict) -> YouTubeProfile:
        profile = YouTubeProfile()
        try:
            title = item.get('title', '')
            snippet = item.get('snippet', '')
            profile.raw_snippet = snippet
            profile.channel_url = item.get('link', '')
            
            # Get name from person field if available, fallback to title
            person_data = item.get('pagemap', {}).get('person', [{}])[0]
            profile.name = person_data.get('name', title)
            
            emails = self.extract_emails(snippet)
            profile.email = emails[0] if emails else ""
            
            subscribers_raw, videos_raw, views_raw = self.extract_stats(snippet)
            profile.subscribers_raw = subscribers_raw or ""
            profile.videos_raw = videos_raw or ""
            profile.views_raw = views_raw or ""
            
            profile.subscribers = self.convert_count_to_number(subscribers_raw)
            profile.videos = self.convert_count_to_number(videos_raw)
            profile.views = self.convert_count_to_number(views_raw)
            
            profile.description = self.clean_description(snippet, profile.email)
            profile.niche_keywords = self.detect_niche(snippet + " " + title)
        except Exception as e:
            logger.error(f"Error parsing YouTube result: {str(e)}")
        return profile

    def to_dataframe(self, profiles: List[YouTubeProfile]) -> pd.DataFrame:
        data = []
        for profile in profiles:
            if isinstance(profile, YouTubeProfile):
                row = {
                    'Name': profile.name,
                    'Channel URL': profile.channel_url,
                    'Email': profile.email,
                    'Subscribers': profile.subscribers,
                    'Videos': profile.videos,
                    'Views': profile.views,
                    'Subscribers (Raw)': profile.subscribers_raw,
                    'Videos (Raw)': profile.videos_raw,
                    'Views (Raw)': profile.views_raw,
                    'Description': profile.description,
                    'Niche Keywords': ', '.join(profile.niche_keywords) if profile.niche_keywords else '',
                    'Raw Snippet': profile.raw_snippet
                }
            else:  # InstagramProfile
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
    def __init__(self, api_key: str, search_engine_id: str):
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"

    def search_profiles(self, niche: str, location: str, email_domains: List[str], pages: int, site: str) -> List[Dict]:
        all_results = []
        for domain in email_domains:
            for page in range(pages):
                start = page * 10 + 1
                domain_query = f'@{domain}' if domain else ''
                query = f'{niche} {location} {domain_query} site:{site}'
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
                        time.sleep(0.1)
                    else:
                        logger.warning(f"API request failed with status {response.status_code}")
                except Exception as e:
                    logger.error(f"Search error: {str(e)}")
        unique_results = []
        seen_urls = set()
        for result in all_results:
            url = result.get('link', '')
            if url not in seen_urls:
                unique_results.append(result)
                seen_urls.add(url)
        return unique_results

def validate_dcs_key(input_key: str) -> bool:
    return input_key.upper() == "DCS"

def create_env_file():
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
    filtered_df = df[selected_columns] if selected_columns else df
    if file_format == 'csv':
        return filtered_df.to_csv(index=False).encode('utf-8')
    elif file_format == 'json':
        return filtered_df.to_json(orient='records', indent=2).encode('utf-8')
    elif file_format == 'excel':
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            filtered_df.to_excel(writer, sheet_name='Profiles', index=False)
        return buffer.getvalue()

def main():
    st.set_page_config(page_title="Profile Extractor", page_icon="🔍", layout="wide")
    st.title("🔍 DCS Social Profile Search Tool")
    if create_env_file():
        st.info("📄 Created sample .env file. Please update it with your API credentials.")

    api_key = os.getenv('GOOGLE_API_KEY', '')
    search_engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID', '')

    with st.sidebar:
        st.header("🔐 Authentication")
        dcs_key = st.text_input("Enter Access Code", type="password", placeholder="Code")
        dcs_valid = validate_dcs_key(dcs_key) if dcs_key else False

        if dcs_key and not dcs_valid:
            st.error("❌ Invalid access code")
        elif dcs_valid:
            st.success("✅ Access granted")

        st.header("🎯 Search Parameters")
        if dcs_valid:
            niche = st.text_input("Niche/Industry", placeholder="e.g., beauty, tech, travel")
            location = st.text_input("Location", placeholder="e.g., Delhi, Mumbai, India")
            email_domains = st.multiselect(
                "Email Domains",
                options=['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'live.com'],
                default=['gmail.com', 'yahoo.com', 'outlook.com']
            )
            pages = st.slider("Number of Pages", min_value=1, max_value=10, value=1)
            st.markdown("**Platforms to Search:**")
            search_instagram = st.checkbox('Search Instagram', value=True)
            search_youtube = st.checkbox('Search YouTube', value=False)
            search_button = st.button("🚀 Start Search", type="primary", disabled=not dcs_valid)
        else:
            st.info("Enter valid access code to unlock search parameters")
            search_button = False
            niche = location = ""
            email_domains = []
            pages = 1
            search_instagram = search_youtube = False

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
        if not (search_instagram or search_youtube):
            st.error("Please select at least one platform to search")
            return

        platforms = []
        if search_instagram:
            platforms.append("instagram.com")
        if search_youtube:
            platforms.append("youtube.com")

        search_api = GoogleSearchAPI(api_key, search_engine_id)
        extractor = InstagramDataExtractor()
        all_profiles = []
        all_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_platforms = len(platforms)
        try:
            for i, site in enumerate(platforms):
                status_text.text(f"🔍 Searching {site.replace('.com', '').capitalize()} profiles...")
                results = search_api.search_profiles(
                    niche=niche,
                    location=location,
                    email_domains=email_domains,
                    pages=pages,
                    site=site
                )
                all_results.extend(results)
                progress_bar.progress(int((i+1) / total_platforms * 50))
            if not all_results:
                st.warning("No results found. Try different search parameters.")
                return
            status_text.text("📊 Parsing profile data...")
            if search_instagram:
                ig_extractor = InstagramDataExtractor()
                ig_results = [r for r in all_results if "instagram.com" in r.get('link', '')]
                instagram_profiles = ig_extractor.parse_instagram_results(ig_results)
                df_instagram = ig_extractor.to_dataframe(instagram_profiles)
                all_profiles.extend(instagram_profiles)
            if search_youtube:
                yt_extractor = YouTubeDataExtractor()
                yt_results = [r for r in all_results if "youtube.com" in r.get('link', '')]
                youtube_profiles = [yt_extractor.parse_youtube_result(item) for item in yt_results]
                df_youtube = yt_extractor.to_dataframe(youtube_profiles)
                all_profiles.extend(youtube_profiles)
            df = pd.concat([df_instagram, df_youtube]) if 'df_instagram' in locals() and 'df_youtube' in locals() else \
                df_instagram if 'df_instagram' in locals() else df_youtube
            status_text.text("📋 Preparing data table...")
            progress_bar.progress(100)
            st.session_state.df = df
            st.session_state.profiles = all_profiles
            status_text.text("✅ Data extraction completed!")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"Application error: {str(e)}")
            return

    if 'df' in st.session_state:
        df = st.session_state.df
        profiles = st.session_state.profiles
        st.success(f"Found {len(profiles)} profiles!")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Profiles", len(profiles))
        with col2:
            profiles_with_email = len([p for p in profiles if p.email])
            st.metric("Profiles with Email", profiles_with_email)
        with col3:
            profiles_with_stats = len([p for p in profiles if (
                (isinstance(p, InstagramProfile) and p.followers is not None) or 
                (isinstance(p, YouTubeProfile) and p.subscribers is not None)
            )])
            st.metric("Profiles with Stats", profiles_with_stats)

        with col4:
            if 'Followers' in df and not df['Followers'].isna().all():
                avg_followers = df['Followers'].dropna().mean()
                st.metric("Avg Followers", f"{avg_followers:,.0f}")
            elif 'Subscribers' in df and not df['Subscribers'].isna().all():
                avg_subscribers = df['Subscribers'].dropna().mean()
                st.metric("Avg Subscribers", f"{avg_subscribers:,.0f}")
            else:
                st.metric("Avg Engagement", "N/A")

        st.subheader("📋 Column Selection")
        col1, col2 = st.columns([2, 1])
        all_columns = list(df.columns)
        if "selected_columns" not in st.session_state:
            st.session_state.selected_columns = all_columns

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

        if selected_columns:
            st.subheader("📊 Extracted Data")
            col1, col2 = st.columns(2)
            with col1:
                email_filter = st.selectbox("Filter by Email Domain",
                    options=["All"] + [f"@{domain}" for domain in ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com']])
            with col2:
                min_followers = st.number_input("Minimum Followers", min_value=0, value=0)
            filtered_df = df[selected_columns].copy()
            if 'Email' in selected_columns and email_filter != "All":
                filtered_df = filtered_df[filtered_df['Email'].str.contains(email_filter.replace("@", ""), na=False)]
            if 'Followers' in selected_columns and min_followers > 0:
                filtered_df = filtered_df[filtered_df['Followers'].fillna(0) >= min_followers]
            st.dataframe(filtered_df, use_container_width=True)
            st.subheader("💾 Download Selected Data")
            if selected_columns:
                col1, col2, col3 = st.columns(3)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                with col1:
                    csv_data = get_downloadable_data(filtered_df, selected_columns, 'csv')
                    st.download_button(
                        label="📄 Download CSV",
                        data=csv_data,
                        file_name=f"profiles_{timestamp}.csv",
                        mime="text/csv"
                    )
                with col2:
                    json_data = get_downloadable_data(filtered_df, selected_columns, 'json')
                    st.download_button(
                        label="📋 Download JSON",
                        data=json_data,
                        file_name=f"profiles_{timestamp}.json",
                        mime="application/json"
                    )
                with col3:
                    excel_data = get_downloadable_data(filtered_df, selected_columns, 'excel')
                    st.download_button(
                        label="📊 Download Excel",
                        data=excel_data,
                        file_name=f"profiles_{timestamp}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                st.info(f"Selected {len(selected_columns)} out of {len(all_columns)} columns for download")
            else:
                st.warning("Please select at least one column to download data")
        else:
            st.warning("No columns selected. Please choose columns to display.")
    elif dcs_valid:
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
   - Select platform(s) to search
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
""")

if __name__ == "__main__":
    main()
