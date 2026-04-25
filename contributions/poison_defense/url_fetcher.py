"""
URL Fetcher for HydraDB++ Memory Poison Defense

Fetches and extracts clean text content from web URLs for poison scanning.
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, Comment


class URLFetcher:
    """Fetches and extracts clean text content from web URLs."""
    
    def __init__(self, timeout: int = 10, user_agent: str = "Mozilla/5.0") -> None:
        """Initialize URL fetcher with configuration.
        
        Args:
            timeout: Request timeout in seconds
            user_agent: User agent string for requests
        """
        self.timeout = timeout
        self.user_agent = user_agent
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})
    
    def fetch(self, url: str) -> Dict[str, Any]:
        """
        Fetch content from a real URL
        
        Steps:
        1. Validate URL format first
           - Must start with http:// or https://
           - If invalid → return error, don't fetch
        
        2. Fetch with requests library
           - timeout=10 seconds
           - headers = {"User-Agent": "Mozilla/5.0"}
           - Handle connection errors gracefully
        
        3. Extract clean text from HTML
           - Use BeautifulSoup to parse HTML
           - Remove script tags
           - Remove style tags
           - Remove nav, header, footer tags
           - Extract only main body text
           - Clean up whitespace
        
        4. Return:
           {
             "success": True/False,
             "url": url,
             "content": extracted_text,
             "content_length": len(text),
             "fetch_time": timestamp,
             "error": None or error message
           }
        """
        start_time = time.time()
        fetch_time = datetime.now().isoformat()
        
        # Step 1: Validate URL format
        validation_result = self._validate_url(url)
        if not validation_result["valid"]:
            return {
                "success": False,
                "url": url,
                "content": "",
                "content_length": 0,
                "fetch_time": fetch_time,
                "error": validation_result["error"]
            }
        
        # Step 2: Fetch content
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            # Check if content type is HTML
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                return {
                    "success": False,
                    "url": url,
                    "content": "",
                    "content_length": 0,
                    "fetch_time": fetch_time,
                    "error": f"Unsupported content type: {content_type}. Expected text/html."
                }
            
            html_content = response.text
            
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "url": url,
                "content": "",
                "content_length": 0,
                "fetch_time": fetch_time,
                "error": f"Request timeout after {self.timeout} seconds"
            }
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "url": url,
                "content": "",
                "content_length": 0,
                "fetch_time": fetch_time,
                "error": "Connection error - unable to reach URL"
            }
        except requests.exceptions.HTTPError as e:
            return {
                "success": False,
                "url": url,
                "content": "",
                "content_length": 0,
                "fetch_time": fetch_time,
                "error": f"HTTP error: {e.response.status_code}"
            }
        except Exception as e:
            return {
                "success": False,
                "url": url,
                "content": "",
                "content_length": 0,
                "fetch_time": fetch_time,
                "error": f"Unexpected error: {str(e)}"
            }
        
        # Step 3: Extract clean text from HTML
        try:
            clean_text = self._extract_clean_text(html_content)
            
            # Calculate fetch duration
            fetch_duration = time.time() - start_time
            
            return {
                "success": True,
                "url": url,
                "content": clean_text,
                "content_length": len(clean_text),
                "fetch_time": fetch_time,
                "fetch_duration": round(fetch_duration, 2)
            }
            
        except Exception as e:
            return {
                "success": False,
                "url": url,
                "content": "",
                "content_length": 0,
                "fetch_time": fetch_time,
                "error": f"Text extraction error: {str(e)}"
            }
    
    def _validate_url(self, url: str) -> Dict[str, Any]:
        """Validate URL format and accessibility."""
        
        # Check if URL starts with http:// or https://
        if not url.startswith(('http://', 'https://')):
            return {
                "valid": False,
                "error": "URL must start with http:// or https://"
            }
        
        # Basic URL format validation
        try:
            parsed = urlparse(url)
            if not parsed.netloc:
                return {
                    "valid": False,
                    "error": "Invalid URL format - missing domain"
                }
            
            # Check for obviously malformed URLs
            if '..' in url or '//' in url.replace('://', '/'):
                return {
                    "valid": False,
                    "error": "Potentially malicious URL format detected"
                }
                
        except Exception:
            return {
                "valid": False,
                "error": "Invalid URL format"
            }
        
        return {"valid": True}
    
    def _extract_clean_text(self, html_content: str) -> str:
        """Extract clean text from HTML content.
        
        Steps:
        1. Parse HTML with BeautifulSoup
        2. Remove script tags
        3. Remove style tags
        4. Remove nav, header, footer tags
        5. Remove HTML comments
        6. Extract text from remaining content
        7. Clean up whitespace
        """
        # Parse HTML
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Remove script and style elements
        for element in soup(['script', 'style']):
            element.decompose()
        
        # Remove navigation, header, footer elements
        for element in soup(['nav', 'header', 'footer', 'aside', 'sidebar']):
            element.decompose()
        
        # Remove HTML comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
        
        # Find main content areas
        main_content = None
        
        # Try to find main content area
        for tag in ['main', 'article', 'section', 'div']:
            main_content = soup.find(tag)
            if main_content:
                break
        
        # If no main content found, use body
        if not main_content:
            main_content = soup.find('body') or soup
        
        # Extract text and clean it up
        text = main_content.get_text(separator=' ', strip=True)
        
        # Clean up whitespace
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def fetch_local_file(self, file_path: str) -> Dict[str, Any]:
        """Fetch content from a local file (for file:// URLs).
        
        Args:
            file_path: Path to local file
            
        Returns:
            Dict with same structure as fetch() method
        """
        fetch_time = datetime.now().isoformat()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # If it's an HTML file, extract clean text
            if file_path.lower().endswith(('.html', '.htm')):
                try:
                    clean_text = self._extract_clean_text(content)
                except:
                    clean_text = content
            else:
                clean_text = content
            
            return {
                "success": True,
                "url": f"file://{file_path}",
                "content": clean_text,
                "content_length": len(clean_text),
                "fetch_time": fetch_time,
                "fetch_duration": 0.0
            }
            
        except FileNotFoundError:
            return {
                "success": False,
                "url": f"file://{file_path}",
                "content": "",
                "content_length": 0,
                "fetch_time": fetch_time,
                "error": f"File not found: {file_path}"
            }
        except UnicodeDecodeError:
            return {
                "success": False,
                "url": f"file://{file_path}",
                "content": "",
                "content_length": 0,
                "fetch_time": fetch_time,
                "error": f"File encoding error: {file_path}"
            }
        except Exception as e:
            return {
                "success": False,
                "url": f"file://{file_path}",
                "content": "",
                "content_length": 0,
                "fetch_time": fetch_time,
                "error": f"Error reading file: {str(e)}"
            }
