import os

from crewai import Agent, Crew, LLM, Task
from logging_config import get_logger
import requests
from bs4 import BeautifulSoup

def fetch_url_content(url):
    """
    Fetch and parse content from a URL, preserving links in markdown format.
    
    Args:
        url: The URL to fetch content from
        
    Returns:
        str: The cleaned text content from the URL with links in markdown format, or an error message
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Replace links with markdown-style links to preserve them
        for link in soup.find_all('a', href=True):
            href = link.get('href', '')
            text = link.get_text(strip=True)
            # Only convert links that have both text and href, and aren't anchors or javascript
            if href and text and not href.startswith('#') and not href.startswith('javascript:'):
                # Replace the link tag with markdown syntax
                link.replace_with(f"[{text}]({href})")
        
        # Get text content
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        return f"Error fetching URL: {str(e)}"


class ResearchCrew:
    def __init__(self, verbose=True, logger=None):
        self.verbose = verbose
        self.logger = logger or get_logger(__name__)
        self.crew = None
        self.logger.info("ResearchCrew initialized")

    def create_crew(self, url):
        """
        Create a crew to transform URL content to markdown.
        
        Args:
            url: The URL to fetch and transform
            
        Returns:
            Crew: A configured CrewAI crew with agents and tasks
        """
        self.logger.info(f"Creating URL to markdown crew for: {url}")
        
        # Fetch content from URL
        url_content = fetch_url_content(url)

        llm = LLM(
            model="gemini/gemini-2.5-flash",
            api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.1,
        )
        # Create Content Scraper Agent
        content_scraper = Agent(
            role='Content Scraper',
            goal=f'Extract exact details from the web content at {url} without analysis',
            backstory='You are an expert at scraping web content and extracting exact details '
                      'from web pages, including all titles, descriptions, and other information '
                      'exactly as they appear on the page.',
            verbose=self.verbose,
            allow_delegation=False,
            llm=llm
        )
        
        # Create Markdown Formatter Agent
        markdown_formatter = Agent(
            role='Markdown Documentation Specialist',
            goal='Transform scraped content into well-structured markdown format without analysis or interpretation',
            backstory='You are a skilled technical writer who excels at creating clean, '
                      'well-organized markdown documentation that preserves exact details from the source '
                      'without adding interpretation or analysis.',
            verbose=self.verbose,
            allow_delegation=False,
            llm=llm
        )
        
        self.logger.info("Created content scraper and markdown formatter agents")
        
        # Task to scrape exact content
        scrape_task = Task(
            description=f"""
            Scrape the following web content from {url} and extract ALL details exactly as they appear:
            
            Content:
            {url_content[:8000]}  
            
            Extract the exact details including:
            - Main title/heading (exact text)
            - Complete description (verbatim)
            - All key details exactly as shown
            - All sections and subsections with their exact content
            - ALL links and references in markdown format [text](url) - preserve these exactly
            - Any other information present on the page
            
            DO NOT analyze, interpret, or summarize. Provide the exact details as they appear on the page.
            IMPORTANT: Preserve all markdown-formatted links [text](url) exactly as they appear.
            """,
            agent=content_scraper,
            expected_output='A complete extraction of all exact details from the webpage without analysis, including all links in markdown format'
        )
        
        # Task to convert to markdown
        markdown_task = Task(
            description="""
            Take the scraped content details and transform them into well-formatted markdown.
            
            The markdown should:
            - Preserve ALL exact details from the scraped content without interpretation
            - Have a clear hierarchical structure with appropriate headers (# ## ###)
            - Use bullet points and lists where appropriate
            - Include all markdown-formatted links [text](url) exactly as they appear in the scraped content
            - Be well-organized and easy to read
            - Use markdown formatting features like bold, italic, code blocks, etc. where appropriate
            - NOT add any analysis, commentary, or interpretation
            
            CRITICAL: Preserve all links in markdown format [text](url) - do not modify or remove them.
            Create a comprehensive markdown document with the exact details including all links.
            """,
            agent=markdown_formatter,
            expected_output='A well-formatted markdown document containing all exact details from the webpage without analysis, with all links preserved in markdown format'
        )
        
        # Create crew
        crew = Crew(
            agents=[content_scraper, markdown_formatter],
            tasks=[scrape_task, markdown_task],
            verbose=self.verbose
        )
        
        self.logger.info("Crew setup completed")
        return crew
