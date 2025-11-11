import os

from crewai import Agent, Crew, LLM, Task
from crewai.tools import tool
from logging_config import get_logger
import requests
from bs4 import BeautifulSoup
import os
import json
import google.generativeai as genai

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


@tool("(CaPRA) Catalyst Proposal Reviewing Assistant")
def analyze_proposal_with_gemini(markdown_content: str) -> str:
    """
    Analyze a Cardano proposal using Gemini API's deep research feature.
    This tool takes markdown content of a proposal and returns a comprehensive
    JSON assessment based on Trust/Reliability, Context/Impact, and Financial criteria.
    
    Args:
        markdown_content: The markdown content of the proposal
        
    Returns:
        str: JSON formatted assessment of the proposal
    """
    try:
        # Configure Gemini API
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            return json.dumps({"error": "GEMINI_API_KEY not configured"})
        
        genai.configure(api_key=gemini_api_key)
        
        # Use Gemini 2.5 Pro or Flash model for deep research
        model = genai.GenerativeModel('gemini-2.5-pro')
        
        # Construct the prompt
        prompt = f"""Role: "You are the 'Cardano Proposal Assessment Expert', an impartial, highly structured, and detail-oriented AI analyst."
Goal: "Your sole function is to take raw proposal content (HTML/Markdown) and rigorously evaluate it against a fixed set of Trust/Reliability, Context/Impact, and Financial criteria. You must generate a consistent, structured output for downstream processing."
Format Constraint: "You MUST output your response as a single, valid JSON object. Do not include any introductory text, markdown, or commentary outside of the JSON block."

Proposal Content:
{markdown_content}

Generate a comprehensive assessment in the following JSON format:
{{
  "OverallSummary": "A 3-sentence, professional summary of the proposal's strengths and weaknesses.",
  "ProposalCompleteness": "(+) / (-) - Indicate if all sections appear fulfilled.",
  "TrustAndReliability": [
    {{"Criteria": "Recency of Success (Time Factor)", "Score": "(+)/(-)", "Rationale": ""}},
    {{"Criteria": "Certified Reputation / Role-based Trust", "Score": "(+)/(-)", "Rationale": ""}},
    {{"Criteria": "History of Finished Proposals", "Score": "(+)/(-)", "Rationale": "Combines successful completion and completion with missing milestones."}},
    {{"Criteria": "Company/Team Public History (e.g., 10yrs listed vs. recently funded)", "Score": "(+)/(-)", "Rationale": ""}},
    {{"Criteria": "Transitivity Rate of Positive Feedback", "Score": "(+)/(-)", "Rationale": ""}},
    {{"Criteria": "Cross-Platform Reputation (Positive/Negative)", "Score": "(+)/(-)", "Rationale": "Combines team representation on other platforms and negative representation."}},
    {{"Criteria": "Deception/Misinformation History", "Score": "(-)", "Rationale": ""}},
    {{"Criteria": "Behavioral Volatility Detection", "Score": "(-)", "Rationale": ""}},
    {{"Criteria": "Discrimination Detection", "Score": "(-)", "Rationale": ""}},
    {{"Criteria": "Adaptability/Dynamism Demonstrated", "Score": "(+)", "Rationale": ""}}
  ],
  "ContextAndImpact": [
    {{"Criteria": "Context/Criteria Similarity Rate", "Score": "(+)", "Rationale": ""}},
    {{"Criteria": "Support for Multiple Success Criteria", "Score": "(+)", "Rationale": ""}},
    {{"Criteria": "Measurable Effect and Alignment to Category", "Score": "(+)/(-)", "Rationale": "Combines Measurable Effect & Proposal Category Alignment."}},
    {{"Criteria": "High Impact Potential", "Score": "(+)/(-)", "Rationale": "Based on evidence of high potential impact."}},
    {{"Criteria": "Social / Human Element Importance", "Score": "(+)/(-)", "Rationale": ""}},
    {{"Criteria": "Potential Risk / Negative Effect", "Score": "(-)/(?)", "Rationale": ""}},
    {{"Criteria": "Plan for Scaling Solution / Reach Limit", "Score": "(+)/(-)", "Rationale": ""}},
    {{"Criteria": "Geographic Problem Focus/Relevance", "Score": "(+)/(-)", "Rationale": ""}}
  ],
  "FinancialAssessment": [
    {{"Criteria": "Cost vs. Overall Assessed Impact", "Score": "(+)/(-)", "Rationale": "Evaluates if cost is justified compared to impact."}},
    {{"Criteria": "Non-Commercial Cost Justification", "Score": "(+)/(-)", "Rationale": "Evaluates cost acceptability for non-commercial projects."}},
    {{"Criteria": "Likelihood of Fast Implementation & Commercial Return", "Score": "(+)/(-)", "Rationale": ""}}
  ]
}}"""
        
        # Generate content with Gemini
        response = model.generate_content(prompt)
        
        # Extract the JSON from the response
        response_text = response.text.strip()
        
        # Try to extract JSON if wrapped in markdown code blocks
        if response_text.startswith("```json"):
            response_text = response_text[7:]  # Remove ```json
        if response_text.startswith("```"):
            response_text = response_text[3:]  # Remove ```
        if response_text.endswith("```"):
            response_text = response_text[:-3]  # Remove trailing ```
        
        response_text = response_text.strip()
        
        # Validate it's valid JSON
        json.loads(response_text)
        
        return response_text
    except Exception as e:
        return json.dumps({"error": f"Error analyzing proposal: {str(e)}"})



class ResearchCrew:
    def __init__(self, verbose=True, logger=None):
        self.verbose = verbose
        self.logger = logger or get_logger(__name__)
        self.crew = None
        self.logger.info("ResearchCrew initialized")

    def create_crew(self, url):
        """
        Create a crew to transform URL content to markdown and analyze it.
        
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
        
        # Create Cardano Proposal Assessment Expert Agent
        proposal_assessor = Agent(
            role='Cardano Proposal Assessment Expert',
            goal='Rigorously evaluate Cardano proposals against Trust/Reliability, Context/Impact, and Financial criteria',
            backstory='You are an impartial, highly structured, and detail-oriented AI analyst specializing in '
                      'Cardano proposal assessment. You evaluate proposals systematically using a fixed set of '
                      'criteria and generate structured JSON assessments for downstream processing.',
            verbose=self.verbose,
            allow_delegation=False,
            tools=[analyze_proposal_with_gemini]
        )
        
        # Create Professional Report Writer Agent
        report_writer = Agent(
            role='Professional Report Writer',
            goal='Transform structured JSON assessment data into beautifully formatted, professional markdown reports',
            backstory='You are an expert technical writer who specializes in creating visually appealing, '
                      'professional reports. You excel at transforming structured data into clear, engaging '
                      'markdown documents with proper formatting, visual indicators, and logical flow. '
                      'You preserve all data integrity while making information accessible and attractive.',
            verbose=self.verbose,
            allow_delegation=False,
            llm=llm
        )
        
        self.logger.info("Created content scraper, markdown formatter, proposal assessor, and report writer agents")
        
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
        
        # Task to analyze the proposal with Gemini
        assessment_task = Task(
            description="""
            Analyze the markdown proposal content using the (CaPRA) Catalyst Proposal Reviewing Assistant tool.
            
            You must:
            1. Take the markdown content from the previous task (Markdown Documentation Specialist)
            2. Use the '(CaPRA) Catalyst Proposal Reviewing Assistant' tool to perform a deep assessment of the proposal
            3. The tool will use Gemini API's deep research capabilities to evaluate the proposal
            4. Return the JSON assessment exactly as provided by the tool
            
            The tool will output a valid JSON object with:
            - OverallSummary (3-sentence professional summary)
            - ProposalCompleteness ((+) or (-))
            - TrustAndReliability (array of 10 criteria assessments)
            - ContextAndImpact (array of 8 criteria assessments)
            - FinancialAssessment (array of 3 criteria assessments)
            
            Simply pass the markdown content to the tool and return its output.
            """,
            agent=proposal_assessor,
            expected_output='A valid JSON object containing a comprehensive structured assessment of the Cardano proposal',
            context=[markdown_task]  # This task depends on the markdown_task output
        )
        
        # Task to create professional markdown report from JSON assessment
        report_task = Task(
            description="""
            Transform the JSON assessment from the Cardano Proposal Assessment Expert into a beautifully 
            formatted, professional markdown report.
            
            The report MUST include the following sections in this order:
            
            1. **Executive Summary** - Use the OverallSummary from the JSON
            2. **Proposal Completeness** - Display the completeness status with visual indicator
            3. **Trust & Reliability Assessment** - Create a formatted table or list of all 10 criteria
            4. **Context & Impact Assessment** - Create a formatted table or list of all 8 criteria
            5. **Financial Assessment** - Create a formatted table or list of all 3 criteria
            6. **Conclusion** - A brief concluding statement
            
            Formatting requirements:
            - Use appropriate markdown headers (# ## ###)
            - Use visual indicators for scores:
              - ✅ for positive (+) scores
              - ❌ for negative (-) scores
              - ⚠️ for conditional (+)/(-) or uncertain (?) scores
            - Use blockquotes (>) for the Executive Summary
            - Use bullet points or tables for criteria lists
            - Use horizontal rules (---) to separate major sections
            - Use bold and italic text appropriately for emphasis
            - Ensure the report is professional, clean, and easy to read
            
            CRITICAL: Preserve ALL data from the JSON assessment without modification. 
            Do not add, remove, or alter any scores or rationales - only format them beautifully.
            """,
            agent=report_writer,
            expected_output='A professional, beautifully formatted markdown report with all assessment data presented in an easy-to-read format with visual indicators',
            context=[assessment_task]  # This task depends on the assessment_task output
        )
        
        # Create crew with all four tasks
        crew = Crew(
            agents=[content_scraper, markdown_formatter, proposal_assessor, report_writer],
            tasks=[scrape_task, markdown_task, assessment_task, report_task],
            verbose=self.verbose
        )
        
        self.logger.info("Crew setup completed")
        return crew
