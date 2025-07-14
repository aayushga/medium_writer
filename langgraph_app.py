#!/usr/bin/env python3
"""
Enhanced LangGraph setup for automated content generation
Includes content expansion, better validation, and improved error handling
"""

import logging
import os
from typing import Dict, List, Optional, TypedDict, Annotated
from datetime import datetime
import asyncio
from dataclasses import dataclass
from enum import Enum
import re
import json

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContentStatus(Enum):
    PLANNING = "planning"
    RESEARCHING = "researching"
    DRAFTING = "drafting"
    REVIEWING = "reviewing"
    EXPANDING = "expanding"  # New status for content expansion
    OPTIMIZING = "optimizing"
    VALIDATING = "validating"  # New status for final validation
    READY = "ready"
    FAILED = "failed"


class ContentState(TypedDict):
    """State schema for content generation workflow"""
    # Input parameters
    topic: str
    target_audience: str
    content_type: str
    keywords: List[str]

    # Workflow state
    status: ContentStatus
    current_step: str
    error_message: Optional[str]
    retry_count: int
    validation_issues: List[str]  # Track validation issues

    # Content artifacts
    content_plan: Optional[Dict]
    research_data: Optional[Dict]
    draft_content: Optional[str]
    expanded_content: Optional[str]  # New field for expanded content
    optimized_content: Optional[str]
    seo_metadata: Optional[Dict]
    verified_sources: List[Dict]  # Track verified sources

    # Quality metrics
    word_count: int
    readability_score: Optional[float]
    seo_score: Optional[float]
    content_depth_score: Optional[float]  # New metric

    # Metadata
    created_at: datetime
    updated_at: datetime
    workflow_id: str


@dataclass
class ContentGenerationConfig:
    """Configuration for content generation"""
    max_retries: int = 3
    min_word_count: int = 800
    max_word_count: int = 2000
    target_readability: float = 60.0
    llm_model: str = "gpt-4o"
    temperature: float = 0.7
    expansion_threshold: float = 0.8  # Expand if below 80% of min word count
    depth_requirements: Dict[str, int] = None  # Requirements for content depth

    def __post_init__(self):
        if self.depth_requirements is None:
            self.depth_requirements = {
                "code_examples": 2,
                "practical_scenarios": 3,
                "detailed_explanations": 4
            }


class ContentGenerator:
    """Main content generation orchestrator"""

    def __init__(self, config: ContentGenerationConfig):
        self.config = config
        self.llm = self._initialize_llm()
        self.graph = self._build_graph()

    def _initialize_llm(self):
        """Initialize the LLM based on configuration"""
        if "claude" in self.config.llm_model.lower():
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable is required for Claude models")
            return ChatAnthropic(
                model=self.config.llm_model,
                temperature=self.config.temperature,
                api_key=api_key
            )
        else:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI models")
            return ChatOpenAI(
                model=self.config.llm_model,
                temperature=self.config.temperature,
                api_key=api_key
            )

    def _build_graph(self) -> StateGraph:
        """Build the content generation workflow graph"""
        workflow = StateGraph(ContentState)

        # Add nodes
        workflow.add_node("plan_content", self.plan_content)
        workflow.add_node("research_topic", self.research_topic)
        workflow.add_node("generate_draft", self.generate_draft)
        workflow.add_node("review_content", self.review_content)
        workflow.add_node("expand_content", self.expand_content)  # New node
        workflow.add_node("optimize_seo", self.optimize_seo)
        workflow.add_node("validate_content", self.validate_content)  # New node
        workflow.add_node("finalize_content", self.finalize_content)

        # Define the flow with conditional branching
        workflow.add_edge(START, "plan_content")
        workflow.add_edge("plan_content", "research_topic")
        workflow.add_edge("research_topic", "generate_draft")
        workflow.add_edge("generate_draft", "review_content")
        workflow.add_edge("review_content", "expand_content")
        workflow.add_edge("expand_content", "optimize_seo")
        workflow.add_edge("optimize_seo", "validate_content")
        workflow.add_edge("validate_content", "finalize_content")
        workflow.add_edge("finalize_content", END)

        return workflow.compile()

    def _datetime_to_string(self, obj):
        """Convert datetime objects to strings for JSON serialization"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._datetime_to_string(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._datetime_to_string(item) for item in obj]
        else:
            return obj

    def _safe_json_dumps(self, obj, indent=2):
        """Safely convert object to JSON string, handling datetime objects"""
        try:
            # Convert datetime objects to strings
            safe_obj = self._datetime_to_string(obj)
            return json.dumps(safe_obj, indent=indent)
        except (TypeError, ValueError) as e:
            logger.warning(f"JSON serialization failed: {e}")
            return str(obj)

    def _calculate_content_depth_score(self, content: str) -> float:
        """Calculate content depth based on various factors"""
        if not content:
            return 0.0

        # Count code examples
        code_blocks = len(re.findall(r'```[\s\S]*?```', content))
        inline_code = len(re.findall(r'`[^`\n]+`', content))

        # Count practical scenarios/examples
        scenarios = len(re.findall(r'\b(example|scenario|case study|instance)\b', content, re.IGNORECASE))

        # Count detailed explanations (paragraphs with technical terms)
        technical_terms = len(
            re.findall(r'\b(implement|configure|execute|analyze|optimize|debug)\b', content, re.IGNORECASE))

        # Calculate depth score (0-100)
        depth_score = min(100, (code_blocks * 10) + (inline_code * 2) + (scenarios * 5) + (technical_terms * 1))

        return depth_score

    def _verify_sources(self, content: str) -> List[Dict]:
        """Extract and validate source citations from content"""
        verified_sources = []

        # Extract URLs and citations
        urls = re.findall(r'https?://[^\s\)]+', content)
        citations = re.findall(r'\[([^\]]+)\]\([^\)]+\)', content)

        for url in urls:
            verified_sources.append({
                "url": url,
                "type": "link",
                "verified": False,  # In a real implementation, you'd check accessibility
                "found_at": datetime.now().isoformat()
            })

        for citation in citations:
            verified_sources.append({
                "citation": citation,
                "type": "reference",
                "verified": True,
                "found_at": datetime.now().isoformat()
            })

        return verified_sources

    def plan_content(self, state: ContentState) -> Dict:
        """Plan content structure and approach"""
        try:
            logger.info(f"Planning content for topic: {state['topic']}")

            planning_prompt = f"""
            Create a comprehensive content plan for the following:
            
            Topic: {state['topic']}
            Target Audience: {state['target_audience']}
            Content Type: {state['content_type']}
            Keywords: {', '.join(state['keywords'])}
            Target Length: {self.config.min_word_count}-{self.config.max_word_count} words
            
            Please provide a detailed JSON structure with:
            1. seo_title: SEO-optimized title (60 chars max)
            2. introduction: Brief introduction approach
            3. main_sections: Array of sections with:
               - title: Section title
               - key_points: Array of key points to cover
               - examples_needed: Number of examples required
               - estimated_words: Estimated word count
            4. conclusion: Conclusion approach
            5. call_to_action: Specific CTA suggestions
            6. required_depth: Specific depth requirements
            
            Ensure the plan accounts for {self.config.depth_requirements} depth requirements.
            """

            messages = [
                SystemMessage(
                    content="You are an expert content strategist and technical writer. Always respond with valid JSON."),
                HumanMessage(content=planning_prompt)
            ]

            response = self.llm.invoke(messages)

            # Try to parse as JSON for better structure
            try:
                content_plan = json.loads(response.content)
            except json.JSONDecodeError:
                logger.warning("Failed to parse response as JSON, using raw response")
                content_plan = {"raw_response": response.content}

            return {
                "status": ContentStatus.PLANNING,
                "current_step": "plan_content",
                "updated_at": datetime.now(),
                "content_plan": {
                    **content_plan,
                    "planned_at": datetime.now().isoformat(),
                    "approach": "structured_outline"
                }
            }

        except Exception as e:
            logger.error(f"Error in plan_content: {str(e)}")
            return {
                "status": ContentStatus.FAILED,
                "error_message": str(e),
                "retry_count": state.get("retry_count", 0) + 1,
                "current_step": "plan_content",
                "updated_at": datetime.now()
            }

    def research_topic(self, state: ContentState) -> Dict:
        """Research the topic and gather supporting information"""
        try:
            logger.info("Researching topic and gathering information")

            if state.get("status") == ContentStatus.FAILED:
                return {"status": ContentStatus.FAILED}

            # Safely convert content_plan to string for the prompt
            content_plan_str = self._safe_json_dumps(state.get('content_plan', {}))

            research_prompt = f"""
            Research the following topic thoroughly:
            
            Topic: {state['topic']}
            Content Plan: {content_plan_str}
            
            Provide comprehensive research including:
            1. Current statistics and data (with sources)
            2. Recent developments and trends (2023-2024)
            3. Common challenges and pain points
            4. Best practices and proven solutions
            5. Real-world examples and case studies
            6. Tool recommendations with specific versions
            7. Performance metrics and benchmarks
            8. Expert opinions and industry insights
            
            For each fact, provide:
            - The specific claim
            - Supporting evidence
            - Reliable source (with URL if possible)
            - Recency of information
            
            Focus on accurate, up-to-date, and verifiable information.
            """

            messages = [
                SystemMessage(
                    content="You are a meticulous researcher with expertise in technology and software development. Provide detailed, well-sourced information."),
                HumanMessage(content=research_prompt)
            ]

            response = self.llm.invoke(messages)

            return {
                "status": ContentStatus.RESEARCHING,
                "current_step": "research_topic",
                "updated_at": datetime.now(),
                "research_data": {
                    "findings": response.content,
                    "researched_at": datetime.now().isoformat(),
                    "method": "llm_research",
                    "depth_level": "comprehensive"
                }
            }

        except Exception as e:
            logger.error(f"Error in research_topic: {str(e)}")
            return {
                "status": ContentStatus.FAILED,
                "error_message": str(e),
                "retry_count": state.get("retry_count", 0) + 1,
                "current_step": "research_topic",
                "updated_at": datetime.now()
            }

    def generate_draft(self, state: ContentState) -> Dict:
        """Generate the initial content draft"""
        try:
            logger.info("Generating content draft")

            if state.get("status") == ContentStatus.FAILED:
                return {"status": ContentStatus.FAILED}

            # Safely convert content_plan to string for the prompt
            content_plan_str = self._safe_json_dumps(state.get('content_plan', {}))

            draft_prompt = f"""
            Write a comprehensive, detailed article based on the following:
            
            Topic: {state['topic']}
            Content Plan: {content_plan_str}
            Research Data: {state.get('research_data', {}).get('findings', '')}
            
            Requirements:
            - Target length: {self.config.min_word_count}-{self.config.max_word_count} words
            - Include at least {self.config.depth_requirements['code_examples']} code examples
            - Include at least {self.config.depth_requirements['practical_scenarios']} practical scenarios
            - Provide {self.config.depth_requirements['detailed_explanations']} detailed explanations
            - Use clear, engaging writing style
            - Include practical, working examples
            - Use proper heading hierarchy (H1, H2, H3)
            - Add relevant code snippets with explanations
            - Include a compelling introduction and conclusion
            - Cite sources where appropriate
            - Focus on actionable insights
            
            Write in Markdown format with proper formatting.
            """

            messages = [
                SystemMessage(
                    content="You are an expert technical writer who creates detailed, engaging, and informative content. Focus on depth and practical value."),
                HumanMessage(content=draft_prompt)
            ]

            response = self.llm.invoke(messages)
            word_count = len(response.content.split())
            depth_score = self._calculate_content_depth_score(response.content)
            verified_sources = self._verify_sources(response.content)

            return {
                "status": ContentStatus.DRAFTING,
                "current_step": "generate_draft",
                "updated_at": datetime.now(),
                "draft_content": response.content,
                "word_count": word_count,
                "content_depth_score": depth_score,
                "verified_sources": verified_sources
            }

        except Exception as e:
            logger.error(f"Error in generate_draft: {str(e)}")
            return {
                "status": ContentStatus.FAILED,
                "error_message": str(e),
                "retry_count": state.get("retry_count", 0) + 1,
                "current_step": "generate_draft",
                "updated_at": datetime.now()
            }

    def review_content(self, state: ContentState) -> Dict:
        """Review and improve the content quality"""
        try:
            logger.info("Reviewing content quality")

            if state.get("status") == ContentStatus.FAILED:
                return {"status": ContentStatus.FAILED}

            review_prompt = f"""
            Review and improve the following article:
            
            {state.get('draft_content', '')}
            
            Current metrics:
            - Word count: {state.get('word_count', 0)}
            - Depth score: {state.get('content_depth_score', 0)}
            
            Please:
            1. Check for clarity and logical flow
            2. Ensure technical accuracy and completeness
            3. Improve readability and engagement
            4. Add smooth transitional phrases
            5. Verify all examples are correct and functional
            6. Enhance explanations with more detail
            7. Check for grammar and style consistency
            8. Ensure proper markdown formatting
            9. Add more practical insights if needed
            10. Strengthen the conclusion and call-to-action
            
            Return the improved version in Markdown format.
            """

            messages = [
                SystemMessage(
                    content="You are a senior technical editor with expertise in creating high-quality, detailed content. Focus on improvement and enhancement."),
                HumanMessage(content=review_prompt)
            ]

            response = self.llm.invoke(messages)
            word_count = len(response.content.split())
            depth_score = self._calculate_content_depth_score(response.content)

            # Simple readability scoring (improved calculation)
            sentences = len(re.findall(r'[.!?]+', response.content))
            avg_words_per_sentence = word_count / max(sentences, 1)
            readability_score = max(0, min(100, 100 - (avg_words_per_sentence - 15) * 2))

            return {
                "status": ContentStatus.REVIEWING,
                "current_step": "review_content",
                "updated_at": datetime.now(),
                "draft_content": response.content,
                "word_count": word_count,
                "content_depth_score": depth_score,
                "readability_score": readability_score
            }

        except Exception as e:
            logger.error(f"Error in review_content: {str(e)}")
            return {
                "status": ContentStatus.FAILED,
                "error_message": str(e),
                "retry_count": state.get("retry_count", 0) + 1,
                "current_step": "review_content",
                "updated_at": datetime.now()
            }

    def expand_content(self, state: ContentState) -> Dict:
        """Expand content if it's below the target length"""
        try:
            logger.info("Checking if content expansion is needed")

            if state.get("status") == ContentStatus.FAILED:
                return {"status": ContentStatus.FAILED}

            current_word_count = state.get("word_count", 0)
            target_threshold = self.config.min_word_count * self.config.expansion_threshold

            if current_word_count >= target_threshold:
                logger.info(f"Content length sufficient: {current_word_count} words")
                return {
                    "status": ContentStatus.EXPANDING,
                    "current_step": "expand_content",
                    "updated_at": datetime.now(),
                    "expanded_content": state.get("draft_content", "")
                }

            logger.info(f"Expanding content from {current_word_count} words to meet target")

            expansion_prompt = f"""
            The following article needs to be expanded to meet the target length of {self.config.min_word_count}-{self.config.max_word_count} words.
            Current length: {current_word_count} words
            
            Current article:
            {state.get('draft_content', '')}
            
            Please expand the article by:
            1. Adding more detailed explanations for complex concepts
            2. Including additional practical examples and use cases
            3. Providing more code snippets with thorough explanations
            4. Adding troubleshooting tips and common pitfalls
            5. Including performance considerations
            6. Adding comparison with alternative approaches
            7. Including more real-world scenarios
            8. Enhancing existing sections with deeper insights
            
            Maintain the same quality and style. Focus on adding value, not just word count.
            Return the expanded version in Markdown format.
            """

            messages = [
                SystemMessage(
                    content="You are an expert technical writer skilled at expanding content with valuable, detailed information."),
                HumanMessage(content=expansion_prompt)
            ]

            response = self.llm.invoke(messages)
            expanded_word_count = len(response.content.split())
            depth_score = self._calculate_content_depth_score(response.content)

            logger.info(f"Content expanded to {expanded_word_count} words")

            return {
                "status": ContentStatus.EXPANDING,
                "current_step": "expand_content",
                "updated_at": datetime.now(),
                "expanded_content": response.content,
                "word_count": expanded_word_count,
                "content_depth_score": depth_score
            }

        except Exception as e:
            logger.error(f"Error in expand_content: {str(e)}")
            return {
                "status": ContentStatus.FAILED,
                "error_message": str(e),
                "retry_count": state.get("retry_count", 0) + 1,
                "current_step": "expand_content",
                "updated_at": datetime.now()
            }

    def optimize_seo(self, state: ContentState) -> Dict:
        """Optimize content for SEO"""
        try:
            logger.info("Optimizing content for SEO")

            if state.get("status") == ContentStatus.FAILED:
                return {"status": ContentStatus.FAILED}

            content_to_optimize = state.get('expanded_content') or state.get('draft_content', '')

            seo_prompt = f"""
            Optimize the following article for SEO:
            
            {content_to_optimize}
            
            Target Keywords: {', '.join(state.get('keywords', []))}
            
            Please provide:
            1. Optimized title (50-60 characters, include primary keyword)
            2. Meta description (150-160 characters, compelling and keyword-rich)
            3. Suggested tags/categories (5-10 relevant tags)
            4. Internal linking opportunities (suggest 3-5 anchor texts)
            5. Header optimization recommendations
            6. Keyword density analysis and improvements
            7. Featured snippet optimization suggestions
            8. Schema markup recommendations
            
            Also provide the SEO-optimized version of the content with:
            - Natural keyword integration
            - Improved heading structure
            - Better readability
            - Enhanced user engagement signals
            
            Return both the optimized content and detailed SEO metadata.
            """

            messages = [
                SystemMessage(
                    content="You are an SEO specialist with deep expertise in technical content optimization and search engine best practices."),
                HumanMessage(content=seo_prompt)
            ]

            response = self.llm.invoke(messages)

            # Calculate SEO score based on keyword presence and structure
            keyword_density = sum(content_to_optimize.lower().count(kw.lower()) for kw in state.get('keywords', []))
            heading_count = len(re.findall(r'^#+\s', content_to_optimize, re.MULTILINE))
            seo_score = min(100, (keyword_density * 5) + (heading_count * 10) + 50)

            return {
                "status": ContentStatus.OPTIMIZING,
                "current_step": "optimize_seo",
                "updated_at": datetime.now(),
                "seo_metadata": {
                    "optimization_response": response.content,
                    "optimized_at": datetime.now().isoformat(),
                    "target_keywords": state.get('keywords', []),
                    "keyword_density": keyword_density,
                    "heading_count": heading_count
                },
                "optimized_content": content_to_optimize,
                "seo_score": seo_score
            }

        except Exception as e:
            logger.error(f"Error in optimize_seo: {str(e)}")
            return {
                "status": ContentStatus.FAILED,
                "error_message": str(e),
                "retry_count": state.get("retry_count", 0) + 1,
                "current_step": "optimize_seo",
                "updated_at": datetime.now()
            }

    def validate_content(self, state: ContentState) -> Dict:
        """Validate content quality and completeness"""
        try:
            logger.info("Validating content quality")

            if state.get("status") == ContentStatus.FAILED:
                return {"status": ContentStatus.FAILED}

            validation_issues = []
            final_content = state.get('optimized_content', '')
            word_count = len(final_content.split())
            depth_score = state.get('content_depth_score', 0)

            # Validate word count
            if word_count < self.config.min_word_count:
                validation_issues.append(
                    f"Content below minimum word count: {word_count} < {self.config.min_word_count}")
            elif word_count > self.config.max_word_count:
                validation_issues.append(
                    f"Content exceeds maximum word count: {word_count} > {self.config.max_word_count}")

            # Validate content depth
            if depth_score < 50:
                validation_issues.append(f"Content depth insufficient: {depth_score} < 50")

            # Validate structure
            if not re.search(r'^#\s', final_content, re.MULTILINE):
                validation_issues.append("Missing main heading (H1)")

            if len(re.findall(r'^##\s', final_content, re.MULTILINE)) < 3:
                validation_issues.append("Insufficient section headings (H2)")

            # Validate code examples
            code_blocks = len(re.findall(r'```[\s\S]*?```', final_content))
            if code_blocks < self.config.depth_requirements['code_examples']:
                validation_issues.append(
                    f"Insufficient code examples: {code_blocks} < {self.config.depth_requirements['code_examples']}")

            # Validate sources
            if not state.get('verified_sources'):
                validation_issues.append("No sources or references found")

            if validation_issues:
                logger.warning(f"Content validation issues found: {validation_issues}")
            else:
                logger.info("Content validation passed")

            return {
                "status": ContentStatus.VALIDATING,
                "current_step": "validate_content",
                "updated_at": datetime.now(),
                "validation_issues": validation_issues
            }

        except Exception as e:
            logger.error(f"Error in validate_content: {str(e)}")
            return {
                "status": ContentStatus.FAILED,
                "error_message": str(e),
                "retry_count": state.get("retry_count", 0) + 1,
                "current_step": "validate_content",
                "updated_at": datetime.now()
            }

    def finalize_content(self, state: ContentState) -> Dict:
        """Finalize the content for publication"""
        try:
            logger.info("Finalizing content")

            if state.get("status") == ContentStatus.FAILED:
                return {"status": ContentStatus.FAILED}

            validation_issues = state.get("validation_issues", [])

            # Log validation issues but don't fail the process
            if validation_issues:
                for issue in validation_issues:
                    logger.warning(f"Validation issue: {issue}")

            final_word_count = len((state.get('optimized_content', '')).split())
            logger.info(f"Final content statistics:")
            logger.info(f"  - Word count: {final_word_count}")
            logger.info(f"  - Depth score: {state.get('content_depth_score', 0)}")
            logger.info(f"  - SEO score: {state.get('seo_score', 0)}")
            logger.info(f"  - Readability score: {state.get('readability_score', 0)}")
            logger.info(f"  - Sources found: {len(state.get('verified_sources', []))}")

            return {
                "status": ContentStatus.READY,
                "current_step": "finalize_content",
                "updated_at": datetime.now(),
                "word_count": final_word_count
            }

        except Exception as e:
            logger.error(f"Error in finalize_content: {str(e)}")
            return {
                "status": ContentStatus.FAILED,
                "error_message": str(e),
                "retry_count": state.get("retry_count", 0) + 1,
                "current_step": "finalize_content",
                "updated_at": datetime.now()
            }

    def generate_content(self,
                         topic: str,
                         target_audience: str = "Software developers",
                         content_type: str = "article",
                         keywords: List[str] = None) -> ContentState:
        """Main entry point for content generation"""

        initial_state = ContentState(
            topic=topic,
            target_audience=target_audience,
            content_type=content_type,
            keywords=keywords or [],
            status=ContentStatus.PLANNING,
            current_step="initialize",
            error_message=None,
            retry_count=0,
            validation_issues=[],
            content_plan=None,
            research_data=None,
            draft_content=None,
            expanded_content=None,
            optimized_content=None,
            seo_metadata=None,
            verified_sources=[],
            word_count=0,
            readability_score=None,
            seo_score=None,
            content_depth_score=None,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            workflow_id=f"content_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        try:
            logger.info(f"Starting content generation for topic: {topic}")
            result = self.graph.invoke(initial_state)

            if result["status"] == ContentStatus.READY:
                logger.info("Content generation completed successfully")
                # Print summary
                print("\n" + "=" * 50)
                print("CONTENT GENERATION SUMMARY")
                print("=" * 50)
                print(f"Status: {result['status']}")
                print(f"Word count: {result['word_count']}")
                print(f"Depth score: {result.get('content_depth_score', 'N/A')}")
                print(f"SEO score: {result.get('seo_score', 'N/A')}")
                print(f"Readability score: {result.get('readability_score', 'N/A')}")
                print(f"Sources found: {len(result.get('verified_sources', []))}")

                validation_issues = result.get('validation_issues', [])
                if validation_issues:
                    print(f"Validation issues: {len(validation_issues)}")
                    for issue in validation_issues:
                        print(f"  - {issue}")
                else:
                    print("No validation issues found")
                print("=" * 50)
            else:
                logger.error(f"Content generation failed: {result.get('error_message', 'Unknown error')}")

            return result

        except Exception as e:
            logger.error(f"Fatal error in content generation: {str(e)}")
            initial_state["status"] = ContentStatus.FAILED
            initial_state["error_message"] = str(e)
            return initial_state


# Example usage
def main():
    """Example usage of the enhanced content generator"""

    # Check for required environment variables
    if not os.getenv('ANTHROPIC_API_KEY') and not os.getenv('OPENAI_API_KEY'):
        print("Error: Please set either ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable")
        print("You can create a .env file with:")
        print("ANTHROPIC_API_KEY=your_key_here")
        print("# or")
        print("OPENAI_API_KEY=your_key_here")
        return

    # Enhanced configuration
    config = ContentGenerationConfig(
        max_retries=2,
        min_word_count=1000,  # Increased minimum
        max_word_count=2000,
        llm_model="gpt-4o",
        temperature=0.7,
        expansion_threshold=0.8,  # Expand if below 80% of min word count
        depth_requirements={
            "code_examples": 3,
            "practical_scenarios": 4,
            "detailed_explanations": 5
        }
    )

    generator = ContentGenerator(config)

    result = generator.generate_content(
        topic="Advanced Python debugging techniques",
        target_audience="Python developers",
        content_type="tutorial",
        keywords=["python", "debugging", "troubleshooting", "development"]
    )

    print(f"Status: {result['status']}")
    print(f"Word count: {result['word_count']}")
    print(f"Workflow ID: {result['workflow_id']}")

    if result['status'] == ContentStatus.READY:
        print("\n--- Generated Content ---")
        print(result['optimized_content'] or result['draft_content'])

        if result['seo_metadata']:
            print("\n--- SEO Metadata ---")
            print(result['seo_metadata']['optimization_response'])
    else:
        print(f"Error: {result['error_message']}")


if __name__ == "__main__":
    main()
