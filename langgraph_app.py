#!/usr/bin/env python3
"""
Fixed LangGraph setup for automated content generation
Resolves authentication and state management issues
"""

import logging
import os
from typing import Dict, List, Optional, TypedDict, Annotated
from datetime import datetime
import asyncio
from dataclasses import dataclass
from enum import Enum

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
    OPTIMIZING = "optimizing"
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

    # Content artifacts
    content_plan: Optional[Dict]
    research_data: Optional[Dict]
    draft_content: Optional[str]
    optimized_content: Optional[str]
    seo_metadata: Optional[Dict]

    # Quality metrics
    word_count: int
    readability_score: Optional[float]
    seo_score: Optional[float]

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
    llm_model: str = "claude-3-5-sonnet-20241022"
    temperature: float = 0.7

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

        # Define the workflow
        workflow = StateGraph(ContentState)

        # Add nodes
        workflow.add_node("plan_content", self.plan_content)
        workflow.add_node("research_topic", self.research_topic)
        workflow.add_node("generate_draft", self.generate_draft)
        workflow.add_node("review_content", self.review_content)
        workflow.add_node("optimize_seo", self.optimize_seo)
        workflow.add_node("finalize_content", self.finalize_content)

        # Define the linear flow
        workflow.add_edge(START, "plan_content")
        workflow.add_edge("plan_content", "research_topic")
        workflow.add_edge("research_topic", "generate_draft")
        workflow.add_edge("generate_draft", "review_content")
        workflow.add_edge("review_content", "optimize_seo")
        workflow.add_edge("optimize_seo", "finalize_content")
        workflow.add_edge("finalize_content", END)

        return workflow.compile()

    def plan_content(self, state: ContentState) -> Dict:
        """Plan content structure and approach"""
        try:
            logger.info(f"Planning content for topic: {state['topic']}")

            # Generate content plan using LLM
            planning_prompt = f"""
            Create a comprehensive content plan for the following:
            
            Topic: {state['topic']}
            Target Audience: {state['target_audience']}
            Content Type: {state['content_type']}
            Keywords: {', '.join(state['keywords'])}
            
            Please provide:
            1. Article title (SEO-optimized)
            2. Brief introduction approach
            3. Main sections with key points
            4. Conclusion approach
            5. Call-to-action suggestions
            
            Format as structured JSON.
            """

            messages = [
                SystemMessage(content="You are an expert content strategist and technical writer."),
                HumanMessage(content=planning_prompt)
            ]

            response = self.llm.invoke(messages)

            # Return updated state fields
            return {
                "status": ContentStatus.PLANNING,
                "current_step": "plan_content",
                "updated_at": datetime.now(),
                "content_plan": {
                    "raw_response": response.content,
                    "planned_at": datetime.now(),
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

            research_prompt = f"""
            Research the following topic thoroughly:
            
            Topic: {state['topic']}
            Content Plan: {state.get('content_plan', {}).get('raw_response', '')}
            
            Provide:
            1. Key facts and statistics
            2. Recent developments or trends
            3. Common challenges or pain points
            4. Best practices or solutions
            5. Examples or case studies
            6. Authoritative sources and references
            
            Focus on accurate, up-to-date information.
            """

            messages = [
                SystemMessage(content="You are a meticulous researcher with expertise in technology and software development."),
                HumanMessage(content=research_prompt)
            ]

            response = self.llm.invoke(messages)

            return {
                "status": ContentStatus.RESEARCHING,
                "current_step": "research_topic",
                "updated_at": datetime.now(),
                "research_data": {
                    "findings": response.content,
                    "researched_at": datetime.now(),
                    "method": "llm_research"
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

            draft_prompt = f"""
            Write a comprehensive article based on the following:
            
            Topic: {state['topic']}
            Content Plan: {state.get('content_plan', {}).get('raw_response', '')}
            Research Data: {state.get('research_data', {}).get('findings', '')}
            
            Requirements:
            - Target length: {self.config.min_word_count}-{self.config.max_word_count} words
            - Clear, engaging writing style
            - Include practical examples
            - Use headings and subheadings
            - Add relevant code snippets if applicable
            - Include a compelling introduction and conclusion
            
            Write in Markdown format.
            """

            messages = [
                SystemMessage(content="You are an expert technical writer who creates engaging, informative content."),
                HumanMessage(content=draft_prompt)
            ]

            response = self.llm.invoke(messages)
            word_count = len(response.content.split())

            return {
                "status": ContentStatus.DRAFTING,
                "current_step": "generate_draft",
                "updated_at": datetime.now(),
                "draft_content": response.content,
                "word_count": word_count
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
            
            Please:
            1. Check for clarity and flow
            2. Ensure technical accuracy
            3. Improve readability
            4. Add transitional phrases
            5. Verify all examples work
            6. Enhance engagement
            7. Check for grammar and style issues
            
            Return the improved version in Markdown format.
            """

            messages = [
                SystemMessage(content="You are a senior editor with expertise in technical content."),
                HumanMessage(content=review_prompt)
            ]

            response = self.llm.invoke(messages)
            word_count = len(response.content.split())

            # Simple readability scoring (placeholder)
            readability_score = min(100, max(0, 100 - (word_count / 20)))

            return {
                "status": ContentStatus.REVIEWING,
                "current_step": "review_content",
                "updated_at": datetime.now(),
                "draft_content": response.content,
                "word_count": word_count,
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

    def optimize_seo(self, state: ContentState) -> Dict:
        """Optimize content for SEO"""
        try:
            logger.info("Optimizing content for SEO")

            if state.get("status") == ContentStatus.FAILED:
                return {"status": ContentStatus.FAILED}

            seo_prompt = f"""
            Optimize the following article for SEO:
            
            {state.get('draft_content', '')}
            
            Target Keywords: {', '.join(state.get('keywords', []))}
            
            Please provide:
            1. Optimized title (60 characters max)
            2. Meta description (160 characters max)
            3. Suggested tags/categories
            4. Internal linking opportunities
            5. SEO improvements made to content
            
            Return both the optimized content and SEO metadata.
            """

            messages = [
                SystemMessage(content="You are an SEO specialist focused on technical content optimization."),
                HumanMessage(content=seo_prompt)
            ]

            response = self.llm.invoke(messages)

            return {
                "status": ContentStatus.OPTIMIZING,
                "current_step": "optimize_seo",
                "updated_at": datetime.now(),
                "seo_metadata": {
                    "optimization_response": response.content,
                    "optimized_at": datetime.now(),
                    "target_keywords": state.get('keywords', [])
                },
                "optimized_content": state.get("draft_content", ""),
                "seo_score": 75.0
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

    def finalize_content(self, state: ContentState) -> Dict:
        """Finalize the content for publication"""
        try:
            logger.info("Finalizing content")

            if state.get("status") == ContentStatus.FAILED:
                return {"status": ContentStatus.FAILED}

            # Final validation
            word_count = state.get("word_count", 0)
            if word_count < self.config.min_word_count:
                logger.warning(f"Content length below target: {word_count} words")

            if word_count > self.config.max_word_count:
                logger.warning(f"Content length exceeds target: {word_count} words")

            return {
                "status": ContentStatus.READY,
                "current_step": "finalize_content",
                "updated_at": datetime.now()
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
            content_plan=None,
            research_data=None,
            draft_content=None,
            optimized_content=None,
            seo_metadata=None,
            word_count=0,
            readability_score=None,
            seo_score=None,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            workflow_id=f"content_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        try:
            logger.info(f"Starting content generation for topic: {topic}")
            result = self.graph.invoke(initial_state)

            if result["status"] == ContentStatus.READY:
                logger.info("Content generation completed successfully")
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
    """Example usage of the content generator"""

    # Check for required environment variables
    if not os.getenv('ANTHROPIC_API_KEY') and not os.getenv('OPENAI_API_KEY'):
        print("Error: Please set either ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable")
        print("You can create a .env file with:")
        print("ANTHROPIC_API_KEY=your_key_here")
        print("# or")
        print("OPENAI_API_KEY=your_key_here")
        return

    config = ContentGenerationConfig(
        max_retries=2,
        min_word_count=800,
        max_word_count=1500,
        llm_model="claude-3-5-sonnet-20241022",  # or "gpt-4" for OpenAI
        temperature=0.7
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