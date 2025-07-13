#!/usr/bin/env python3
"""
Improved LangGraph setup for automated content generation
Uses modern StateGraph API with proper error handling and state management
"""

import logging
from typing import Dict, List, Optional, TypedDict, Annotated
from datetime import datetime
import asyncio
from dataclasses import dataclass
from enum import Enum

from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
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
    topic: Optional[str]
    target_audience: Optional[str]
    content_type: Optional[str]  # article, tutorial, guide, etc.
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
    target_readability: float = 60.0  # Flesch reading ease score
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
            return ChatAnthropic(
                model=self.config.llm_model,
                temperature=self.config.temperature
            )
        else:
            return ChatOpenAI(
                model=self.config.llm_model,
                temperature=self.config.temperature
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
        workflow.add_node("handle_error", self.handle_error)

        # Define the flow
        workflow.add_edge(START, "plan_content")
        workflow.add_edge("plan_content", "research_topic")
        workflow.add_edge("research_topic", "generate_draft")
        workflow.add_edge("generate_draft", "review_content")
        workflow.add_edge("review_content", "optimize_seo")
        workflow.add_edge("optimize_seo", "finalize_content")
        workflow.add_edge("finalize_content", END)

        # Error handling edges
        workflow.add_edge("handle_error", END)

        # Conditional edges for error handling
        workflow.add_conditional_edges(
            "plan_content",
            self._should_continue_or_error,
            {
                "continue": "research_topic",
                "error": "handle_error"
            }
        )

        return workflow.compile()

    def _should_continue_or_error(self, state: ContentState) -> str:
        """Determine if workflow should continue or handle error"""
        if state["status"] == ContentStatus.FAILED:
            return "error"
        return "continue"

    async def plan_content(self, state: ContentState) -> ContentState:
        """Plan content structure and approach"""
        try:
            logger.info(f"Planning content for topic: {state.get('topic', 'Unknown')}")

            state["status"] = ContentStatus.PLANNING
            state["current_step"] = "plan_content"
            state["updated_at"] = datetime.now()

            # Generate content plan using LLM
            planning_prompt = f"""
            Create a comprehensive content plan for the following:
            
            Topic: {state.get('topic', 'General programming tips')}
            Target Audience: {state.get('target_audience', 'Software developers')}
            Content Type: {state.get('content_type', 'article')}
            Keywords: {', '.join(state.get('keywords', []))}
            
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

            response = await self.llm.ainvoke(messages)

            # Parse and store the plan
            state["content_plan"] = {
                "raw_response": response.content,
                "planned_at": datetime.now(),
                "approach": "structured_outline"
            }

            logger.info("Content planning completed successfully")
            return state

        except Exception as e:
            logger.error(f"Error in plan_content: {str(e)}")
            state["status"] = ContentStatus.FAILED
            state["error_message"] = str(e)
            state["retry_count"] = state.get("retry_count", 0) + 1
            return state

    async def research_topic(self, state: ContentState) -> ContentState:
        """Research the topic and gather supporting information"""
        try:
            logger.info("Researching topic and gathering information")

            state["status"] = ContentStatus.RESEARCHING
            state["current_step"] = "research_topic"
            state["updated_at"] = datetime.now()

            research_prompt = f"""
            Research the following topic thoroughly:
            
            Topic: {state.get('topic')}
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

            response = await self.llm.ainvoke(messages)

            state["research_data"] = {
                "findings": response.content,
                "researched_at": datetime.now(),
                "method": "llm_research"
            }

            logger.info("Topic research completed successfully")
            return state

        except Exception as e:
            logger.error(f"Error in research_topic: {str(e)}")
            state["status"] = ContentStatus.FAILED
            state["error_message"] = str(e)
            state["retry_count"] = state.get("retry_count", 0) + 1
            return state

    async def generate_draft(self, state: ContentState) -> ContentState:
        """Generate the initial content draft"""
        try:
            logger.info("Generating content draft")

            state["status"] = ContentStatus.DRAFTING
            state["current_step"] = "generate_draft"
            state["updated_at"] = datetime.now()

            draft_prompt = f"""
            Write a comprehensive article based on the following:
            
            Topic: {state.get('topic')}
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

            response = await self.llm.ainvoke(messages)

            state["draft_content"] = response.content
            state["word_count"] = len(response.content.split())

            logger.info(f"Draft generated successfully ({state['word_count']} words)")
            return state

        except Exception as e:
            logger.error(f"Error in generate_draft: {str(e)}")
            state["status"] = ContentStatus.FAILED
            state["error_message"] = str(e)
            state["retry_count"] = state.get("retry_count", 0) + 1
            return state

    async def review_content(self, state: ContentState) -> ContentState:
        """Review and improve the content quality"""
        try:
            logger.info("Reviewing content quality")

            state["status"] = ContentStatus.REVIEWING
            state["current_step"] = "review_content"
            state["updated_at"] = datetime.now()

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

            response = await self.llm.ainvoke(messages)

            state["draft_content"] = response.content
            state["word_count"] = len(response.content.split())

            # Simple readability scoring (placeholder)
            state["readability_score"] = min(100, max(0, 100 - (state["word_count"] / 20)))

            logger.info("Content review completed successfully")
            return state

        except Exception as e:
            logger.error(f"Error in review_content: {str(e)}")
            state["status"] = ContentStatus.FAILED
            state["error_message"] = str(e)
            state["retry_count"] = state.get("retry_count", 0) + 1
            return state

    async def optimize_seo(self, state: ContentState) -> ContentState:
        """Optimize content for SEO"""
        try:
            logger.info("Optimizing content for SEO")

            state["status"] = ContentStatus.OPTIMIZING
            state["current_step"] = "optimize_seo"
            state["updated_at"] = datetime.now()

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

            response = await self.llm.ainvoke(messages)

            # Extract SEO metadata and optimized content
            state["seo_metadata"] = {
                "optimization_response": response.content,
                "optimized_at": datetime.now(),
                "target_keywords": state.get('keywords', [])
            }

            state["optimized_content"] = state["draft_content"]  # Placeholder
            state["seo_score"] = 75.0  # Placeholder score

            logger.info("SEO optimization completed successfully")
            return state

        except Exception as e:
            logger.error(f"Error in optimize_seo: {str(e)}")
            state["status"] = ContentStatus.FAILED
            state["error_message"] = str(e)
            state["retry_count"] = state.get("retry_count", 0) + 1
            return state

    async def finalize_content(self, state: ContentState) -> ContentState:
        """Finalize the content for publication"""
        try:
            logger.info("Finalizing content")

            state["status"] = ContentStatus.READY
            state["current_step"] = "finalize_content"
            state["updated_at"] = datetime.now()

            # Final validation
            if state["word_count"] < self.config.min_word_count:
                raise ValueError(f"Content too short: {state['word_count']} words")

            if state["word_count"] > self.config.max_word_count:
                logger.warning(f"Content length exceeds target: {state['word_count']} words")

            logger.info("Content finalization completed successfully")
            return state

        except Exception as e:
            logger.error(f"Error in finalize_content: {str(e)}")
            state["status"] = ContentStatus.FAILED
            state["error_message"] = str(e)
            state["retry_count"] = state.get("retry_count", 0) + 1
            return state

    async def handle_error(self, state: ContentState) -> ContentState:
        """Handle errors and determine retry logic"""
        logger.error(f"Handling error: {state.get('error_message', 'Unknown error')}")

        if state.get("retry_count", 0) < self.config.max_retries:
            logger.info(f"Retrying... Attempt {state['retry_count']}/{self.config.max_retries}")
            state["status"] = ContentStatus.PLANNING  # Reset to planning
            state["current_step"] = "retry"
        else:
            logger.error("Max retries reached. Workflow failed.")
            state["status"] = ContentStatus.FAILED

        state["updated_at"] = datetime.now()
        return state

    async def generate_content(self,
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
            result = await self.graph.ainvoke(initial_state)

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
async def main():
    """Example usage of the content generator"""

    config = ContentGenerationConfig(
        max_retries=2,
        min_word_count=800,
        max_word_count=1500,
        llm_model="claude-3-5-sonnet-20241022",
        temperature=0.7
    )

    generator = ContentGenerator(config)

    result = await generator.generate_content(
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
    asyncio.run(main())