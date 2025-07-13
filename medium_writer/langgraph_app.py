# Basic LangGraph setup
from langgraph.graph import Graph
from langgraph.prebuilt import tools


def plan_content(context: dict) -> dict:
    """Simple planning step."""
    context['plan'] = 'Write an article about Python tips.'
    return context


def generate_content(context: dict) -> dict:
    """Generate content placeholder."""
    plan = context.get('plan', '')
    context['article'] = f"This is a draft generated from the plan: {plan}"
    return context


def build_graph() -> Graph:
    g = Graph()
    g.add_node('plan', plan_content)
    g.add_node('generate', generate_content)
    g.add_edge('plan', 'generate')
    g.set_entry_point('plan')
    g.set_finish_point('generate')
    return g


if __name__ == '__main__':
    graph = build_graph()
    result = graph.invoke({})
    print(result['article'])
