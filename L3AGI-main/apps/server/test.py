# from langchain.smith import RunEvalConfig, run_on_dataset
# from langchain_community.chat_models import ChatOpenAI
# from langsmith import Client
# from langchain.agents import AgentType, initialize_agent
# from langchain.tools import get_tools
# from langchain.prompts import SystemMessage
# from langchain.output_parsers import ConvoOutputParser

# Import XAgent
from xagent import XAgent  # Import XAgent library

# Define the XAgent factory function
def agent_factory():
    # Create and configure the XAgent instance
    xagent = XAgent(model="gpt-3.5-turbo", temperature=0.5, max_iterations=5)
    return xagent

# Initialize the agent using the factory function
agent = agent_factory()

# Define evaluation configuration
eval_config = {
    "evaluators": ["qa", {"criteria": "helpfulness"}, {"criteria": "conciseness"}],
    "input_key": "input",
    "eval_llm": {"model": "gpt-3.5-turbo", "temperature": 0.5}
}

# Define a function to run the evaluation on a dataset
def run_on_dataset(agent_factory, dataset_name, eval_config):
    # Implement the dataset evaluation logic using XAgent
    # Example:
    xagent = agent_factory()
    results = xagent.run(dataset=dataset_name, evaluation_config=eval_config)
    return results

# Run the evaluation on the dataset
chain_results = run_on_dataset(agent_factory, "test-dataset", eval_config)

# Print the results
print(chain_results)
