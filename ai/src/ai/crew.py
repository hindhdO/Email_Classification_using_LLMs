from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from dataClasses import Result, EvaluationResult, TestClassification
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Fetch model names and API base URL from environment variables
LLM_MODEL1 = os.getenv("Classification_llm")
LLM_MODEL2 = os.getenv("Evaluation_llm")
API_BASE = os.getenv("API_BASE")

# Initialize LLM models for classification and evaluation tasks
Classification_llm = LLM(
    model=LLM_MODEL1,
    api_key=API_BASE
)

Evaluation_llm = LLM(
    model=LLM_MODEL2,
    api_key=API_BASE
)

@CrewBase
class Project():
    """
    Project dedicated to email classification using a reasoning chain based on an LLM model.
    
    Attributes:
        agents_config (str): Path to the agents configuration file.
        tasks_config (str): Path to the tasks configuration file.
    """

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def classifier(self) -> Agent:
        """
        Creates an agent responsible for classifying emails.
        
        Returns:
            Agent: The classification agent configured with an LLM model and specific configuration.
        """
        return Agent(
            config=self.agents_config['classifier'],
            verbose=True,
            llm=Classification_llm,
        )
    
    @agent
    def evaluator(self) -> Agent:
        """
        Creates an agent responsible for evaluating the classifier agent's response after a classification ,gives feedback 
        and proposes improvements .
        
        Returns:
            Agent: The evaluation agent configured with an LLM model and specific configuration.
        """
        return Agent(
            config=self.agents_config['evaluator'],
            verbose=True,
            llm=Evaluation_llm,
        )
    
    @agent
    def tester(self) -> Agent:
        """
        Creates an agent responsible for testing the classifier agent's performance through different test cases .
        
        Returns:
            Agent: The testing agent configured with an LLM model and specific configuration.
        """
        return Agent(
            config=self.agents_config["tester"],  
            llm=Evaluation_llm,
            verbose=True
        )

    @task
    def classification_task(self) -> Task:
        """
        Creates the email classification task.
        
        Returns:
            Task: The classification task configured with a specific Pydantic output model.
        """
        return Task(
            config=self.tasks_config['classification_task'],
            output_pydantic=Result,
        )

    @task
    def evaluation_task(self) -> Task:
        """
        Creates the task for evaluating the email classification.
        
        Returns:
            Task: The evaluation task configured with a specific Pydantic output model, with a classification task context.
        """
        return Task(
            config=self.tasks_config['evaluation_task'],
            output_pydantic=EvaluationResult,
            context=[self.classification_task()],
        )

    @task 
    def refine_classification_task(self) -> Task:
        """
        Creates the task for refining email classification by the classifier agent.
        
        Returns:
            Task: The refinement task configured with a specific Pydantic output model, with contexts for classification and evaluation tasks.
        """
        return Task(
            config=self.tasks_config['refine_classification_task'],
            output_pydantic=Result, 
            context=[
                self.classification_task(),  
                self.evaluation_task()     
            ],
        )
    
    @task
    def test_classification_task(self) -> Task:
        """
        Creates the task for testing the email classification.
        
        Returns:
            Task: The test classification task configured with a specific Pydantic output model, with a classification task context.
        """
        return Task(
            config=self.tasks_config["test_classification_task"],
            output_pydantic=TestClassification,
            context=[self.classification_task()]
        )

    @crew
    def crew(self) -> Crew:
        """
        Creates the crew responsible for the email classification process.
        
        Returns:
            Crew: The crew with agents and tasks configured to perform the email classification process sequentially.
        """
        return Crew(
            agents=[
                self.classifier(),
                self.evaluator(),
                self.tester(),
            ],
            tasks=[
                self.classification_task(),
                self.evaluation_task(),
                self.test_classification_task(),
            ],
            process=Process.sequential,
            verbose=True,
        )
