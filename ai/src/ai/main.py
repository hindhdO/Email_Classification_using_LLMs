from crew import Project
import warnings
import json

from sklearn.metrics import classification_report

# Suppress specific warnings related to the pysbd module
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run(email: str) -> dict:
    """
    Executes the complete process of email classification, evaluation, and refinement using CrewAi.

    This function chains the following steps:
        1. Classify the email.
        2. Evaluate the classification.
        3. Refine the classification only if the evaluator disagrees with the classifier.

    Each step uses the tasks and agents defined in the `Project`.

    Parameters:
        email (str): The email to be processed for classification, evaluation, and refinement.

    Returns:
        dict: A dictionary containing the results of each step:
            - classification_output: The result of classifying the email.
            - evaluation_output: The result of evaluating the classification.
            - refinement_output: The result of refining the classification.

    Raises:
        Exception: If an error occurs during the execution of the process.
    """
    try:
        # Initialize the CrewAi project
        project = Project()

        # 1. Classification
        classification_task = project.classification_task()
        classifier = project.classifier()
        classification_result = classifier.execute_task(
            classification_task,
            {"email": email}
        )

        # 2. Evaluation
        evaluation_task = project.evaluation_task()
        evaluator = project.evaluator()
        evaluation_result = evaluator.execute_task(
            evaluation_task,
            {
                "email": email,
                "classification_output": classification_result
            }
        )

        # Convert evaluation result to a dictionary
        if hasattr(evaluation_result, "dict"):
            evaluation_result_dict = evaluation_result.dict()
        elif isinstance(evaluation_result, str):
            try:
                evaluation_result_dict = json.loads(evaluation_result)
            except Exception:
                evaluation_result_dict = {}
        else:
            evaluation_result_dict = evaluation_result

        # 3. Refinement
        refined_output = None
        if evaluation_result_dict.get("verdict") in ["Incorrect", "Partially correct"]:
            refine_task = project.refine_classification_task()
            refined_output = classifier.execute_task(refine_task, {
                "email": email,
                "classification_output": classification_result,
                "evaluation_output": evaluation_result
            })

        return {
            "classification_output": classification_result,
            "evaluation_output": evaluation_result,
            "refinement_output": refined_output.dict() if refined_output and hasattr(refined_output, "dict") else refined_output
        }

    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def run_test(num_tests: int) -> dict:
    """
    Runs a series of tests to evaluate the performance of the email classification process.

    Parameters:
        num_tests (int): The number of test cases to run.

    Returns:
        dict: A dictionary containing the test results.
            - test_output: The result of the classification tests.

    Raises:
        Exception: If an error occurs during the execution of the test suite.
    """
    try:
        project = Project()
        tester = project.tester()
        test_classification_task = project.test_classification_task()
        test_result = tester.execute_task(
            test_classification_task,
            {"num_tests": num_tests}
        )
        return {"test_output": test_result}
    except Exception as e:
        raise Exception(f"An error occurred while running the test suite: {e}")


def compute_metrics(test_cases: list[str]) -> dict:
    """
    Computes classification metrics based on the expected and predicted values for the test cases.

    Parameters:
        test_cases (list): A list of test cases containing expected and predicted classification values.

    Returns:
        dict: A dictionary containing classification metrics such as precision, recall, f1-score.
    """
    expected = [case["expected"] for case in test_cases]
    predicted = [case["predicted"] for case in test_cases]

    # Generate classification report using sklearn's classification_report
    report = classification_report(expected, predicted, output_dict=True, zero_division=0)
    return report
