from pydantic import BaseModel
from typing import Literal, Optional, List

class Result(BaseModel):
    """
    Represents the structured result of an email classification task.
    
    Attributes:
        spam (bool): Indicates whether the email is classified as spam.
        category (Literal["Spam", "Urgent", "Not Urgent", "To Read"]): The category of the email.
        reasoning (str): The reasoning behind the classification.
        suggested_response (Optional[str]): The suggested response for the email, if any.
    """
    spam: bool
    category: Literal["Spam", "Urgent", "Not Urgent", "To Read"]
    reasoning: str
    suggested_response: Optional[str] = None


class EvaluationResult(BaseModel):
    """
    Represents the result of an email classification evaluation.
    
    Attributes:
        verdict (Literal["Correct", "Incorrect", "Partially correct"]): The evaluation verdict.
        explanation (str): The explanation behind the verdict.
        evaluator_classification (Literal["Spam", "Urgent", "Not Urgent", "To Read"]): The classification provided by the evaluator.
        evaluator_spam (bool): Indicates whether the evaluator considers the email as spam.
        suggested_correction (Optional[str]): Suggested correction for the classification, if any.
    """
    verdict: Literal["Correct", "Incorrect", "Partially correct"]
    explanation: str
    evaluator_classification: Literal["Spam", "Urgent", "Not Urgent", "To Read"]
    evaluator_spam: bool
    suggested_correction: Optional[str] = None

class TestCase(BaseModel):
    """
    Represents a test case for evaluating email classification.
    
    Attributes:
        email (str): The email content for the test.
        expected (Literal["Spam", "Urgent", "Not Urgent", "To Read"]): The expected classification for the email.
        predicted (Literal["Spam", "Urgent", "Not Urgent", "To Read"]): The predicted classification for the email.
        success (bool): Indicates whether the classification was successful.
        reasoning (Optional[str]): The reasoning behind the prediction, if provided.
    """
    email: str
    expected: Literal["Spam", "Urgent", "Not Urgent", "To Read"]
    predicted: Literal["Spam", "Urgent", "Not Urgent", "To Read"]
    success: bool
    reasoning: Optional[str] = None


class TestClassification(BaseModel):
    """
    Represents the results of a series of classification tests.
    
    Attributes:
        tests (List[TestCase]): A list of test cases for the classification task.
        score (int): The overall score of the classification tests.
        suggestion (Optional[str]): A suggested improvement or feedback based on the test results.
    """
    tests: List[TestCase]
    score: int
    suggestion: Optional[str] = None
