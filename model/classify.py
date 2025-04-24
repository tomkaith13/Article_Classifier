import dspy
from typing import Literal

class Classify(dspy.Signature):
    """
    Determine if a given news article portrays the  given person of interest in a positive or negative light. 
    If person is not mentioned in the article, classify the sentiment as "unrelated". 
    """

    news_article: str = dspy.InputField()
    person_of_interest: str = dspy.InputField()
    sentiment: Literal['unrelated', 'positive', 'negative'] = dspy.OutputField()
    confidence: float = dspy.OutputField()
    reasoning: str = dspy.OutputField()

class SentimentClassifier(dspy.Module):
    """Classify a news article sentiment based on a given person and the news article itself."""
    def __init__(self):
        super().__init__()
        self.classify = dspy.ChainOfThoughtWithHint(Classify)
    
    def forward(self, news_article: str, person_of_interest: str) -> Classify:
        return self.classify(news_article=news_article, person_of_interest=person_of_interest)