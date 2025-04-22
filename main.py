import dspy
import dspy.evaluate
import requests
from bs4 import BeautifulSoup
from typing import Literal
import gradio as gr
from functools import lru_cache
from urllib.parse import urlparse
from training.training_set import generate_dspy_training_examples, sentiment_match_metric
from dspy.teleprompt import BootstrapFewShot, MIPROv2, LabeledFewShot, BootstrapFewShotWithRandomSearch



lm = dspy.LM(
    'ollama_chat/llama3.2:latest',
    api_base='http://localhost:11434',api_key='',cache=False, temperature=0.1, max_tokens=4096)
dspy.configure(lm=lm)
dspy.settings.configure(track_usage=True)

url = "https://www.bbc.com/news/articles/c20l2evgny6o"
# url = 'https://www.cbc.ca/news/politics/liberal-oppo-csfn-1.7509217'
# url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
# url = 'https://www.cbc.ca/news/politics/english-leaders-debate-election-2025-1.7513834'

headers = {'User-Agent' : 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15'}

# evaluate_flag is used to run the training_set to get a metric of how we did thus far
evaluate_flag = False

# If True, shows prompts
show_history = False

# If true, optimizes the model using specified optimizer
optimize = True

# If True, serializes (state-only. See https://dspy.ai/tutorials/saving/#whole-program-saving for details) the optimized model for classification
save_model = True



@lru_cache(maxsize=1024)
def parse_paras_out_of_news_url(url: str) -> str:
    """
    Fetches the content of a news article from the given URL.

    This function sends an HTTP GET request to the specified URL and returns the 
    response content as a string if the request is successful. If the request fails 
    or encounters an error, an appropriate error message is returned.

    Args:
        url (str): The URL of the news article to fetch.

    Returns:
        str: The content of the news article if the request is successful, or an 
        error message if the request fails or times out.

    Exceptions:
        - Returns "Error: Request timed out" if the request exceeds the timeout limit.
        - Returns "Error: <error_message>" for any other exceptions encountered.
    """
    try:
        r = requests.get(url, timeout=10, headers=headers)
        if r.status_code == 200:
            html_resp = r.text
            soup = BeautifulSoup(html_resp, 'html.parser')
            article : str = ''
        
            for i in soup.find_all('p'):
                article += i.get_text()
            return article

        else:
            return f"Error: {r.status_code}"
    except requests.exceptions.Timeout:
        return "Error: Request timed out"
    except Exception as e:
        return f"Error: {str(e)}"


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
        self.classify = dspy.ChainOfThought(Classify)
    
    def forward(self, news_article: str, person_of_interest: str) -> Classify:
        return self.classify(news_article=news_article, person_of_interest=person_of_interest)




def main():
    classify = dspy.Predict(Classify)
    training_set = generate_dspy_training_examples()

    if evaluate_flag:
        # print('training_set:', training_set)
        evaluator = dspy.Evaluate(devset=training_set, num_threads=5,display_progress=True, display_table=True)
        evaluator(classify, metric=sentiment_match_metric)

    if optimize:
        optimized_classifier = SentimentClassifier()
        optimizerBootStrap = BootstrapFewShotWithRandomSearch(metric=sentiment_match_metric)
        optimized_classifier = optimizerBootStrap.compile(optimized_classifier, trainset=training_set)

        # Zero-shot instruction optimization
        # optimizer = MIPROv2(metric=sentiment_match_metric, auto='light', num_threads=15)
        # optimized_classifier = optimizer.compile(optimized_classifier, teacher=optimized_classifier,
        #                                          trainset=training_set, max_labeled_demos=0, max_bootstrapped_demos=0,
        #                                          requires_permission_to_run=False)

        evaluator = dspy.Evaluate(devset=training_set, num_threads=5,display_progress=True, display_table=True)
        evaluator(optimized_classifier, metric=sentiment_match_metric)
    
        if save_model:
            optimized_classifier.save('./optimized_classifier.json')




    def GetSentiment(url: str, subject : str) -> str:
        if urlparse(url)[0] != "https":
            return "Invalid URL"
        article = parse_paras_out_of_news_url(url)
        
        # print("Article:", article)
        if optimize:
            print("Running Optimized Classifier")
            resp = optimized_classifier(news_article=article, person_of_interest=subject)
        else:
            print("Running Classifier")
            resp = classify(news_article=article, person_of_interest=subject)

        print("Response:", resp)
        if show_history:
            dspy.inspect_history(n=1)
        print("lm usage:", resp.get_lm_usage())
        return f'sentiment: {resp.sentiment}, \n\nconfidence: {resp.confidence},\n\nreasoning: {resp.reasoning}'

    demo = gr.Interface(
        fn=GetSentiment,
        inputs=[gr.Textbox(label="Enter URL"), gr.Textbox(label="Person of Interest")],
        outputs=gr.Textbox(label="Sentiment"),
        title="News Article Sentiment Classifier",
        description="""Classify the sentiment of a news article as postive, negative or nuetral based on a given subject.
        Also provide the confidence score ranging from 0 to 1. Also provides reasning on why the sentiment is classified as such.""",
    )
    demo.launch()

if __name__ == "__main__":
    main()
