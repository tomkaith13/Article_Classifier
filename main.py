import dspy
import requests
from bs4 import BeautifulSoup
from typing import Literal
import gradio as gr
from functools import lru_cache
from urllib.parse import urlparse


lm = dspy.LM(
    'ollama_chat/llama3.2:latest',
    api_base='http://localhost:11434',api_key='',cache=True, temperature=0.1)
dspy.configure(lm=lm)
rag = dspy.ChainOfThought('context, question -> answer')

url = "https://www.bbc.com/news/articles/c20l2evgny6o"
# url = 'https://www.cbc.ca/news/politics/liberal-oppo-csfn-1.7509217'
# url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
# url = 'https://www.cbc.ca/news/politics/english-leaders-debate-election-2025-1.7513834'

headers = {'User-Agent' : 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15'}

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
    """Classify sentiment of a given news article for a given subject as being portrayed 
    in the article as either positive or negative. If the subject is not mentioned, we classify it as unrelated"""

    news_article: str = dspy.InputField()
    subject: str = dspy.InputField()
    sentiment: Literal['unrelated', 'positive', 'negative'] = dspy.OutputField()
    confidence: float = dspy.OutputField()
    reasoning: str = dspy.OutputField()



def main():
    classify = dspy.Predict(Classify)

    def GetSentiment(url: str, subject : str) -> str:
        if urlparse(url)[0] != "https":
            return "Invalid URL"
        article = parse_paras_out_of_news_url(url)
        
        # print("Article:", article)
        resp = classify(news_article=article, subject=subject)
        print("Response:", resp)
        return f'sentiment: {resp.sentiment}, \n\nconfidence: {resp.confidence},\n\nreasoning: {resp.reasoning}'

    demo = gr.Interface(
        fn=GetSentiment,
        inputs=[gr.Textbox(label="Enter URL"), gr.Textbox(label="Subject")],
        outputs=gr.Textbox(label="Sentiment"),
        title="News Article Sentiment Classifier",
        description="""Classify the sentiment of a news article as postive, negative or nuetral based on a given subject.
        Also provide the confidence score ranging from 0 to 1. Also provides reasning on why the sentiment is classified as such.""",
    )
    demo.launch()

if __name__ == "__main__":
    main()
