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

url = "https://www.bbc.com/news/articles/c20l2evgny6o"
# url = 'https://www.cbc.ca/news/politics/liberal-oppo-csfn-1.7509217'
# url = "https://en.wikipedia.org/wiki/Python_(programming_language)"

headers = {'User-Agent' : 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15'}

def search_wikipedia(query: str):
    """
    Searches Wikipedia abstracts using the ColBERTv2 model and returns the top results.

    Args:
        query (str): The search query string.

    Returns:
        list: A list of strings containing the text of the top search results.
    """
    results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=10)
    return [x['text'] for x in results]

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
            return r.text
        else:
            return f"Error: {r.status_code}"
    except requests.exceptions.Timeout:
        return "Error: Request timed out"
    except Exception as e:
        return f"Error: {str(e)}"

rag = dspy.ChainOfThought('context, question -> answer')

class Classify(dspy.Signature):
    """Classify sentiment of a given news article as being either nuetral, positive or negative."""

    news_article: str = dspy.InputField()
    sentiment: Literal['nuetral', 'positive', 'negative'] = dspy.OutputField()
    confidence: float = dspy.OutputField()



def main():
    # print("Hello from dspy-test!")
    # lm("Say this is a test!", temperature=0.1)  # => ['This is a test!']
    # resp = lm(messages=[{"role": "user", "content": 'Start by greeting "Hello Leaguer".'}])
    # # print(resp)
    # print("Response:", resp)
    # # print("results from search_wiki:",res)
    # candidate_question = """
    # Make a case whether Mark Joseph Carney is good for Canada?
    # """
    
    # html_resp = parse_paras_out_of_news_url(url)
    # # print("Response from URL:", html_resp)
    # soup = BeautifulSoup(html_resp, 'html.parser')
    # # print(soup.prettify())

    # article = ''
    
    # for i in soup.find_all('p'):
    #     article += i.get_text()
    
    # print("Article:", article)



    # # res = search_wikipedia(carney_question)
    # # print("results from search_wiki:", res, len(res))
    
    # wiki_results = rag(context=search_wikipedia, question=candidate_question)
    # print("result:", wiki_results)

    # classify = dspy.Predict(Classify)
    # news_res = classify(news_article=article)
    # print("Classification  Of News Result:", news_res)

    # wiki_results = classify(news_article=wiki_results.answer)
    # print("Classification Of Wiki Result:", wiki_results)
    classify = dspy.Predict(Classify)

    def GetSentiment(url: str) -> str:
        if urlparse(url)[0] != "https":
            return "Invalid URL"
        html_resp = parse_paras_out_of_news_url(url)
        soup = BeautifulSoup(html_resp, 'html.parser')
        article : str = ''
    
        for i in soup.find_all('p'):
            article += i.get_text()

        # print("Article:", article)
        resp = classify(news_article=article)
        print("Response:", resp)
        return f'sentiment: {resp.sentiment}, confidence: {resp.confidence}'

    demo = gr.Interface(
        fn=GetSentiment,
        inputs=gr.Textbox(label="Enter URL"),
        outputs=gr.Textbox(label="Sentiment"),
        title="News Article Sentiment Classifier",
        description="Classify the sentiment of a news article as postive, negative or nuetral. Also provide the confidence score ranging from 0 to 1.",
    )
    demo.launch()
    

    
       
    

if __name__ == "__main__":
    main()
