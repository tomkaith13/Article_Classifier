import dspy
import requests
from bs4 import BeautifulSoup
from typing import Literal

lm = dspy.LM(
    'ollama_chat/llama3.2:latest',
    api_base='http://localhost:11434',api_key='',cache=True, temperature=0.1)
dspy.configure(lm=lm)

url = "https://www.bbc.com/news/articles/c20l2evgny6o"
# url = "https://en.wikipedia.org/wiki/Python_(programming_language)"

def search_wikipedia(query: str):
    results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=10)
    return [x['text'] for x in results]

def parse_url(url: str) -> str:
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.text
        else:
            return f"Error: {r.status_code}"
    except requests.exceptions.Timeout:
        return "Error: Request timed out"

rag = dspy.ChainOfThought('context, question -> answer')

class Classify(dspy.Signature):
    """Classify sentiment of a given news article as being either unbiased or biased (either positively or negatively)."""

    news_article: str = dspy.InputField()
    sentiment: Literal['biased', 'unbiased'] = dspy.OutputField()
    confidence: float = dspy.OutputField()

def main():
    print("Hello from dspy-test!")
    lm("Say this is a test!", temperature=0.1)  # => ['This is a test!']
    resp = lm(messages=[{"role": "user", "content": 'Start by greeting "Hello Leaguer".'}])
    # print(resp)
    print("Response:", resp)
    # print("results from search_wiki:",res)
    carney_question = """
    Make a case whether Mark Joseph Carney is good for Canada?
    """
    
    html_resp = parse_url(url)
    # print("Response from URL:", html_resp)
    soup = BeautifulSoup(html_resp, 'html.parser')
    # print(soup.prettify())

    article = ''
    
    for i in soup.find_all('p'):
        article += i.get_text()
    
    print("Article:", article)



    # res = search_wikipedia(carney_question)
    # print("results from search_wiki:", res, len(res))
    
    wiki_results = rag(context=search_wikipedia, question=carney_question)
    print("result:", wiki_results)

    classify = dspy.Predict(Classify)
    classification_res = classify(news_article=article)
    print("Classification Result:", classification_res)

    classification_carney = classify(news_article=wiki_results.answer)
    print("Classification Of Carney Result:", classification_carney)
    

if __name__ == "__main__":
    main()
