# News Article Classifier using LLMs and DSPy
## Pre-requisites
- Uses [Ollama](https://ollama.com/) to run LLM locally and go easy on the wallet and the cost of time/latency.
- Uses [DSPy](https://dspy.ai/) to use programming—rather than prompting large language models.
- Uses Gradio to stitch together a UI to take url as input and return a 3 keyword tuple (Sentiment, Confidence and Reasoning)
- Uses [uv](https://docs.astral.sh/uv/) to manage pkg dependecies

## Running instructions
- Use `uv sync` to install all packages to setup the deps
- Use `ollama run llama3.2:latest` to get llama3.2:latest running locally
- Use `uv run main.py` to kickstart the Gradio app
- Follow Instruction on terminal for the url for the Gradio app

### Screenshots
![Screenshot of Positive Example](./screenshots/positve-example.png)
![Screenshot of Negative Example](./screenshots/negative-example.png)

## Next steps
- Add development set with examples for the model evaluation
- Add a metric for the classifier
- Add optimizer for fine-tuning my classifier


