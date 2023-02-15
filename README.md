# 📚🤖 S2QA: QA on research papers

Have you ever wondered what research papers have to say about your burning questions? Look no further than Semantic Scholar Q&A with GPT-3! 🙌

This Python script lets you enter a question, and it uses the power of Semantic Scholar and GPT-3 to generate an answer based on the content of the top research papers. 🤖🔍

- [s2qa_nb.ipynb](s2qa_nb.ipynb) - Main notebook
- [utils.py](utils.py) - Has all the necessary functions for search and GPT-3  prompting

## Requirements 🧰

- `OpenAI API key`
- `Semantic Scholar Academic Graph API key` - https://www.semanticscholar.org/product/api

These can be added in the (constants.py)[constants.py]

## Pipeline 🚀

1️⃣ We begin by searching the vast and ever-growing database of Semantic Scholar to find the most up-to-date and relevant papers and articles related to your question.

2️⃣ We then use [SPECTER](https://github.com/allenai/specter) to embed these papers and re-rank the search results, ensuring that the most informative and relevant articles appear at the top of your search results.

3️⃣ Finally, we use the powerful natural language processing capabilities of GPT-3 to generate informative and accurate answers to your question, using custom prompts to ensure the best results.

## Customizable 🖊️

- Try other open embeddings methods on huggingface to see better re-ranking results. 

- Try other prompts or refine the current prompt to get the best answer.