# 📚🤖 S2QA: Question Answering on research papers from Semantic Scholar

Have you ever wondered what research papers have to say about your burning questions? Look no further than Semantic Scholar Q&A with GPT-3! 🙌

This Python script lets you enter a question, and it uses the power of Semantic Scholar and GPT-3 to generate an answer based on the content of the top research papers. 🤖🔍

- [s2qa_nb.ipynb](s2qa_nb.ipynb) - main notebook
- [utils.py](utils.py) - has all the necessary functions for search and GPT-3  prompting

----

## Examples

```python
>> query = "How does iron supplementation affect anemia?"

>> answer_question(df, question=query, debug=False)

'Iron supplementation can reduce anemia in pregnant women with mild or no anemia, but it can also increase the risk of neonatal jaundice. Iron supplementation can also improve iron stores and decrease anemia in non-pregnant women, but it can also increase the risk of diarrhea. Good adherence and initiation of supplementation before conception are needed to reduce anemia during early pregnancy.'
```


```python
>> query = "What are the effects of sleep training on infants?"

>> answer_question(df, question=query, debug=False)

'Sleep training can lead to improved sleeping patterns, decreased parental stress, and increased parental competence. It can also lead to improved sleep efficiency, sleep onset latency, and sleep duration.'
```

```python
>> query = "What is the impact of creatine on cognition?"

>> answer_question(df, question=query, debug=False)

'Preliminary studies indicate that creatine supplementation (and guanidinoacetic acid; GAA) has the ability to increase brain creatine content in humans. Furthermore, creatine has shown some promise for attenuating symptoms of concussion, mild traumatic brain injury and depression but its effect on neurodegenerative diseases appears to be lacking. However, acute supplementation of creatine does not appear to enhance cognition in healthy subjects, and there is no evidence that creatine supplementation alters participant\'s cognitive function when acutely exposed to hypoxia.'
```
---

## Requirements 🧰

- `OpenAI API key`
- `Semantic Scholar Academic Graph API key` - https://www.semanticscholar.org/product/api

These can be added in the [constants.py](constants.py)

## Pipeline 🚀

1️⃣ `Searching` : We begin by searching the vast and ever-growing database of Semantic Scholar to find the most up-to-date and relevant papers and articles related to your question.

2️⃣ `Ranking` : We then use [SPECTER](https://github.com/allenai/specter) to embed these papers and re-rank the search results, ensuring that the most informative and relevant articles appear at the top of your search results.

3️⃣ `Answering` : Finally, we use the powerful natural language processing capabilities of GPT-3 to generate informative and accurate answers to your question, using custom prompts to ensure the best results.

## Customizable 🖊️

- Try other open embeddings methods on huggingface to see better re-ranking results. 

- Try other prompts or refine the current prompt to get the best answer.