# 📚🤖 S2QA: Question Answering on research papers from Semantic Scholar

Have you ever wondered what research papers have to say about your burning questions? Look no further than Semantic Scholar Q&A with GPT-3! 🙌

This Python script lets you enter a question, and it uses the power of Semantic Scholar and GPT-3 to generate an answer based on the content of the top research papers. 🤖🔍

- [s2qa_nb.ipynb](s2qa_nb.ipynb) - main notebook
- [utils.py](utils.py) - has all the necessary functions for search and GPT-3  prompting
- [s2qa_sources_langchain.ipynb](s2qa_sources_langchain.ipynb) - Get better answers with langchain mapreduce but this is very expensive. This returns the sources of the results as well.
----

## Examples

# 🤖 Getting answers from ChatGPT (coming soon)
---
## Question

### How does iron supplementation affect anemia?

## ChatGPT Response:

Iron supplementation is a common approach used to treat and prevent anemia, which is a condition characterized by a decrease in the number of red blood cells or hemoglobin levels in the blood. Here are some findings on how iron supplementation affects anemia:

1.  Daily vs. Intermittent Administration: A study conducted on pregnant women found that daily iron supplementation was more effective in improving hemoglobin levels and reducing the risk of anemia compared to intermittent administration [[1](https://www.semanticscholar.org/paper/a6d2264066edf46f03f1c7df46b0e8079f8e0b63)].
    
2.  Iron Supplementation with Vitamin C: A study conducted on young women found that iron supplementation with vitamin C was more effective in increasing hemoglobin levels compared to iron supplementation alone [[2](https://www.semanticscholar.org/paper/e056f91c0ba1bc3214e0f89aaf375cafc95f3da3)].
    
3.  High Hemoglobin: A study conducted on pregnant women with high hemoglobin levels found that iron supplementation did not improve birth outcomes [[3](https://www.semanticscholar.org/paper/f312120e9e0fa93842dadf5cc55ee367e4d22615)].
    
4.  Early Life Iron Supplementation: A review of studies found that early life iron supplementation may have benefits for cognitive development but may also increase the risk of infections and chronic diseases [[4](https://www.semanticscholar.org/paper/459828144e59ace72493adae1068d460d0989c46)].
    
5.  Chinese Women: A study conducted on Chinese women with no or mild anemia during pregnancy found that iron-containing micronutrient supplementation improved iron status but did not affect perinatal anemia [[5](https://www.semanticscholar.org/paper/c910b35eb94711fb0f17075f88e132f8cf0bf85a)].
    
6.  Novel Approaches: Various novel approaches to treating iron deficiency anemia in pregnancy are being explored, including the use of intravenous iron therapy and non-oral routes of iron administration [[6](https://www.semanticscholar.org/paper/08a1b7100972a9204076cf863448dcdf23ba9c38)].
    
Overall, iron supplementation can be an effective approach to treating and preventing anemia, but the optimal dosing and administration strategies may vary depending on the population being treated. It is important to consult with a healthcare provider before starting iron supplementation to determine the appropriate dosage and to monitor for potential side effects.

---

# 🤖 Answers with sources and langchain mapreduce

<img src="https://github.com/shauryr/S2QA/blob/master/demo.jpg" alt="s2 with langchain and sources" width="500">

# 🤖 Answers with regular "stuffing" context

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



---

## Requirements 🧰

- `OpenAI API key`
- `Semantic Scholar Academic Graph API key` - https://www.semanticscholar.org/product/api

These can be added in the [constants.py](constants.py)

The main third-party package requirements are `tiktoken`, `openai`, `transformers` and `langchain`.

To install all the required packages
```
pip install -r requirements.txt
```

## Pipeline 🚀

1️⃣ `Searching` : We begin by searching the vast and ever-growing database of Semantic Scholar to find the most up-to-date and relevant papers and articles related to your question.

2️⃣ `Re-Ranking` : We then use [SPECTER](https://github.com/allenai/specter) to embed these papers and re-rank the search results, ensuring that the most informative and relevant articles appear at the top of your search results.

3️⃣ `Answering` : Finally, we use the powerful natural language processing capabilities of GPT-3 to generate informative and accurate answers to your question, using custom prompts to ensure the best results.

## Customizable 🖊️

- Try other open embedding methods on huggingface to see better re-ranking results. 

- Try other prompts or refine the current prompt to get even better answers.

## TODO 👈

- ~~Add citations to the statements generated by GPT-3. As we have links to the actual paper this shouldn't be hard to do.~~ See [s2qa_sources_langchain.ipynb](s2qa_sources_langchain.ipynb)
- Evaluate for some questions. Report results