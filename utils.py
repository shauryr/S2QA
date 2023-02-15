import constants
import requests
import nltk
from transformers import AutoTokenizer, AutoModel
import openai

# important functions
tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
model = AutoModel.from_pretrained('allenai/specter')

def search(query, limit=20, fields=['title', 'abstract']):
    # space between the  query to be removed and replaced with +
    query = query.replace(' ', '+')
    url = f'https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit={limit}&fields={",".join(fields)}'
    headers = {
        'Accept': '*/*',
        'X-API-Key': constants.S2_KEY
    }
    
    response = requests.get(url, headers=headers)
    return response.json()

# function to preprocess the query and remove the stopwords before passing it to the search function
def preprocess_query(query):
    query = query.lower()
    # remove stopwords from the query
    stopwords = set(nltk.corpus.stopwords.words('english'))
    query = ' '.join([word for word in query.split() if word not in stopwords])
    return query

def get_specter_embeddings(text):
    # tokenize the text
    tokens = tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=512)
    # get the embeddings
    embeddings = model(**tokens).pooler_output
    # return the embeddings
    return embeddings.detach().numpy()

def create_context(
    question, df, max_len=3800, size="davinci"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.iterrows():
        
        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4
        
        # If the context is too long, break
        if cur_len > max_len:
            break
        
        # Else add it to the text that is being returned
        returns.append(row["title_abs"])

    # Return the context
    return "\n\n###\n\n".join(returns)

def answer_question(
    df,
    model="text-davinci-003",
    question="What is the impact of creatine on cognition?",
    max_len=3800,
    size="ada",
    debug=False,
    max_tokens=150,
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the question and context
        response = openai.Completion.create(
            prompt=f"Answer the question based on the context below\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""

