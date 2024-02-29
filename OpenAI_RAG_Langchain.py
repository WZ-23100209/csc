# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + colab={"base_uri": "https://localhost:8080/"} id="qMA2EB73ENtq" outputId="9b803376-bc09-41ad-f7b6-389d5139ec57"
# # !pip install -qqq langchain
# # !pip install -qqq openai
# # !pip install -qqq faiss-gpu

# + colab={"base_uri": "https://localhost:8080/"} id="Im2eLTAQQW7f" outputId="b6bfc219-748d-4e03-fb9f-052a6e7b8306"
# # %pip install --upgrade --quiet  langsmith langchainhub --quiet
# # %pip install --upgrade --quiet  pandas duckduckgo-search --quiet

# + colab={"base_uri": "https://localhost:8080/"} id="CxMXv-g-M3lo" outputId="9ffe10cd-79d9-4c47-d9d6-fdd7195ccdc3"
# # !pip install -qqq langchain-openai

# + id="Nr7b3yYQSLUT"
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Testing_langchain_and_langsmith"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "ls__ca83d326d2574770bc46e469346dc63d"  # Update to your API key

# + id="hQw7BYBQQkAk"
from langsmith import Client

client = Client()

# + id="4gy7U44SRGGV"
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# + id="nERQf53FRMyO"
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import tool

# + id="wyEjxBB4SwH7"
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

# + id="hlIfMtqzV2u9"
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# + colab={"base_uri": "https://localhost:8080/"} id="oe7FqPobJgh3" outputId="d630214e-d3c1-4a60-ec4e-30427d65cfaf"
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature = 0.1,openai_api_key="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")
llm.temperature


# + [markdown] id="GYUXYzeg4-E_" jp-MarkdownHeadingCollapsed=true
# # word length and websearch tools

# + colab={"base_uri": "https://localhost:8080/"} id="_CeVodhAL1me" outputId="153048c0-5c5e-4d4d-9f30-244089f759c8"
@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)


get_word_length.invoke("abc")


# + id="1gr_UbOxVOMw"
@tool
def web_search(query: str) -> str:
    """Returns the search results for a query"""
    wrapper = DuckDuckGoSearchAPIWrapper(max_results=2)

    search = DuckDuckGoSearchResults(api_wrapper=wrapper, source="news")

    return search.run(query)


# + id="ugkzA_uDL6Q5"
tools = [get_word_length, web_search]

# + [markdown] id="KDz3bZ7QbKaB"
# # Adding RAG

# + colab={"base_uri": "https://localhost:8080/"} id="diJXhvdebyDr" outputId="4e973b29-9043-405d-969b-a3f5d5aee524"
# # !pip install -qqq pypdf

# + colab={"base_uri": "https://localhost:8080/"} id="oPSnu4fscIUN" outputId="165f6df1-b3cd-432c-d7e8-f8b1febbdf56"
# from google.colab import drive
# drive.mount('/content/drive')

# + id="Cn47Rty9bNV0"
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("./policy_doc_v2.pdf")
documents = loader.load()

# + id="IsnTGOI8d6TL"
tools = []

# + id="g8Hj0catbP5D"
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings(openai_api_key="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")
db = FAISS.from_documents(texts, embeddings)

retriever = db.as_retriever(
    search_kwargs={'k': 1})

# from langchain.tools.retriever import create_retriever_tool

# tool = create_retriever_tool(
#     retriever,
#     "search_tkxel_policy",
#     "Searches and returns excerpts from the policy docs of tkxel, an online retailer of apparel",
# )
# tools.append(tool)

# + colab={"base_uri": "https://localhost:8080/", "height": 192} id="ia3w_6VGiSAr" outputId="25b31585-3403-4349-8f08-3e21d63d2627"
docs = retriever.get_relevant_documents("I want to refund my product")
docs[0].page_content

# + [markdown] id="9wgZVJacfLtf"
# # Defining API functions

# + id="E3xN6envfZ64"
api_url = "http://localhost:5000"

# + id="LxU-4yHUfK40"
import requests
from langchain.agents import tool
# import BaseException
@tool
def get_customer(id : int):

    """gets customer on the basis of customer id, returns the following schema: {'address': string, 'customer_id': int, 'name': string, 'phone': string}"""

    url = f'{api_url}/customer'
    params = {'customer_id': id}

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        customer = data.get('customer')
        return customer
    else:
        return "the arguments were missing/invalid, or the server failed to fetch data"

@tool
def get_order( filter, value):
    """gets order on the basis of a filter 'order_id' or 'product_id' or 'customer_id', and its value"""
    url = f'{api_url}/order'
    params = {'filter': filter, 'value': value}

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        order = data.get('order')

        return order
    else:
        return "the arguments were missing/invalid, or the server failed to fetch data"

@tool
def get_product(id):
    """gets the product on the basis of a product id, it will return the following schema: product_id, product_name, price, stock"""
    url = f'{api_url}/product'
    params = {'product_id': id}

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        product = data.get('product')
        return product
    else:
        return "the arguments were missing/invalid, or the server failed to fetch data"

@tool
def convert_unix_to_date(unix_time: int):
    """converts unix time to human readable format"""
    import datetime

    dt = datetime.datetime.utcfromtimestamp(unix_time)

    return dt

@tool
def initiate_return(order_id):
    """initiates the return process, call only when the user provides order_id"""
    return "return has been initiated"

@tool
def check_return_status(return_id):
    """check the status of an initiated return"""
    return "return is being processed, please remain patient"

@tool
def initiate_exchange(order_id):
    """initiate the process for exchange of goods"""
    return "exchange initiated"

@tool
def multiply(a, b)->float:
    """you can use this to multiply quantity with price"""
    a = float(a)
    b = float(b)
    return a*b

@tool
def add(a:float, b:float)->float:
    """you can use this to add prices"""
    return a+b

tools = [get_customer, get_order, get_product, convert_unix_to_date, initiate_return, check_return_status, initiate_exchange, add, multiply]

# tools.append(get_customer)
# tools.append(get_order)
# tools.append(get_product)
# tools.append(convert_unix_to_date)
# tools.append(initiate_return)
# tools.append(check_return_status)
# tools.append(initiate_exchange)
# tools.append(add)
# tools.append(multiply)

# + [markdown] id="EwGROLbSfHcv" jp-MarkdownHeadingCollapsed=true
# # Prompt the model

# + id="aJlgXKDEMF_Z"
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a customer service agent who is provided the user's prompt history, and our system's response. Priorotise responding to the user's most recent prompt. Do not make anything up, but explain to the user
            in non technical terms what the problem is. Do not mention anything about the api or code/variables as they are secret implementational details and should not be shared with the customer.
            Ensure that your response is confident, clear, concise, and directly addresses the customer's question. Address the customer in a friendly
            and professional manner. Ensure that your response is short.""",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# + id="u_0PoWCEMKVq"
llm_with_tools = llm.bind_tools(tools)

# + id="PW88FFz2mg0s"
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
setup_and_retrieval = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)

# + id="FQNj0QTsNhbD"
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
    }
    | setup_and_retrieval
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

# + id="T0NQwvLINj9G"
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# + colab={"base_uri": "https://localhost:8080/"} id="uqFGkrmsNmFu" outputId="9cdff1ee-7e71-422a-bec8-fdcef679dd66"
list(agent_executor.stream({"input": "i wanna cancel my order"}))

# + [markdown] id="fCOGoxzB6jVq"
# # Adding Memory custom

# + id="uDg71XRtNQok"
llm_with_tools = llm.bind_tools(tools)

# + id="uQtuIqqo6oKP"
from langchain.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
MEMORY_KEY = "chat_history"
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a customer service agent, do not answer any non retail questions, do not make any future predictions and do not assume anything, ask the customer for any missing information. Ensure that your response is confident, clear, concise, and directly addresses the customer's question. Address the customer in a friendly
            and professional manner. Ensure that your response is short. You are also provided the strore policy as context.
            policy context:{context}
            """,
        ),
        MessagesPlaceholder(variable_name=MEMORY_KEY),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


# + id="-VYyFBibquDU"
def get_relevant_docs(x:dict):
    docs = retriever.get_relevant_documents(x["input"])
    print(docs[0].page_content)
    return docs[0].page_content


# + id="Vbz_-1K37ba9"
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x["chat_history"],
        "context": lambda x: get_relevant_docs(x)
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# + id="cSUo7ch87j4V"
chat_history = []


# + id="Vxk2tyYkKNEF"
def answer_question(message, history):
    input2 = message
    result = agent_executor.invoke({"input": input2, "chat_history": history})
    history.extend(
        [
            HumanMessage(content=input2),
            AIMessage(content=result["output"]),
        ]
    )
    return result["output"]


# + colab={"base_uri": "https://localhost:8080/", "height": 298} id="u_xUurGWKNvS" outputId="56b75825-9d72-4073-faa6-72f8cf255a07"
answer_question("what is the total cost for this order", chat_history)

# + colab={"base_uri": "https://localhost:8080/"} id="52yVgJPhKTFu" outputId="6c79dd0e-f3ff-4c1c-f201-1d6571dd64b7"
answer_question("My order id is ABCDEF", chat_history)

# + [markdown] id="AeyDKMvhOHWv" jp-MarkdownHeadingCollapsed=true
# # Adding Memory

# + id="WOgMGMQrQ_Sm"
from langchain.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

# + id="1olSxVYUNvn8"
MEMORY_KEY = "chat_history"
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are very powerful assistant, but bad at calculating lengths of words, and don't have knowledge of facts",
        ),
        MessagesPlaceholder(variable_name=MEMORY_KEY),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# + id="YdieuQgBOSSr"
chat_history = []

# + id="BKD2RMz1OV8b"
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# + colab={"base_uri": "https://localhost:8080/"} id="apMwQ53mOYR-" outputId="adc8cfd7-9722-49d4-f4c6-7ee882ea5d0f"
input1 = "how many letters does the word kimpambe have?"
result = agent_executor.invoke({"input": input1, "chat_history": chat_history})
chat_history.extend(
    [
        HumanMessage(content=input1),
        AIMessage(content=result["output"]),
    ]
)
# input2 = "iyes do that please"
# result = agent_executor.invoke({"input": input2, "chat_history": chat_history})
# chat_history.extend(
#     [
#         HumanMessage(content=input2),
#         AIMessage(content=result["output"]),
#     ]
# )

# + colab={"base_uri": "https://localhost:8080/"} id="_xZwRCJcXzvj" outputId="9c9a634e-feb2-4a24-cf97-16f196786a32"
result

# + colab={"base_uri": "https://localhost:8080/"} id="b3CVE_G8OdEa" outputId="ca1b3d5a-9473-4e45-8591-4a16714c2ebe"
agent_executor.invoke({"input": "why is niggas in paris called niggas in paris", "chat_history": chat_history})

# + [markdown] id="njNYub9jPtnt"
# # Fast Api

# + id="VqzoafmZP8jX"
from typing import Any
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    inputs: str

app.dialogue_history = []

@app.get("/")
def status_gpu_check() -> dict[str, str]:
    gpu_msg = "Available"
    return {
        "status": "I am ALIVE!",
        "gpu": gpu_msg
    }

@app.post("/generate/")
async def generate_text(data: TextInput) -> dict[str, str]:
    try:
        # print(type(data))
        print(data)
        response = answer_question(data.inputs, app.dialogue_history)
        
        output= str(response)
        print(output)
        return {"generated_text": output}
    except Exception as e:
        print("Type of recieved input: ",type(data))
        print("Actual input: ", data)
        print("ERROR: ",e)

@app.post("/clear_history")
async def clear_history():
    try:
        app.dialogue_history = []
        
        return {"message": str(app.dialogue_history)}
    except Exception as e:
        return {"message": e}, 500

# -

import uvicorn
config = uvicorn.Config(app)
server = uvicorn.Server(config)
await server.serve()

text = """
Installation →
CI Documentation Status codecov.io Code style: black GitHub language count Conda Version Pypi pyversions Binder:lab Binder:notebook launch - renku 

Jupytext
Have you always wished Jupyter notebooks were plain text documents? Wished you could edit them in your favorite IDE? And get clear and meaningful diffs when doing version control? Then, Jupytext may well be the tool you’re looking for!

Text Notebooks
A Python notebook encoded in the py:percent format has a .py extension and looks like this:

# %% [markdown]
# This is a markdown cell

# %%
def f(x):
  return 3*x+1
Only the notebook inputs (and optionally, the metadata) are included. Text notebooks are well suited for version control. You can also edit or refactor them in an IDE - the .py notebook above is a regular Python file.

We recommend the percent format for notebooks that mostly contain code. The percent format is available for Julia, Python, R and many other languages.

If your notebook is documentation-oriented, a Markdown-based format (text notebooks with a .md extension) might be more appropriate. Depending on what you plan to do with your notebook, you might prefer the Myst Markdown format, which interoperates very well with Jupyter Book, or Quarto Markdown, or even Pandoc Markdown.

Installation
Install Jupytext in the Python environment that you use for Jupyter. Use either

pip install jupytext
or

conda install jupytext -c conda-forge
Then, restart your Jupyter Lab server, and make sure Jupytext is activated in Jupyter: .py and .md files have a Notebook icon, and you can open them as Notebooks with a right click in Jupyter Lab.



Paired Notebooks
Text notebooks with a .py or .md extension are well suited for version control. They can be edited or authored conveniently in an IDE. You can open and run them as notebooks in Jupyter Lab with a right click. However, the notebook outputs are lost when the notebook is closed, as only the notebook inputs are saved in text notebooks.

A convenient alternative to text notebooks are paired notebooks. These are a set of two files, say .ipynb and .py, that contain the same notebook, but in different formats.

You can edit the .py version of the paired notebook, and get the edits back in Jupyter by selecting reload notebook from disk. The outputs will be reloaded from the .ipynb file, if it exists. The .ipynb version will be updated or recreated the next time you save the notebook in Jupyter.

To pair a notebook in Jupyter Lab, use the command Pair Notebook with percent Script from the Command Palette:



To pair all the notebooks in a certain directory, create a configuration file with this content:

# jupytext.toml at the root of your notebook directory
formats = "ipynb,py:percent"
Command line
Jupytext is also available at the command line. You can

pair a notebook with jupytext --set-formats ipynb,py:percent notebook.ipynb

synchronize the paired files with jupytext --sync notebook.py (the inputs are loaded from the most recent paired file)

convert a notebook in one format to another with jupytext --to ipynb notebook.py (use -o if you want a specific output file)

pipe a notebook to a linter with e.g. jupytext --pipe black notebook.ipynb

Sample use cases
Notebooks under version control
This is a quick how-to:

Open your .ipynb notebook in Jupyter and pair it to a .py notebook, using either the pair command in Jupyter Lab, or a global configuration file

Save the notebook - this creates a .py notebook

Add this .py notebook to version control

You might exclude .ipynb files from version control (unless you want to see the outputs versioned!). Jupytext will recreate the .ipynb files locally when the users open and save the .py notebooks.

Collaborating on notebooks with Git
Collaborating on Jupyter notebooks through Git becomes as easy as collaborating on text files.

Assume that you have your .py notebooks under version control (see above). Then,

Your collaborator pulls the .py notebook

They open it as a notebook in Jupyter (right-click in Jupyter Lab)

At that stage the notebook has no outputs. They run the notebook and save it. Outputs are regenerated, and a local .ipynb file is created

They edit the notebook, and push the updated notebook.py file. The diff is nothing else than a standard diff on a Python script.

You pull the updated notebook.py script, and refresh your browser. The input cells are updated based on the new content of notebook.py. The outputs are reloaded from your local .ipynb file. Finally, the kernel variables are untouched, so you have the option to run only the modified cells to get the new outputs.

Editing or refactoring a notebook in an IDE
Once your notebook is paired with a .py file, you can easily edit or refactor the .py representation of the notebook in an IDE.

Once you are done editing the .py notebook, you will just have to reload the notebook in Jupyter to get the latest edits there.

Note: It is simpler to close the .ipynb notebook in Jupyter when you edit the paired .py file. There is no obligation to do so; however, if you don’t, you should be prepared to read carefully the pop-up messages. If Jupyter tries to save the notebook while the paired .py file has also been edited on disk since the last reload, a conflict will be detected and you will be asked to decide which version of the notebook (in memory or on disk) is the appropriate one.

More resources
Read more about Jupytext in the documentation.

If you’re new to Jupytext, you may want to start with the FAQ or with the Tutorials.

There is also this short introduction to Jupytext: .

Table of Contents
Installation
Text notebooks
Paired notebooks
Frontend extension
Jupytext’s configuration file
Advanced options
Notebooks as code
Notebooks as Markdown
Supported Languages
Jupytext CLI
Pre-commit hook
FAQ
Blog posts and talks
Contributing
Developing Jupytext
Jupytext ChangeLog
Sponsored: EthicalAds
Reach specific developers on the open source, privacy-first ad network: EthicalAds
Ad by EthicalAds   ·   ℹ️
 
Installation →
Logo



Navigation
Installation
Text notebooks
Paired notebooks
Frontend extension
Jupytext’s configuration file
Advanced options
Notebooks as code
Notebooks as Markdown
Supported Languages
Jupytext CLI
Pre-commit hook
FAQ
Blog posts and talks
Contributing
Developing Jupytext
Jupytext ChangeLog
Quick search
©2018-2023, The Jupytext Team. | Powered by Sphinx 7.2.6 & Alabaster 0.7.12 | Page source
Fork me on GitHub
  v: latest """

# +
# %%time
from openai import OpenAI
client = OpenAI(api_key = "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")

response = client.embeddings.create(
    input='abc',
    model="text-embedding-3-small"
)

# print(response.data[0].embedding)
