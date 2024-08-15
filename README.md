# softname

A script

Install the deps:

```
pip install -r requirements.txt
```

```
cp tpl.dotenv .env
```

Fill the the settings in the `.env` file.


# ------------------------------------------------------------------------ #

This script has been customized for identifying software names in the PubMed abstracts. 
Please check the source code for more details.

For development, first create a virtual environment using conda or venv.

then, install dependecies:

```
pip install -r requirements.txt
```

then, copy and create the `.env`

```
cp tpl.dotenv .env
```

If you want to use the offical OpenAI API, just fill the OpenAI API key and leave the `OPENAI_API_BASE_URL` empty. For example:

```
OPENAI_API_KEY=sk-proj-q7KM6R32Ynmxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_API_BASE_URL=
OPENAI_API_MODEL=gpt-4o-mini
```

If you use other OpenAI-like API, such as ollama, torchchat, or vllm, fill any key or leave blank in the `OPENAI_API_KEY`, fill the right URL and model name. For example, when using a local ollama server, you can use the following settings:

```
OPENAI_API_KEY=
OPENAI_API_BASE_URL=http://localhost:11434/v1/
OPENAI_API_MODEL=llama3.1
```

Fill the API key, and also ensure you placed the pubmed data files in the correct path `/data/pubmed/metadata_36m.tsv`.

Now you can run the following to see whether it works:

```
python main.py demo
```

then you can run the following to see how to cache all the results in a local database:

```
python main.py extract
```

then, you can run the notebook to see how a few samples looks like:

```
python notebook.py
```

For example:

```
[{
    "pmid": "36035133",
    "software": [
      {
        "name": "CIBERSORT",
        "contexts": [
          "The CIBERSORT algorithm was used to explore immune cell infiltration, and the ESTIMATE algorithm was applied to calculate immune and stromal scores. "
        ]
      },
      {
        "name": "ESTIMATE",
        "contexts": [
          "The CIBERSORT algorithm was used to explore immune cell infiltration, and the ESTIMATE algorithm was applied to calculate immune and stromal scores. "
        ]
      }
    ]
}]
```