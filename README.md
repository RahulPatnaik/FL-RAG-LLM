# Federated Learning LLM with Retrieval Augmented Generation
This model is currently under progress.

## To use the RAG feature with your ollama models

1. Set up venv
```
python -m venv venv
./venv/Scripts/activate
```

2. Donwload requirements
```
pip install -r requirements.txt
```

3. Run the streamlit app
```
streamlit run app.py
```

## Selecting model in the rag.py
```
def __init__(self, llm_model: str = "mistral"):
        self.model = ChatOllama(
            model=llm_model,
            num_gpu=1,
            num_thread=4,
            temperature=0.7,
            num_predict=2048,
            top_k=30,
            repeat_penalty=1.1
        )
```
You can change  ``` str  = "<model_name>" ``` parameter to use the model of your choice present in your laptop's ollama.
