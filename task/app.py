from task._constants import API_KEY
from task.chat.chat_completion_client import DialChatCompletionClient
from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.embeddings.text_processor import TextProcessor, SearchMode
from task.models.conversation import Conversation
from task.models.message import Message
from task.models.role import Role

DIMENSIONS = 1536

# TODO:
# Create system prompt with info that it is RAG powered assistant.
# Explain user message structure (firstly will be provided RAG context and the user question).
# Provide instructions that LLM should use RAG Context when answer on User Question, will restrict LLM to answer
# questions that are not related microwave usage, not related to context or out of history scope
SYSTEM_PROMPT = """You are a RAG-powered assistant that assists users with their questions about microwave usage.
            
## Structure of User message:
`RAG CONTEXT` - Retrieved documents relevant to the query.
`USER QUESTION` - The user's actual question.

## Instructions:
- Use information from `RAG CONTEXT` as context when answering the `USER QUESTION`.
- Cite specific sources when using information from the context.
- Answer ONLY based on conversation history and RAG context.
- If no relevant information exists in `RAG CONTEXT` or conversation history, state that you cannot answer the question.
"""

# TODO:
# Provide structured system prompt, with RAG Context and User Question sections.
USER_PROMPT = """##RAG CONTEXT:
{context}


##USER QUESTION: 
{query}"""

embeddings_client = DialEmbeddingsClient(deployment_name='text-embedding-3-small-1')

text_processor = TextProcessor(
    embeddings_client=embeddings_client,
    db_config={'host': 'localhost', 'port': 5433, 'database': 'vectordb', 'user': 'postgres', 'password': 'postgres'}
)
text_processor.process_text_file(file_name='embeddings/microwave_manual.txt', chunk_size=400, overlap=50, dimensions=DIMENSIONS, truncate_table=True)

def run():
    chat_completion_client = DialChatCompletionClient(deployment_name='gpt-4o', api_key=API_KEY)

    user_input = input("Enter your query: ").strip()
    while user_input.lower() != 'exit':
        context_chunks = text_processor.search(
            search_mode=SearchMode.COSINE_DISTANCE,
            user_request=user_input,
            top_k=4,
            score_threshold=0.5,
            dimensions=DIMENSIONS
        )
        context = "\n".join(context_chunks)

        augmented_prompt = USER_PROMPT.format(context=context, query=user_input)

        messages = [
            Message(Role.SYSTEM, SYSTEM_PROMPT),
            Message(Role.USER, augmented_prompt)
        ]
        response_message = chat_completion_client.get_completion(messages)
        print(f"AI: {response_message.content}\n")

        user_input = input("Enter your query: ").strip()

run()
# TODO:
# - create embeddings client with 'text-embedding-3-small-1' model
# - create chat completion client
# - create text processor, DB config: {'host': 'localhost','port': 5433,'database': 'vectordb','user': 'postgres','password': 'postgres'}
# ---
# Create method that will run console chat with such steps:
# - get user input from console
# - retrieve context
# - perform augmentation
# - perform generation
# - it should run in `while` loop (since it is console chat)


# TODO:
#  PAY ATTENTION THAT YOU NEED TO RUN Postgres DB ON THE 5433 WITH PGVECTOR EXTENSION!
#  RUN docker-compose.yml
