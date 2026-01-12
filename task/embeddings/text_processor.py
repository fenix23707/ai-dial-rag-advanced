from enum import StrEnum

import psycopg2
from psycopg2.extras import RealDictCursor

from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.utils.text import chunk_text


class SearchMode(StrEnum):
    EUCLIDIAN_DISTANCE = "euclidean"  # Euclidean distance (<->)
    COSINE_DISTANCE = "cosine"  # Cosine distance (<=>)


class TextProcessor:
    """Processor for text documents that handles chunking, embedding, storing, and retrieval"""

    def __init__(self, embeddings_client: DialEmbeddingsClient, db_config: dict):
        self.embeddings_client = embeddings_client
        self.db_config = db_config

    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(
            host=self.db_config['host'],
            port=self.db_config['port'],
            database=self.db_config['database'],
            user=self.db_config['user'],
            password=self.db_config['password']
        )

    def process_text_file(self, file_name:str, chunk_size:int, overlap:int, dimensions:int, truncate_table:bool):
        with open(file_name, 'r', encoding='utf-8') as f:
            chunks = chunk_text(text=f.read(), chunk_size=chunk_size, overlap=overlap)

        with self._get_connection() as connection:
            with connection.cursor() as cursor:
                if truncate_table:
                    cursor.execute('TRUNCATE TABLE vectors;')

                embeddings_dict = self.embeddings_client.get_embeddings(chunks, dimensions)
                data = [
                    (file_name, chunks[index], self._to_vector_string(embeddings_dict[index])) for index in range(len(chunks))
                ]
                cursor.executemany( 'INSERT INTO vectors (document_name, text, embedding) VALUES (%s, %s, %s::vector);', data)

    #TODO:
    # provide method `process_text_file` that will:
    #   - apply file name, chunk size, overlap, dimensions and bool of the table should be truncated
    #   - truncate table with vectors if needed
    #   - load content from file and generate chunks (in `utils.text` present `chunk_text` that will help do that)
    #   - generate embeddings from chunks
    #   - save (insert) embeddings and chunks to DB
    #       hint 1: embeddings should be saved as string list
    #       hint 2: embeddings string list should be casted to vector ({embeddings}::vector)


    def search(self, search_mode: SearchMode, user_request: str, top_k: int, score_threshold: float, dimensions: int) -> list[str]:
        embeddings_dict = self.embeddings_client.get_embeddings(input_list=[user_request],dimensions=dimensions)

        mode_to_operator = {
            SearchMode.EUCLIDIAN_DISTANCE: '<->',
            SearchMode.COSINE_DISTANCE: '<=>'
        }
        vector_string = self._to_vector_string(embeddings_dict[0])
        mode = mode_to_operator.get(search_mode)
        if search_mode == SearchMode.COSINE_DISTANCE:
            max_distance = 1.0 - score_threshold
        else:
            max_distance = float('inf') if score_threshold == 0 else (1.0 / score_threshold) - 1.0

        sql = f"""
        SELECT text, embedding {mode} %s::vector AS distance
        FROM vectors
        WHERE embedding {mode} %s::vector <= %s
        ORDER BY distance
        LIMIT %s;
        """

        with self._get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql, (vector_string, vector_string, max_distance, top_k))
                results = cursor.fetchall()
                return [row[0] for row in results]

    def _to_vector_string(self, vector: list[float]) -> str:
        return '[' + ','.join(map(str, vector)) + ']'
    #TODO:
    # provide method `search` that will:
    #   - apply search mode, user request, top k for search, min score threshold and dimensions
    #   - generate embeddings from user request
    #   - search in DB relevant context
    #     hint 1: to search it in DB you need to create just regular select query
    #     hint 2: Euclidean distance `<->`, Cosine distance `<=>`
    #     hint 3: You need to extract `text` from `vectors` table
    #     hint 4: You need to filter distance in WHERE clause
    #     hint 5: To get top k use `limit`

