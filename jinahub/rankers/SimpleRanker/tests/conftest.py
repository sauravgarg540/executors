import random
import pytest

from jina import DocumentArray, Document


@pytest.fixture
def documents_chunk():
    document_array = DocumentArray()
    document = Document(tags={'query_size': 35, 'query_price': 31, 'query_brand': 1})
    for i in range(0, 10):
        chunk = Document()
        for j in range(0, 10):
            match = Document(
                tags={
                    'level': 'chunk',
                }
            )
            match.scores['cosine'] = random.random()
            match.parent_id = i
            chunk.matches.append(match)
        document.chunks.append(chunk)

    document_array.extend([document])
    return document_array


@pytest.fixture
def documents_chunk_chunk():
    document_array = DocumentArray()
    document = Document(tags={'query_size': 35, 'query_price': 31, 'query_brand': 1})
    for i in range(0, 10):
        chunk = Document()
        for j in range(0, 10):
            chunk_chunk = Document()
            for k in range(0, 10):

                match = Document(
                    tags={
                        'level': 'chunk',
                    }
                )
                match.scores['cosine'] = random.random()
                match.parent_id = j
                chunk_chunk.matches.append(match)
            chunk.chunks.append(chunk_chunk)
        document.chunks.append(chunk)

    document_array.extend([document])
    return document_array
