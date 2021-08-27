__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import copy
import functools
from typing import Dict, Optional

from jina import DocumentArray, Executor, requests
from jina_commons import get_logger

try:
    from jinahub.indexers.searcher.FaissSearcher import FaissSearcher
except:  # noqa: E722
    from jina_executors.indexers.searcher.FaissSearcher.faiss_searcher import (
        FaissSearcher,
    )

try:
    from jinahub.indexers.storage.PostgreSQLStorage import PostgreSQLStorage
except:  # noqa: E722
    from jina_executors.indexers.storage.PostgreSQLStorage import PostgreSQLStorage


class FaissPostgresSearcher(Executor):
    """A Compound Indexer made up of a FaissSearcher (for vectors) and a Postgres
    Indexer

    :param dump_path: a path to a dump folder containing
    the dump data obtained by calling jina_commons.dump_docs
    :param use_dump_func: whether to use the dump
     function of PostgreSQLStorage, when dump_path is not provided
    """

    def __init__(
        self,
        dump_path: Optional[str] = None,
        use_dump_func: bool = False,
        total_shards: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.logger = get_logger(self)

        self.total_shards = total_shards
        if self.total_shards is None:
            self.logger.warning(
                'total_shards is None, rolling update '
                'via PSQL import will not be possible.'
            )

        # when constructed from rolling update the dump_path is passed via a
        # runtime_arg
        dump_path = dump_path or kwargs.get('runtime_args').get('dump_path')
        use_dump_func = use_dump_func or kwargs.get('runtime_args').get('use_dump_func')

        self._kv_indexer = None
        self._vec_indexer = None
        self._init_kwargs = kwargs

        if dump_path is None and use_dump_func is None:
            name = getattr(self.metas, 'name', self.__class__.__name__)
            self.logger.warning(
                f'No "dump_path" or "use_dump_func" provided '
                f'for {name}. Use .rolling_update() to re-initialize...'
            )

        if use_dump_func:
            self._kv_indexer = PostgreSQLStorage(**kwargs)
            dump_func = self._kv_indexer.get_snapshot
            self._vec_indexer = FaissSearcher(dump_func=dump_func, **kwargs)
        else:
            self._kv_indexer = PostgreSQLStorage(**kwargs)
            self._vec_indexer = FaissSearcher(dump_path=dump_path, **kwargs)

    @requests(on='/reload')
    def reload(self, **kwargs):
        """
        Reload the Searchers in this Compound,
        using the get_snapshot method of the PostgreSQLStorage
        """
        if self.total_shards:
            dump_func = functools.partial(
                self._kv_indexer.get_snapshot, total_shards=self.total_shards
            )
            self._vec_indexer = FaissSearcher(dump_func=dump_func, **self._init_kwargs)
        else:
            self.logger.warning(
                'total_shards is None, rolling update '
                'via PSQL import will not be possible.'
            )

    @requests(on='/search')
    def search(self, docs: 'DocumentArray', parameters: Dict = None, **kwargs):
        if self._kv_indexer and self._vec_indexer:
            self._vec_indexer.search(docs, parameters)
            kv_parameters = copy.deepcopy(parameters)
            kv_parameters['traversal_paths'] = [
                path + 'm' for path in kv_parameters.get('traversal_paths', ['r'])
            ]
            self._kv_indexer.search(docs, kv_parameters)
        else:
            return
