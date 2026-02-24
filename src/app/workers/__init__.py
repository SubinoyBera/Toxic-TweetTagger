# 
import time
import threading
from queue import Queue, Empty, Full
from src.core.logger import logging 

class BufferedEventConsumerWorker:
    def __init__(self, mongo_client, database_name, collection_name, queue_maxsize: int = 1200, 
                 max_batch_size: int = 1000, flush_interval: int = 30, mongo_timeout: int = 5):
        """
        Initializes the buffered event consumer worker.

        Args:
            mongo_client (MongoClient): The MongoDB client.
            database_name (str): The name of the MongoDB database.
            collection_name (str): The name of the MongoDB collection.
            queue_maxsize (int, optional): The maximum size of the queue. Defaults to 1200.
            max_batch_size (int, optional): The maximum batch size for write operations. Defaults to 1000.
            flush_interval (int, optional): The interval (in seconds) at which to flush the queue. Defaults to 30.
            mongo_timeout (int, optional): The timeout (in seconds) for write operations. Defaults to 5.
        """
        self.queue = Queue(queue_maxsize)
        self.max_batch_size = max_batch_size                
        self.flush_interval = flush_interval
        self.mongo_timeout = mongo_timeout

        self.shutdown_event = threading.Event()

        self.client = mongo_client 
        self.database_name = database_name
        self.collection_name = collection_name

        self.worker = threading.Thread(
            target=self._writer_worker,
            daemon=True
            )
        self.worker.start()

    def add_event(self, record: dict) -> None:  
        """
        Adds a record to the buffer queue.

        Args:
            record (dict): The record to add to the buffer queue.
        """
        try:
            self.queue.put(record, timeout=0.3)
        except Full:
            logging.warning(f"Failed to add record in buffer queue: Queue is full")

    def shutdown(self) -> None:
        """
        Gracefully shut down the worker thread.
        Ensures final flush before exit.
        """
        self.shutdown_event.set()

        # Wake up worker if it's blocked on queue.get()
        self.queue.put(None)
        # Wait the main thread until worker fully exits.
        self.worker.join()


    # INTERNAL WORKER
    def _writer_worker(self) -> None:
        """
        The worker thread responsible for consuming records from the buffer queue and writing them to MongoDB.

        It runs indefinitely until the shutdown event is set, at which point it will drain the queue quickly and exit.
        The worker thread tries to flush the queue at regular intervals, or when the batch size reaches the maximum threshold.
        If the queue is empty, it will wait indefinitely for new records to arrive. If a timeout is reached, it will flush the queue and reset the timer.
        """
        batch = []
        first_record_time = None

        while not self.shutdown_event.is_set():
            try:
                if first_record_time is None:
                    # No records yet → wait indefinitely
                    record = self.queue.get()
                else:
                    elapsed = time.time() - first_record_time
                    remaining = max(self.flush_interval - elapsed, 0)
                    record = self.queue.get(timeout=remaining)
                
                if record is None:
                    break
                
                batch.append(record)

                if first_record_time is None:
                    first_record_time = time.time()

                # Drain quickly if batch growing
                while len(batch) < self.max_batch_size:
                    try:
                        record = self.queue.get_nowait()
                        if record is None:
                            break
                        batch.append(record)
                    except Empty:
                        break

            except Empty:
                # Timeout reached
                pass

            # Flush conditions
            if batch and (
                len(batch) >= self.max_batch_size or
                (first_record_time and
                 time.time() - first_record_time >= self.flush_interval)
            ):
                self._flush(batch)
                batch.clear()
                first_record_time = None

        # Final flush on shutdown
            if batch:
                self._flush(batch)
            logging.info("BufferedEventConsumer worker stopped cleanly.")

    # Database flush
    def _flush(self, batch_records: list):
        """
        Writes batch records to MongoDB with basic failure handling.

        Args:
            batch_records (list): The list of records to flush to the database.

        Raises:
            Exception: If an error occurs while flushing the batch records to the database.
        """
        try:
            self.client.insert_docs(self.collection_name, 
                                    self.database_name, 
                                    batch_records,
                                    timeout = self.mongo_timeout
                                )
            logging.info(f"Flushed {len(batch_records)} records to MongoDB")
        
        except Exception as e:
            logging.error(f"BufferedBatchWriter failed to flush: {e}", exc_info=True)