import time
import threading
from queue import Queue, Empty, Full
from src.constant.constants import DATABASE_NAME, PRODUCTION_COLLECTION_NAME
from src.core.logger import logging 

class BufferedBatchWriter:
    def __init__(self, mongo_client):
        """
        Initializes the buffered batch writer.
        The buffered batch writer will write records to the MongoDB collection in batches.

        The worker thread will flush the batch to the collection when the batch reaches its max size or when the flush
        interval has elapsed since the last flush.
        """
        self.queue = Queue(maxsize=500)
        self.max_batch_size = 200                 
        self.flush_interval = 30
        self.shutdown_flag = threading.Event()

        self.client = mongo_client 
        self.database_name = DATABASE_NAME
        self.collection_name = PRODUCTION_COLLECTION_NAME

        self.worker = threading.Thread(target=self._writer_worker, daemon=True)
        self.worker.start()


    def add(self, record: dict):                
        """
        Adds a record to the buffered batch writer.

        Args:
            record (dict): The record to be added to the batch.
        """
        try:
            self.queue.put_nowait(record)
        except Full:
            logging.warning(f"Failed to add record in buffer queue: Queue is full")

    def _writer_worker(self):
        """
        Writer worker thread that reads from the queue and flushes (writes) batch records to the database.
        The thread will break if the shutdown flag is set.

        The thread will flush the batch to the database when the batch reaches its max size or when the flush 
        interval has elapsed since the last flush.

        The thread will perform a final flush on shutdown if there are any records left in the batch.
        """
        batch = []
        last_flush_time = None

        while not self.shutdown_flag.is_set():
            try:
                # Wait for at least 1 item
                record = self.queue.get(timeout=1)
                batch.append(record)

                if last_flush_time is None:
                    last_flush_time = time.time()
            
            except Empty:
                pass
            
            # Drain the queue when we have at least one record to flush. If queue is found empty, break
            while len(batch) < self.max_batch_size:
                try:
                    record = self.queue.get_nowait()
                    batch.append(record)
                
                except Empty:
                    break

            current_time = time.time()

            # Flush conditions
            if (
                len(batch) >= self.max_batch_size or
                (last_flush_time and current_time - last_flush_time >= self.flush_interval)
            ):
                self._flush(batch)
                batch.clear()
                last_flush_time = None

        # Final flush on shutdown
        if batch:
            self._flush(batch)


    def _flush(self, batch_records: list):
        """
        Flushes (writes) the given batch of records to the MongoDB database collection.

        Args:
            batch_records (list): A list of records to be flushed to the MongoDB collection.
        """
        try:
            self.client.insert_docs(self.collection_name, self.database_name, batch_records)
            logging.info(f"Flushed {len(batch_records)} records to MongoDB")
        
        except Exception as e:
            logging.warning(f"BufferedBatchWriter failed to flush: {e}")


    def shutdown(self):
        """
        Shuts down the worker thread and waits for it to finish.

        This method should be called when the application is shutting down to ensure
        that all records are flushed to the database.
        """
        self.shutdown_flag.set()
        # Wait the main thread until worker fully exits.
        self.worker.join()
