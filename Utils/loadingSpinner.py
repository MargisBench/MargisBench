from logging import config, getLogger
from logging_config import TEST_LOGGING_CONFIG
config.dictConfig(TEST_LOGGING_CONFIG)
logger = getLogger(__name__)

import threading
import time
import sys
import itertools
from typing import Dict, List, Any, Optional, Tuple, Union


BLUE = "\x1b[34m"
RESET = "\x1b[0m"

class LoadingSpinner:
    
    def __init__(self, message: Optional[str]="Processing", delay:Optional[int]=0.5):
        """
        Creates the LoadingSpinner object. This is useful to show a little animation during subprocessing 
        with silenced output. 

        Parameters
        ----------
        - message: str
        The message to print before the loading animation.
        - delay: int
        The delays between the printing of the dots.


        Returns
        -------
        - None
        """

        self.message = message
        self.delay = delay
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._animate)

    def _animate(self) -> None:
        """
        Utility function for dots animation.
        

    
        """
        
        for dots in itertools.cycle(['.  ', '.. ', '...', '   ']):
            if self.stop_event.is_set():
                break
            # \r moves cursor to start of line, end='' prevents new line
            formatted_line = f"\r[{BLUE}INFO{RESET}] {self.message} {dots}"
            sys.stdout.write(formatted_line)
            sys.stdout.flush()
            time.sleep(self.delay)

    def __enter__(self):
        """
        Enter function of the thread. 
        """
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit function of the thread. 
        """
        self.stop_event.set()
        self.thread.join()
        # Clear the line after finishing
        sys.stdout.write(f"\r[{BLUE}INFO{RESET}] {self.message}... DONE!   \n")
        sys.stdout.flush()