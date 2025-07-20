# Import necessary classes and functions required in every component

import sys
import gc
import pandas as pd
from pathlib import Path
from core.logger.logging import logging
from core.exception.exceptions import AppException
from core.configuration.config import AppConfiguration
from utils.common import *