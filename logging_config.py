import logging
import logging.config

class ColoredFormatter(logging.Formatter):
    """
    This is a class definition ColoredFormatter, a custom formatted for the
    Logger instance. 
    It allows to create a custom Logger which each different level has a different
    reference color.
    """
    RESET = "\x1b[0m"
    RED = "\x1b[31m"
    YELLOW = "\x1b[33m"
    GREEN = "\x1b[32m"
    BLUE = "\x1b[34m"
    MAGENTA = "\x1b[35m"

    LEVEL_COLORS = {
        logging.DEBUG: MAGENTA,
        logging.INFO: BLUE,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: RED,
    }

    def format(self, record):
        original_levelname = record.levelname
        color = self.LEVEL_COLORS.get(record.levelno, self.RESET)
        record.levelname = f"{color}{original_levelname}{self.RESET}"   
        formatted_message = super().format(record)
        record.levelname = original_levelname

        return formatted_message

#CUSTOM LOGGER CONFIGURATION...
TEST_LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    
    # --- DEFINE FORMATS ---
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
        'simple': {
            '()': ColoredFormatter,
            'format': '[%(levelname)s] %(message)s'
        },
    },
    
    # --- DEFINE HANDLERS (Destinations) ---
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
            'stream': 'ext://sys.stdout',
        },
        'test_file': {
            'class': 'logging.FileHandler',
            'formatter': 'standard',
            'filename': 'BenchProtoLogFile.log',
            'mode': 'w', 
        },
    },
    
    # --- CONFIGURE LOGGERS ---
    'loggers': {
        
        'BenchmarkingFactory': {
            'level': 'INFO',
            'handlers': ['console', 'test_file'],
            'propagate': False 
        },

        'main': {  
        'level': 'INFO',
        'handlers': ['console', 'test_file'],
        'propagate': False
        },

        'BenchmarkingFactory.dataWrapper': {  
        'level': 'INFO',
        'handlers': ['console', 'test_file'],
        'propagate': False
        },

        'BenchmarkingFactory.aiModel': {
        'level': 'INFO',
        'handlers': ['console', 'test_file'],
        'propagate': False
        },

        'BenchmarkingFactory.doe': {
        'level': 'INFO',
        'handlers': ['console', 'test_file'],
        'propagate': False
        },

        'ConfigurationModule.configurationManager': {
        'level': 'INFO',
        'handlers': ['console', 'test_file'],
        'propagate': False
        },

        'ProbeHardwareModule.probeHardwareManager': {
        'level': 'INFO',
        'handlers': ['console', 'test_file'],
        'propagate': False
        },

        'PackageDownloadModule.packageDownloadManager': {
        'level': 'INFO',
        'handlers': ['console', 'test_file'],
        'propagate': False
        },

        'Utils.calculateStats': {
        'level': 'INFO',
        'handlers': ['console', 'test_file'],
        'propagate': False
        },

        'Utils.utilsFunctions': {
        'level': 'INFO',
        'handlers': ['console', 'test_file'],
        'propagate': False
        },


        # 2. Silencing onnxruntime
        'onnxruntime': {
            'level': 'ERROR', # Only show ERRORs or higher
            'handlers': ['console', 'test_file'],
            'propagate': False # Stop logs from going to root
        },
        
        # 3. Silencing onnxscript
        'onnxscript': {
            'level': 'ERROR', # Only show ERRORs or higher
            'handlers': ['console', 'test_file'],
            'propagate': False # Stop logs from going to root
        }
    },
    
    # --- THE DEFAULT (ROOT) LOGGER ---
    'root': {
        'level': 'INFO', # Default for all other libraries
        'handlers': ['console', 'test_file']
    }
}


