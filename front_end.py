import logging
from SimulatedUser import SimulatedUser

from conf import *
from Deploy import *

start_logs()
check_model_requirements(model=MODEL)

logging.info('Input type: %s' % (INPUT_TYPE,))
logging.info('Model: %s' % (MODEL,))
logging.info('File path (for video): %s' % (FILE_PATH,))
logging.info('Output path (for video): %s' % (OUTPUT_PATH,))

the_user = SimulatedUser(INPUT_TYPE, MODEL, FILE_PATH, OUTPUT_PATH)
the_user.use_lots()
