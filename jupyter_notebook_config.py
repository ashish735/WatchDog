# @Author: abhijit
# @Date:   2018-08-19T20:47:30+05:30
# @Last modified by:   abhijit
# @Last modified time: 2018-09-15T00:17:14+05:30

import os
from IPython.lib import passwd

c.NotebookApp.ip = '*'
c.NotebookApp.port = 8888
c.NotebookApp.open_browser = False

# sets a password if PASSWORD is set in the environment
c.NotebookApp.password = passwd('deep_learning')
