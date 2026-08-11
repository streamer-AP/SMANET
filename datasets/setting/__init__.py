import sys

from . import CroHD
from . import MovingDroneCrowd
from . import SENSE

HT21 = CroHD
MDC = MovingDroneCrowd
SenseCrowd = SENSE

sys.modules[__name__ + ".HT21"] = CroHD
sys.modules[__name__ + ".MDC"] = MovingDroneCrowd
sys.modules[__name__ + ".SenseCrowd"] = SENSE
