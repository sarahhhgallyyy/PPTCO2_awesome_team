#The cooling process can be split up in two sections.
#The first part is the cooling down part, where the reactor is cooled down from room temp.
#The second part is the steady state part, where the reactor is at a low constant temp.

#Import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

#Define the path to the data file

