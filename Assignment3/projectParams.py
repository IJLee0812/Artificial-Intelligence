# projectParams.py
# ----------------
'''
Copyright (C) Computer Science & Engineering, Soongsil University. This material is for educational uses only. 
Some contents are based on the material provided by other paper/book authors and may be copyrighted by them. 
Written by Haneul Pyeon, October 2024.
'''



STUDENT_CODE_DEFAULT = 'analysis.py,searchAgents.py,search.py'
PROJECT_TEST_CLASSES = 'searchTestClasses.py'
PROJECT_NAME = 'Assignment 3: Search'
BONUS_PIC = False
ERROR_HINT_MAP = {
  'q1': {
    "<type 'exceptions.IndexError'>": """
      We noticed that your project threw an IndexError on q1.
      While many things may cause this, it may have been from
      assuming a certain number of successors from a state space
      or assuming a certain number of actions available from a given
      state. Try making your code more general (no hardcoded indices)
      and submit again!
    """
  },
  'q3': {
      "<type 'exceptions.AttributeError'>": """
        We noticed that your project threw an AttributeError on q3.
        While many things may cause this, it may have been from assuming
        a certain size or structure to the state space. For example, if you have
        a line of code assuming that the state is (x, y) and we run your code
        on a state space with (x, y, z), this error could be thrown. Try
        making your code more general and submit again!

    """
  }
}
