import functools
import numpy as np

LINE_LENGTH = 79
PADDING_CHARACTER = '='
TERMINATE_CHARACTER = '|'

ICON = """
       _            _  _  _        _  _  _  _    _           _ 
     _(_)_       _ (_)(_)(_) _   _(_)(_)(_)(_)_ (_)_       _(_)
   _(_) (_)_    (_)         (_) (_)          (_)  (_)_   _(_)  
 _(_)     (_)_  (_)    _  _  _  (_)          (_)    (_)_(_)    
(_) _  _  _ (_) (_)   (_)(_)(_) (_)          (_)     _(_)_     
(_)(_)(_)(_)(_) (_)         (_) (_)          (_)   _(_) (_)_   
(_)         (_) (_) _  _  _ (_) (_)_  _  _  _(_) _(_)     (_)_ 
(_)         (_)    (_)(_)(_)(_)   (_)(_)(_)(_)  (_)         (_)  v{} \n
"""

def find_spaces(string):
    return np.array([i for i in range(len(string)) if string.startswith(' ', i)])

def line_breaker(string):
    all_strings = []
    total_string = string
    if len(total_string) > LINE_LENGTH:
        safety_counter = 0
        while len(total_string) > LINE_LENGTH and safety_counter < 100:
            safety_counter += 1
            spaces = find_spaces(total_string)
            break_index = spaces[spaces < LINE_LENGTH]
            if len(break_index) == 0:
                all_strings += total_string
                break
            break_index = np.max(spaces)
            all_strings.append(total_string[0:break_index])
            total_string = total_string[break_index:]

        all_strings.append(total_string)
    else:
        all_strings.append(total_string)
    return all_strings
        
def header_print(string):
    
    string = ' ' + string + ' '
    num_markers = int((LINE_LENGTH - len(string))/2) - 1
    header_string = TERMINATE_CHARACTER + num_markers * PADDING_CHARACTER + string + num_markers * PADDING_CHARACTER + TERMINATE_CHARACTER

    if len(header_string) < LINE_LENGTH:
        header_string = header_string[:-2] + PADDING_CHARACTER*2 + TERMINATE_CHARACTER        

    print(header_string)

def pretty_print(string, *args, **kwargs):
    string = str(string)
    for arg in args:
        string += str(arg)

    all_strings = line_breaker(string)
    for string in all_strings:
        string = TERMINATE_CHARACTER + ' ' + string + (LINE_LENGTH - len(string)-3) * ' ' + TERMINATE_CHARACTER       
        print(string, **kwargs)

def agox_writer(func):
    @functools.wraps(func)
    def wrapper(self, state, *args, **kwargs):
        if not self.use_counter:
            header_print(self.name)
        state = func(self, state, *args, **kwargs)
        if self.use_counter and len(self.lines_to_print) > 0:
            header_print(self.name)
            for string, args, kwargs in self.lines_to_print:
                string = self.writer_prefix + str(string)
                pretty_print(string, *args, **kwargs)
        self.lines_to_print = []
        
    return wrapper

class Writer:

    def __init__(self, verbose=True, use_counter=True, prefix=''):
        self.verbose = verbose
        self.use_counter = use_counter
        self.lines_to_print = []
        self.writer_prefix = prefix

    def writer(self, string, *args, **kwargs):
        if self.verbose:
            if self.use_counter:
                self.lines_to_print.append((string, args, kwargs))
            else:
                string = self.writer_prefix + str(string)
                pretty_print(string, *args, **kwargs)

    def header_print(self, string):
        if self.verbose:
            header_print(string)