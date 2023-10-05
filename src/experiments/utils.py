


def in_interactive_session():
    try:
        get_ipython
        return True
    except NameError:
        return False
    
