import signal
from typing import Callable, Optional


def wait_user_input(
    valid_input: Callable[[str], bool],
    prompt: str = '',
    default: str = '',
    continue_after: Optional[int] = None,
) -> str:
    # valid_input: check if user input is valid
    # default: default value if user input is empty
    # continue_after: if not None, pause for a while before return the default value
    # return: the user's input

    class TimeoutExpired(Exception):
        pass

    def timeout_handler(*args, **kwargs):
        raise TimeoutExpired

    if continue_after is not None:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(continue_after))

    try:
        while not valid_input(user_input := input(prompt)):
            print('Invalid input') 
    except TimeoutExpired:
        user_input = ''

    if user_input == '':
        user_input = default
    return user_input