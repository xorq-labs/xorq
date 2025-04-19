from subprocess import (
    PIPE,
    Popen,
)


def subprocess_run(args, do_decode=False):
    popened = Popen(args, stdout=PIPE, stderr=PIPE)
    (stdout, stderr) = popened.communicate()
    if do_decode:
        (stdout, stderr) = (el.decode("ascii") for el in popened.communicate())
    return (popened.returncode, stdout, stderr)
