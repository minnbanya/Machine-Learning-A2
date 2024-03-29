"""
This type stub file was generated by pyright.
"""

"""Hooks for IPython.

In Python, it is possible to overwrite any method of any object if you really
want to.  But IPython exposes a few 'hooks', methods which are *designed* to
be overwritten by users for customization purposes.  This module defines the
default versions of all such hooks, which get used by IPython if not
overridden by the user.

Hooks are simple functions, but they should be declared with ``self`` as their
first argument, because when activated they are registered into IPython as
instance methods. The self argument will be the IPython running instance
itself, so hooks have full access to the entire IPython object.

If you wish to define a new hook and activate it, you can make an :doc:`extension
</config/extensions/index>` or a :ref:`startup script <startup_files>`. For
example, you could use a startup file like this::

    import os

    def calljed(self,filename, linenum):
        "My editor hook calls the jed editor directly."
        print "Calling my own editor, jed ..."
        if os.system('jed +%d %s' % (linenum,filename)) != 0:
            raise TryNext()

    def load_ipython_extension(ip):
        ip.set_hook('editor', calljed)

"""
__all__ = ["editor", "synchronize_with_editor", "show_in_pager", "pre_prompt_hook", "clipboard_get"]
deprecated = ...
def editor(self, filename, linenum=..., wait=...): # -> None:
    """Open the default editor at the given filename and linenumber.

    This is IPython's default editor hook, you can use it as an example to
    write your own modified one.  To set your own editor function as the
    new editor hook, call ip.set_hook('editor',yourfunc)."""
    ...

def synchronize_with_editor(self, filename, linenum, column): # -> None:
    ...

class CommandChainDispatcher:
    """ Dispatch calls to a chain of commands until some func can handle it

    Usage: instantiate, execute "add" to add commands (with optional
    priority), execute normally via f() calling mechanism.

    """
    def __init__(self, commands=...) -> None:
        ...
    
    def __call__(self, *args, **kw):
        """ Command chain is called just like normal func.

        This will call all funcs in chain with the same args as were given to
        this function, and return the result of first func that didn't raise
        TryNext"""
        ...
    
    def __str__(self) -> str:
        ...
    
    def add(self, func, priority=...): # -> None:
        """ Add a func to the cmd chain with given priority """
        ...
    
    def __iter__(self):
        """ Return all objects in chain.

        Handy if the objects are not callable.
        """
        ...
    


def show_in_pager(self, data, start, screen_lines):
    """ Run a string through pager """
    ...

def pre_prompt_hook(self): # -> None:
    """ Run before displaying the next prompt

    Use this e.g. to display output from asynchronous operations (in order
    to not mess up text entry)
    """
    ...

def clipboard_get(self):
    """ Get text from the clipboard.
    """
    ...

