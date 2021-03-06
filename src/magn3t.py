# This file was automatically generated by SWIG (http://www.swig.org).
# Version 3.0.12
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
if _swig_python_version_info >= (2, 7, 0):
    def swig_import_helper():
        import importlib
        pkg = __name__.rpartition('.')[0]
        mname = '.'.join((pkg, '_magn3t')).lstrip('.')
        try:
            return importlib.import_module(mname)
        except ImportError:
            return importlib.import_module('_magn3t')
    _magn3t = swig_import_helper()
    del swig_import_helper
elif _swig_python_version_info >= (2, 6, 0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_magn3t', [dirname(__file__)])
        except ImportError:
            import _magn3t
            return _magn3t
        try:
            _mod = imp.load_module('_magn3t', fp, pathname, description)
        finally:
            if fp is not None:
                fp.close()
        return _mod
    _magn3t = swig_import_helper()
    del swig_import_helper
else:
    import _magn3t
del _swig_python_version_info

try:
    _swig_property = property
except NameError:
    pass  # Python < 2.2 doesn't have 'property'.

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_setattr_nondynamic(self, class_type, name, value, static=1):
    if (name == "thisown"):
        return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name, None)
    if method:
        return method(self, value)
    if (not static):
        if _newclass:
            object.__setattr__(self, name, value)
        else:
            self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)


def _swig_setattr(self, class_type, name, value):
    return _swig_setattr_nondynamic(self, class_type, name, value, 0)


def _swig_getattr(self, class_type, name):
    if (name == "thisown"):
        return self.this.own()
    method = class_type.__swig_getmethods__.get(name, None)
    if method:
        return method(self)
    raise AttributeError("'%s' object has no attribute '%s'" % (class_type.__name__, name))


def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except __builtin__.Exception:
    class _object:
        pass
    _newclass = 0

class SwigPyIterator(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, SwigPyIterator, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, SwigPyIterator, name)

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _magn3t.delete_SwigPyIterator
    __del__ = lambda self: None

    def value(self):
        return _magn3t.SwigPyIterator_value(self)

    def incr(self, n=1):
        return _magn3t.SwigPyIterator_incr(self, n)

    def decr(self, n=1):
        return _magn3t.SwigPyIterator_decr(self, n)

    def distance(self, x):
        return _magn3t.SwigPyIterator_distance(self, x)

    def equal(self, x):
        return _magn3t.SwigPyIterator_equal(self, x)

    def copy(self):
        return _magn3t.SwigPyIterator_copy(self)

    def next(self):
        return _magn3t.SwigPyIterator_next(self)

    def __next__(self):
        return _magn3t.SwigPyIterator___next__(self)

    def previous(self):
        return _magn3t.SwigPyIterator_previous(self)

    def advance(self, n):
        return _magn3t.SwigPyIterator_advance(self, n)

    def __eq__(self, x):
        return _magn3t.SwigPyIterator___eq__(self, x)

    def __ne__(self, x):
        return _magn3t.SwigPyIterator___ne__(self, x)

    def __iadd__(self, n):
        return _magn3t.SwigPyIterator___iadd__(self, n)

    def __isub__(self, n):
        return _magn3t.SwigPyIterator___isub__(self, n)

    def __add__(self, n):
        return _magn3t.SwigPyIterator___add__(self, n)

    def __sub__(self, *args):
        return _magn3t.SwigPyIterator___sub__(self, *args)
    def __iter__(self):
        return self
SwigPyIterator_swigregister = _magn3t.SwigPyIterator_swigregister
SwigPyIterator_swigregister(SwigPyIterator)

class IntVector(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, IntVector, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, IntVector, name)
    __repr__ = _swig_repr

    def iterator(self):
        return _magn3t.IntVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _magn3t.IntVector___nonzero__(self)

    def __bool__(self):
        return _magn3t.IntVector___bool__(self)

    def __len__(self):
        return _magn3t.IntVector___len__(self)

    def __getslice__(self, i, j):
        return _magn3t.IntVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _magn3t.IntVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _magn3t.IntVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _magn3t.IntVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _magn3t.IntVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _magn3t.IntVector___setitem__(self, *args)

    def pop(self):
        return _magn3t.IntVector_pop(self)

    def append(self, x):
        return _magn3t.IntVector_append(self, x)

    def empty(self):
        return _magn3t.IntVector_empty(self)

    def size(self):
        return _magn3t.IntVector_size(self)

    def swap(self, v):
        return _magn3t.IntVector_swap(self, v)

    def begin(self):
        return _magn3t.IntVector_begin(self)

    def end(self):
        return _magn3t.IntVector_end(self)

    def rbegin(self):
        return _magn3t.IntVector_rbegin(self)

    def rend(self):
        return _magn3t.IntVector_rend(self)

    def clear(self):
        return _magn3t.IntVector_clear(self)

    def get_allocator(self):
        return _magn3t.IntVector_get_allocator(self)

    def pop_back(self):
        return _magn3t.IntVector_pop_back(self)

    def erase(self, *args):
        return _magn3t.IntVector_erase(self, *args)

    def __init__(self, *args):
        this = _magn3t.new_IntVector(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def push_back(self, x):
        return _magn3t.IntVector_push_back(self, x)

    def front(self):
        return _magn3t.IntVector_front(self)

    def back(self):
        return _magn3t.IntVector_back(self)

    def assign(self, n, x):
        return _magn3t.IntVector_assign(self, n, x)

    def resize(self, *args):
        return _magn3t.IntVector_resize(self, *args)

    def insert(self, *args):
        return _magn3t.IntVector_insert(self, *args)

    def reserve(self, n):
        return _magn3t.IntVector_reserve(self, n)

    def capacity(self):
        return _magn3t.IntVector_capacity(self)
    __swig_destroy__ = _magn3t.delete_IntVector
    __del__ = lambda self: None
IntVector_swigregister = _magn3t.IntVector_swigregister
IntVector_swigregister(IntVector)

class DoubleVector(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, DoubleVector, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, DoubleVector, name)
    __repr__ = _swig_repr

    def iterator(self):
        return _magn3t.DoubleVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _magn3t.DoubleVector___nonzero__(self)

    def __bool__(self):
        return _magn3t.DoubleVector___bool__(self)

    def __len__(self):
        return _magn3t.DoubleVector___len__(self)

    def __getslice__(self, i, j):
        return _magn3t.DoubleVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _magn3t.DoubleVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _magn3t.DoubleVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _magn3t.DoubleVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _magn3t.DoubleVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _magn3t.DoubleVector___setitem__(self, *args)

    def pop(self):
        return _magn3t.DoubleVector_pop(self)

    def append(self, x):
        return _magn3t.DoubleVector_append(self, x)

    def empty(self):
        return _magn3t.DoubleVector_empty(self)

    def size(self):
        return _magn3t.DoubleVector_size(self)

    def swap(self, v):
        return _magn3t.DoubleVector_swap(self, v)

    def begin(self):
        return _magn3t.DoubleVector_begin(self)

    def end(self):
        return _magn3t.DoubleVector_end(self)

    def rbegin(self):
        return _magn3t.DoubleVector_rbegin(self)

    def rend(self):
        return _magn3t.DoubleVector_rend(self)

    def clear(self):
        return _magn3t.DoubleVector_clear(self)

    def get_allocator(self):
        return _magn3t.DoubleVector_get_allocator(self)

    def pop_back(self):
        return _magn3t.DoubleVector_pop_back(self)

    def erase(self, *args):
        return _magn3t.DoubleVector_erase(self, *args)

    def __init__(self, *args):
        this = _magn3t.new_DoubleVector(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def push_back(self, x):
        return _magn3t.DoubleVector_push_back(self, x)

    def front(self):
        return _magn3t.DoubleVector_front(self)

    def back(self):
        return _magn3t.DoubleVector_back(self)

    def assign(self, n, x):
        return _magn3t.DoubleVector_assign(self, n, x)

    def resize(self, *args):
        return _magn3t.DoubleVector_resize(self, *args)

    def insert(self, *args):
        return _magn3t.DoubleVector_insert(self, *args)

    def reserve(self, n):
        return _magn3t.DoubleVector_reserve(self, n)

    def capacity(self):
        return _magn3t.DoubleVector_capacity(self)
    __swig_destroy__ = _magn3t.delete_DoubleVector
    __del__ = lambda self: None
DoubleVector_swigregister = _magn3t.DoubleVector_swigregister
DoubleVector_swigregister(DoubleVector)

class ComplexVector(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, ComplexVector, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, ComplexVector, name)
    __repr__ = _swig_repr

    def iterator(self):
        return _magn3t.ComplexVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _magn3t.ComplexVector___nonzero__(self)

    def __bool__(self):
        return _magn3t.ComplexVector___bool__(self)

    def __len__(self):
        return _magn3t.ComplexVector___len__(self)

    def __getslice__(self, i, j):
        return _magn3t.ComplexVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _magn3t.ComplexVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _magn3t.ComplexVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _magn3t.ComplexVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _magn3t.ComplexVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _magn3t.ComplexVector___setitem__(self, *args)

    def pop(self):
        return _magn3t.ComplexVector_pop(self)

    def append(self, x):
        return _magn3t.ComplexVector_append(self, x)

    def empty(self):
        return _magn3t.ComplexVector_empty(self)

    def size(self):
        return _magn3t.ComplexVector_size(self)

    def swap(self, v):
        return _magn3t.ComplexVector_swap(self, v)

    def begin(self):
        return _magn3t.ComplexVector_begin(self)

    def end(self):
        return _magn3t.ComplexVector_end(self)

    def rbegin(self):
        return _magn3t.ComplexVector_rbegin(self)

    def rend(self):
        return _magn3t.ComplexVector_rend(self)

    def clear(self):
        return _magn3t.ComplexVector_clear(self)

    def get_allocator(self):
        return _magn3t.ComplexVector_get_allocator(self)

    def pop_back(self):
        return _magn3t.ComplexVector_pop_back(self)

    def erase(self, *args):
        return _magn3t.ComplexVector_erase(self, *args)

    def __init__(self, *args):
        this = _magn3t.new_ComplexVector(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def push_back(self, x):
        return _magn3t.ComplexVector_push_back(self, x)

    def front(self):
        return _magn3t.ComplexVector_front(self)

    def back(self):
        return _magn3t.ComplexVector_back(self)

    def assign(self, n, x):
        return _magn3t.ComplexVector_assign(self, n, x)

    def resize(self, *args):
        return _magn3t.ComplexVector_resize(self, *args)

    def insert(self, *args):
        return _magn3t.ComplexVector_insert(self, *args)

    def reserve(self, n):
        return _magn3t.ComplexVector_reserve(self, n)

    def capacity(self):
        return _magn3t.ComplexVector_capacity(self)
    __swig_destroy__ = _magn3t.delete_ComplexVector
    __del__ = lambda self: None
ComplexVector_swigregister = _magn3t.ComplexVector_swigregister
ComplexVector_swigregister(ComplexVector)

class StringVector(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, StringVector, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, StringVector, name)
    __repr__ = _swig_repr

    def iterator(self):
        return _magn3t.StringVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _magn3t.StringVector___nonzero__(self)

    def __bool__(self):
        return _magn3t.StringVector___bool__(self)

    def __len__(self):
        return _magn3t.StringVector___len__(self)

    def __getslice__(self, i, j):
        return _magn3t.StringVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _magn3t.StringVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _magn3t.StringVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _magn3t.StringVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _magn3t.StringVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _magn3t.StringVector___setitem__(self, *args)

    def pop(self):
        return _magn3t.StringVector_pop(self)

    def append(self, x):
        return _magn3t.StringVector_append(self, x)

    def empty(self):
        return _magn3t.StringVector_empty(self)

    def size(self):
        return _magn3t.StringVector_size(self)

    def swap(self, v):
        return _magn3t.StringVector_swap(self, v)

    def begin(self):
        return _magn3t.StringVector_begin(self)

    def end(self):
        return _magn3t.StringVector_end(self)

    def rbegin(self):
        return _magn3t.StringVector_rbegin(self)

    def rend(self):
        return _magn3t.StringVector_rend(self)

    def clear(self):
        return _magn3t.StringVector_clear(self)

    def get_allocator(self):
        return _magn3t.StringVector_get_allocator(self)

    def pop_back(self):
        return _magn3t.StringVector_pop_back(self)

    def erase(self, *args):
        return _magn3t.StringVector_erase(self, *args)

    def __init__(self, *args):
        this = _magn3t.new_StringVector(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def push_back(self, x):
        return _magn3t.StringVector_push_back(self, x)

    def front(self):
        return _magn3t.StringVector_front(self)

    def back(self):
        return _magn3t.StringVector_back(self)

    def assign(self, n, x):
        return _magn3t.StringVector_assign(self, n, x)

    def resize(self, *args):
        return _magn3t.StringVector_resize(self, *args)

    def insert(self, *args):
        return _magn3t.StringVector_insert(self, *args)

    def reserve(self, n):
        return _magn3t.StringVector_reserve(self, n)

    def capacity(self):
        return _magn3t.StringVector_capacity(self)
    __swig_destroy__ = _magn3t.delete_StringVector
    __del__ = lambda self: None
StringVector_swigregister = _magn3t.StringVector_swigregister
StringVector_swigregister(StringVector)

class FloatVector(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, FloatVector, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, FloatVector, name)
    __repr__ = _swig_repr

    def iterator(self):
        return _magn3t.FloatVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _magn3t.FloatVector___nonzero__(self)

    def __bool__(self):
        return _magn3t.FloatVector___bool__(self)

    def __len__(self):
        return _magn3t.FloatVector___len__(self)

    def __getslice__(self, i, j):
        return _magn3t.FloatVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _magn3t.FloatVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _magn3t.FloatVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _magn3t.FloatVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _magn3t.FloatVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _magn3t.FloatVector___setitem__(self, *args)

    def pop(self):
        return _magn3t.FloatVector_pop(self)

    def append(self, x):
        return _magn3t.FloatVector_append(self, x)

    def empty(self):
        return _magn3t.FloatVector_empty(self)

    def size(self):
        return _magn3t.FloatVector_size(self)

    def swap(self, v):
        return _magn3t.FloatVector_swap(self, v)

    def begin(self):
        return _magn3t.FloatVector_begin(self)

    def end(self):
        return _magn3t.FloatVector_end(self)

    def rbegin(self):
        return _magn3t.FloatVector_rbegin(self)

    def rend(self):
        return _magn3t.FloatVector_rend(self)

    def clear(self):
        return _magn3t.FloatVector_clear(self)

    def get_allocator(self):
        return _magn3t.FloatVector_get_allocator(self)

    def pop_back(self):
        return _magn3t.FloatVector_pop_back(self)

    def erase(self, *args):
        return _magn3t.FloatVector_erase(self, *args)

    def __init__(self, *args):
        this = _magn3t.new_FloatVector(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def push_back(self, x):
        return _magn3t.FloatVector_push_back(self, x)

    def front(self):
        return _magn3t.FloatVector_front(self)

    def back(self):
        return _magn3t.FloatVector_back(self)

    def assign(self, n, x):
        return _magn3t.FloatVector_assign(self, n, x)

    def resize(self, *args):
        return _magn3t.FloatVector_resize(self, *args)

    def insert(self, *args):
        return _magn3t.FloatVector_insert(self, *args)

    def reserve(self, n):
        return _magn3t.FloatVector_reserve(self, n)

    def capacity(self):
        return _magn3t.FloatVector_capacity(self)
    __swig_destroy__ = _magn3t.delete_FloatVector
    __del__ = lambda self: None
FloatVector_swigregister = _magn3t.FloatVector_swigregister
FloatVector_swigregister(FloatVector)

class VectorOfDoubleVector(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, VectorOfDoubleVector, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, VectorOfDoubleVector, name)
    __repr__ = _swig_repr

    def iterator(self):
        return _magn3t.VectorOfDoubleVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _magn3t.VectorOfDoubleVector___nonzero__(self)

    def __bool__(self):
        return _magn3t.VectorOfDoubleVector___bool__(self)

    def __len__(self):
        return _magn3t.VectorOfDoubleVector___len__(self)

    def __getslice__(self, i, j):
        return _magn3t.VectorOfDoubleVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _magn3t.VectorOfDoubleVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _magn3t.VectorOfDoubleVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _magn3t.VectorOfDoubleVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _magn3t.VectorOfDoubleVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _magn3t.VectorOfDoubleVector___setitem__(self, *args)

    def pop(self):
        return _magn3t.VectorOfDoubleVector_pop(self)

    def append(self, x):
        return _magn3t.VectorOfDoubleVector_append(self, x)

    def empty(self):
        return _magn3t.VectorOfDoubleVector_empty(self)

    def size(self):
        return _magn3t.VectorOfDoubleVector_size(self)

    def swap(self, v):
        return _magn3t.VectorOfDoubleVector_swap(self, v)

    def begin(self):
        return _magn3t.VectorOfDoubleVector_begin(self)

    def end(self):
        return _magn3t.VectorOfDoubleVector_end(self)

    def rbegin(self):
        return _magn3t.VectorOfDoubleVector_rbegin(self)

    def rend(self):
        return _magn3t.VectorOfDoubleVector_rend(self)

    def clear(self):
        return _magn3t.VectorOfDoubleVector_clear(self)

    def get_allocator(self):
        return _magn3t.VectorOfDoubleVector_get_allocator(self)

    def pop_back(self):
        return _magn3t.VectorOfDoubleVector_pop_back(self)

    def erase(self, *args):
        return _magn3t.VectorOfDoubleVector_erase(self, *args)

    def __init__(self, *args):
        this = _magn3t.new_VectorOfDoubleVector(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def push_back(self, x):
        return _magn3t.VectorOfDoubleVector_push_back(self, x)

    def front(self):
        return _magn3t.VectorOfDoubleVector_front(self)

    def back(self):
        return _magn3t.VectorOfDoubleVector_back(self)

    def assign(self, n, x):
        return _magn3t.VectorOfDoubleVector_assign(self, n, x)

    def resize(self, *args):
        return _magn3t.VectorOfDoubleVector_resize(self, *args)

    def insert(self, *args):
        return _magn3t.VectorOfDoubleVector_insert(self, *args)

    def reserve(self, n):
        return _magn3t.VectorOfDoubleVector_reserve(self, n)

    def capacity(self):
        return _magn3t.VectorOfDoubleVector_capacity(self)
    __swig_destroy__ = _magn3t.delete_VectorOfDoubleVector
    __del__ = lambda self: None
VectorOfDoubleVector_swigregister = _magn3t.VectorOfDoubleVector_swigregister
VectorOfDoubleVector_swigregister(VectorOfDoubleVector)


def random_double(a, b):
    return _magn3t.random_double(a, b)
random_double = _magn3t.random_double

def MRCtoVector(fileName):
    return _magn3t.MRCtoVector(fileName)
MRCtoVector = _magn3t.MRCtoVector
class cubee(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, cubee, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, cubee, name)
    __repr__ = _swig_repr
    __swig_setmethods__["Nx"] = _magn3t.cubee_Nx_set
    __swig_getmethods__["Nx"] = _magn3t.cubee_Nx_get
    if _newclass:
        Nx = _swig_property(_magn3t.cubee_Nx_get, _magn3t.cubee_Nx_set)
    __swig_setmethods__["Ny"] = _magn3t.cubee_Ny_set
    __swig_getmethods__["Ny"] = _magn3t.cubee_Ny_get
    if _newclass:
        Ny = _swig_property(_magn3t.cubee_Ny_get, _magn3t.cubee_Ny_set)
    __swig_setmethods__["Nz"] = _magn3t.cubee_Nz_set
    __swig_getmethods__["Nz"] = _magn3t.cubee_Nz_get
    if _newclass:
        Nz = _swig_property(_magn3t.cubee_Nz_get, _magn3t.cubee_Nz_set)
    __swig_setmethods__["N_tot"] = _magn3t.cubee_N_tot_set
    __swig_getmethods__["N_tot"] = _magn3t.cubee_N_tot_get
    if _newclass:
        N_tot = _swig_property(_magn3t.cubee_N_tot_get, _magn3t.cubee_N_tot_set)
    __swig_setmethods__["xx"] = _magn3t.cubee_xx_set
    __swig_getmethods__["xx"] = _magn3t.cubee_xx_get
    if _newclass:
        xx = _swig_property(_magn3t.cubee_xx_get, _magn3t.cubee_xx_set)
    __swig_setmethods__["Volume"] = _magn3t.cubee_Volume_set
    __swig_getmethods__["Volume"] = _magn3t.cubee_Volume_get
    if _newclass:
        Volume = _swig_property(_magn3t.cubee_Volume_get, _magn3t.cubee_Volume_set)
    __swig_setmethods__["Surface"] = _magn3t.cubee_Surface_set
    __swig_getmethods__["Surface"] = _magn3t.cubee_Surface_get
    if _newclass:
        Surface = _swig_property(_magn3t.cubee_Surface_get, _magn3t.cubee_Surface_set)
    __swig_setmethods__["Val"] = _magn3t.cubee_Val_set
    __swig_getmethods__["Val"] = _magn3t.cubee_Val_get
    if _newclass:
        Val = _swig_property(_magn3t.cubee_Val_get, _magn3t.cubee_Val_set)

    def modulo(self, *args):
        return _magn3t.cubee_modulo(self, *args)

    def index(self, i, j, k):
        return _magn3t.cubee_index(self, i, j, k)

    def vecini(self, centru, vecX, vecY, vecZ):
        return _magn3t.cubee_vecini(self, centru, vecX, vecY, vecZ)

    def index3D(self, ind):
        return _magn3t.cubee_index3D(self, ind)

    def test(self):
        return _magn3t.cubee_test(self)

    def index2(self, i, j, k):
        return _magn3t.cubee_index2(self, i, j, k)

    def __init__(self, *args):
        this = _magn3t.new_cubee(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def close(self):
        return _magn3t.cubee_close(self)

    def getVal(self):
        return _magn3t.cubee_getVal(self)

    def getSafe(self, i1, i2, i3):
        return _magn3t.cubee_getSafe(self, i1, i2, i3)

    def get(self, *args):
        return _magn3t.cubee_get(self, *args)

    def set(self, *args):
        return _magn3t.cubee_set(self, *args)

    def setSafe(self, i1, i2, i3, val):
        return _magn3t.cubee_setSafe(self, i1, i2, i3, val)

    def normalize(self):
        return _magn3t.cubee_normalize(self)

    def __call__(self, i1, i2, i3):
        return _magn3t.cubee___call__(self, i1, i2, i3)

    def __sub__(self, *args):
        return _magn3t.cubee___sub__(self, *args)

    def __add__(self, *args):
        return _magn3t.cubee___add__(self, *args)

    def __mul__(self, x):
        return _magn3t.cubee___mul__(self, x)

    def __truediv__(self, *args):
        return _magn3t.cubee___truediv__(self, *args)
    __div__ = __truediv__



    def copy(self):
        return _magn3t.cubee_copy(self)

    def erode(self, culoare, n):
        return _magn3t.cubee_erode(self, culoare, n)

    def addSphere(self, x, y, z, raza, smo=0):
        return _magn3t.cubee_addSphere(self, x, y, z, raza, smo)

    def interpolation(self, x, y, z):
        return _magn3t.cubee_interpolation(self, x, y, z)

    def fillInterpolation(self, *args):
        return _magn3t.cubee_fillInterpolation(self, *args)
    __swig_destroy__ = _magn3t.delete_cubee
    __del__ = lambda self: None
cubee_swigregister = _magn3t.cubee_swigregister
cubee_swigregister(cubee)
cvar = _magn3t.cvar


def readMRC(arg1):
    return _magn3t.readMRC(arg1)
readMRC = _magn3t.readMRC

def writeMRC(arg1, arg2):
    return _magn3t.writeMRC(arg1, arg2)
writeMRC = _magn3t.writeMRC

def applyThreshold(c, th):
    return _magn3t.applyThreshold(c, th)
applyThreshold = _magn3t.applyThreshold

def fd(x, r, t):
    return _magn3t.fd(x, r, t)
fd = _magn3t.fd

def volume(*args):
    return _magn3t.volume(*args)
volume = _magn3t.volume

def surface(c, culoare):
    return _magn3t.surface(c, culoare)
surface = _magn3t.surface

def surfaceModificat(c, culoare):
    return _magn3t.surfaceModificat(c, culoare)
surfaceModificat = _magn3t.surfaceModificat

def applyThreshold_linear(*args):
    return _magn3t.applyThreshold_linear(*args)
applyThreshold_linear = _magn3t.applyThreshold_linear

def findValue(c, value):
    return _magn3t.findValue(c, value)
findValue = _magn3t.findValue

def fill(c, i1, i2, i3, target, newcolor):
    return _magn3t.fill(c, i1, i2, i3, target, newcolor)
fill = _magn3t.fill

def fill_particule(c, startColor):
    return _magn3t.fill_particule(c, startColor)
fill_particule = _magn3t.fill_particule

def fill_particule_volume(c, startColor):
    return _magn3t.fill_particule_volume(c, startColor)
fill_particule_volume = _magn3t.fill_particule_volume

def volumeList(c):
    return _magn3t.volumeList(c)
volumeList = _magn3t.volumeList

def fill_particule_volume_2(c, startColor):
    return _magn3t.fill_particule_volume_2(c, startColor)
fill_particule_volume_2 = _magn3t.fill_particule_volume_2

def fillParticlesRandom(c, startColor):
    return _magn3t.fillParticlesRandom(c, startColor)
fillParticlesRandom = _magn3t.fillParticlesRandom

def fillParticlesRandom_2(c, startColor, startColor2):
    return _magn3t.fillParticlesRandom_2(c, startColor, startColor2)
fillParticlesRandom_2 = _magn3t.fillParticlesRandom_2

def erode(c, culoare, n):
    return _magn3t.erode(c, culoare, n)
erode = _magn3t.erode

def dilate(c, culoare, n):
    return _magn3t.dilate(c, culoare, n)
dilate = _magn3t.dilate

def dilate2(c, culoare, n):
    return _magn3t.dilate2(c, culoare, n)
dilate2 = _magn3t.dilate2

def erode2(c, culoare, n):
    return _magn3t.erode2(c, culoare, n)
erode2 = _magn3t.erode2

def print2(a1, Nx, Ny, Nz):
    return _magn3t.print2(a1, Nx, Ny, Nz)
print2 = _magn3t.print2

def unVector(n):
    return _magn3t.unVector(n)
unVector = _magn3t.unVector

def erodeGrayscale(c, n):
    return _magn3t.erodeGrayscale(c, n)
erodeGrayscale = _magn3t.erodeGrayscale

def erodeGrayscaleCheb(c, n):
    return _magn3t.erodeGrayscaleCheb(c, n)
erodeGrayscaleCheb = _magn3t.erodeGrayscaleCheb

def dilateGrayscale(c, n):
    return _magn3t.dilateGrayscale(c, n)
dilateGrayscale = _magn3t.dilateGrayscale

def dilateGrayscaleCheb(c, n):
    return _magn3t.dilateGrayscaleCheb(c, n)
dilateGrayscaleCheb = _magn3t.dilateGrayscaleCheb

def dilateGrayscaleGeneral(c, n):
    return _magn3t.dilateGrayscaleGeneral(c, n)
dilateGrayscaleGeneral = _magn3t.dilateGrayscaleGeneral

def blurGrayscale(c, n):
    return _magn3t.blurGrayscale(c, n)
blurGrayscale = _magn3t.blurGrayscale

def medianFilter(c):
    return _magn3t.medianFilter(c)
medianFilter = _magn3t.medianFilter

def isOk(i, n):
    return _magn3t.isOk(i, n)
isOk = _magn3t.isOk

def distanceMapChessEfficient(c, culoare):
    return _magn3t.distanceMapChessEfficient(c, culoare)
distanceMapChessEfficient = _magn3t.distanceMapChessEfficient

def distanceMapChebEfficient(c, culoare):
    return _magn3t.distanceMapChebEfficient(c, culoare)
distanceMapChebEfficient = _magn3t.distanceMapChebEfficient

def distanceMap(c, culoare):
    return _magn3t.distanceMap(c, culoare)
distanceMap = _magn3t.distanceMap

def distanceMapGeneralEfficient(c, culoare):
    return _magn3t.distanceMapGeneralEfficient(c, culoare)
distanceMapGeneralEfficient = _magn3t.distanceMapGeneralEfficient

def distanceMapGeneralEfficientMare(cu, culoare):
    return _magn3t.distanceMapGeneralEfficientMare(cu, culoare)
distanceMapGeneralEfficientMare = _magn3t.distanceMapGeneralEfficientMare

def distanceMapGeneralEfficientMic(cu, culoare):
    return _magn3t.distanceMapGeneralEfficientMic(cu, culoare)
distanceMapGeneralEfficientMic = _magn3t.distanceMapGeneralEfficientMic

def distanceMapCheb(c, culoare):
    return _magn3t.distanceMapCheb(c, culoare)
distanceMapCheb = _magn3t.distanceMapCheb

def morphologicalReconstruction(marker, mask):
    return _magn3t.morphologicalReconstruction(marker, mask)
morphologicalReconstruction = _magn3t.morphologicalReconstruction

def morphologicalReconstructionHybrid(marker, mask):
    return _magn3t.morphologicalReconstructionHybrid(marker, mask)
morphologicalReconstructionHybrid = _magn3t.morphologicalReconstructionHybrid
class thing(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, thing, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, thing, name)
    __repr__ = _swig_repr
    __swig_setmethods__["pixel"] = _magn3t.thing_pixel_set
    __swig_getmethods__["pixel"] = _magn3t.thing_pixel_get
    if _newclass:
        pixel = _swig_property(_magn3t.thing_pixel_get, _magn3t.thing_pixel_set)
    __swig_setmethods__["valoare"] = _magn3t.thing_valoare_set
    __swig_getmethods__["valoare"] = _magn3t.thing_valoare_get
    if _newclass:
        valoare = _swig_property(_magn3t.thing_valoare_get, _magn3t.thing_valoare_set)

    def __lt__(self, rhs):
        return _magn3t.thing___lt__(self, rhs)

    def __gt__(self, rhs):
        return _magn3t.thing___gt__(self, rhs)

    def __le__(self, rhs):
        return _magn3t.thing___le__(self, rhs)

    def __ge__(self, rhs):
        return _magn3t.thing___ge__(self, rhs)

    def __init__(self):
        this = _magn3t.new_thing()
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this
    __swig_destroy__ = _magn3t.delete_thing
    __del__ = lambda self: None
thing_swigregister = _magn3t.thing_swigregister
thing_swigregister(thing)


def priorityFloodModificat(d, seed):
    return _magn3t.priorityFloodModificat(d, seed)
priorityFloodModificat = _magn3t.priorityFloodModificat

def priorityFlood(*args):
    return _magn3t.priorityFlood(*args)
priorityFlood = _magn3t.priorityFlood

def otsu(*args):
    return _magn3t.otsu(*args)
otsu = _magn3t.otsu

def addSphere(c, x, y, z, raza, smo=0):
    return _magn3t.addSphere(c, x, y, z, raza, smo)
addSphere = _magn3t.addSphere

def euler_angles(rot):
    return _magn3t.euler_angles(rot)
euler_angles = _magn3t.euler_angles

def breit_angles(rot):
    return _magn3t.breit_angles(rot)
breit_angles = _magn3t.breit_angles

def euler_matrix(alfa, beta, gamma):
    return _magn3t.euler_matrix(alfa, beta, gamma)
euler_matrix = _magn3t.euler_matrix

def breit_matrix(alfa, beta, gamma):
    return _magn3t.breit_matrix(alfa, beta, gamma)
breit_matrix = _magn3t.breit_matrix

def addEllipse(cc, x1, x2, x3, a1, a2, a3, alfa, beta, gamma, scala):
    return _magn3t.addEllipse(cc, x1, x2, x3, a1, a2, a3, alfa, beta, gamma, scala)
addEllipse = _magn3t.addEllipse

def addEllipse2(cc, x1, x2, x3, a1, a2, a3, m, scala, color=1):
    return _magn3t.addEllipse2(cc, x1, x2, x3, a1, a2, a3, m, scala, color)
addEllipse2 = _magn3t.addEllipse2

def addEllipse3(cc, x1, x2, x3, a1, a2, a3, alfa, beta, gamma, scala):
    return _magn3t.addEllipse3(cc, x1, x2, x3, a1, a2, a3, alfa, beta, gamma, scala)
addEllipse3 = _magn3t.addEllipse3

def to_vec1(str):
    return _magn3t.to_vec1(str)
to_vec1 = _magn3t.to_vec1

def read_fis(nume_fila):
    return _magn3t.read_fis(nume_fila)
read_fis = _magn3t.read_fis

def addPolygon(*args):
    return _magn3t.addPolygon(*args)
addPolygon = _magn3t.addPolygon

def volumeAnalysis(c):
    return _magn3t.volumeAnalysis(c)
volumeAnalysis = _magn3t.volumeAnalysis

def volumeAnalysisOrientation(c, numeOutput):
    return _magn3t.volumeAnalysisOrientation(c, numeOutput)
volumeAnalysisOrientation = _magn3t.volumeAnalysisOrientation

def getParticle(c, culoare):
    return _magn3t.getParticle(c, culoare)
getParticle = _magn3t.getParticle

def volumeAnalysisOrientationColor(c, color):
    return _magn3t.volumeAnalysisOrientationColor(c, color)
volumeAnalysisOrientationColor = _magn3t.volumeAnalysisOrientationColor

def momentOfInertia(cc, culoare):
    return _magn3t.momentOfInertia(cc, culoare)
momentOfInertia = _magn3t.momentOfInertia

def shapeAnalysisOrientationColor(c, color):
    return _magn3t.shapeAnalysisOrientationColor(c, color)
shapeAnalysisOrientationColor = _magn3t.shapeAnalysisOrientationColor

def main():
    return _magn3t.main()
main = _magn3t.main
# This file is compatible with both classic and new-style classes.


